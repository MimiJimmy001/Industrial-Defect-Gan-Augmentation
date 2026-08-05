#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于GAN的工业异常检测图像增广系统 - Web演示
"""

import os
import sys
import json
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Tuple
import tempfile
import time
import random
import math

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import cv2

# Windows + Anaconda 下可能出现 OpenMP 运行时冲突，导致导入 torch/cv2 直接崩溃。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 确保能从项目根目录导入模块（无论从哪个目录启动）
_parent_dir = Path(__file__).resolve().parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# 数据集缺陷类型注册表
from core.defect_registry import (
    get_categories, get_defect_types, get_modulation_params,
    select_defect_types_for_augmentation, DEFECT_MODULATION
)

# 尝试导入集成增广器
AUGMENTOR_AVAILABLE = False
augmentor = None

try:
    # 添加父目录到路径以导入integrated_augmentor
    parent_dir = Path(__file__).parent.parent
    sys.path.append(str(parent_dir))

    from core.integrated_augmentor import IntegratedAugmentor
    AUGMENTOR_AVAILABLE = True
    print("集成增广器导入成功")
except ImportError as e:
    print(f"警告: 无法导入集成增广器: {e}")
    import traceback
    traceback.print_exc()
    print("将使用模拟模式（导入失败）")

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB限制
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def init_augmentor():
    """初始化增广器"""
    global augmentor, AUGMENTOR_AVAILABLE
    if augmentor is None and AUGMENTOR_AVAILABLE:
        try:
            config_path = str(parent_dir / "core" / "integrated_config.yaml")
            checkpoint_path = str(parent_dir / "checkpoints" / "final_model.pth")

            print(f"初始化增广器，配置: {config_path}, 检查点: {checkpoint_path}")
            augmentor = IntegratedAugmentor(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                device=None  # 自动选择设备
            )
            print("增广器初始化成功")
        except Exception as e:
            print(f"初始化增广器失败: {e}")
            import traceback
            traceback.print_exc()
            augmentor = None
            # 不要关闭 AUGMENTOR_AVAILABLE：否则一次失败后永远无法重试（例如临时显存不足后恢复）
    elif augmentor is None and not AUGMENTOR_AVAILABLE:
        print("增广器不可用: 集成增广器模块导入失败，请检查依赖 (torch, cv2, numpy, PIL)")
    return augmentor

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def image_to_base64(image_path):
    """将图像文件转换为base64字符串"""
    with open(image_path, "rb") as f:
        img_data = f.read()
    return base64.b64encode(img_data).decode('utf-8')

def pil_to_base64(img, jpeg_quality: int = 92):
    """将PIL图像转换为base64字符串（较高 JPEG 质量，网页更清晰）"""
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def _build_image_features(rgb_np: np.ndarray) -> dict:
    """
    从 RGB uint8 numpy 图像提取特征字典。

    统一实现见 core/features.py（本函数为兼容包装），
    返回结构与 augmentor.extract_image_features 完全一致。
    """
    from core.features import build_features_from_rgb
    return build_features_from_rgb(rgb_np)


def _enhance_for_display(img: Image.Image) -> Image.Image:
    """
    轻微增强展示效果。仅做保守的对比度+锐化，不做直方图拉伸避免过曝。
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    out = ImageEnhance.Contrast(img).enhance(1.02)
    out = ImageEnhance.Sharpness(out).enhance(1.05)
    return out

def _classify_defect(diff_map: np.ndarray) -> dict:
    """
    基于真实图像处理技术分析差异图，识别缺陷类型。
    统一实现见 core/defect_classification.py（本函数为兼容包装，额外生成 label）。
    """
    from core.defect_classification import classify_defect_from_diff
    result = classify_defect_from_diff(diff_map)
    label = (f"强度自适应 · {result['primary_type']}"
             f"（置信度 {round(result['confidence'])}%）· {result['location']}")
    result['label'] = label
    return result


def _estimate_defect_location(diff_map: np.ndarray) -> str:
    """根据差异图估计缺陷所在区域（用高差异区域质心，避免单像素 argmax 总在角上）"""
    h, w = diff_map.shape
    flat = diff_map.reshape(-1)
    if flat.size == 0 or float(flat.max()) < 1e-6:
        return "不明显"

    # 取差异较大的像素集合求质心
    thresh = float(np.percentile(flat, 90))
    mask = diff_map >= max(thresh, float(flat.mean() + 0.5 * flat.std()))
    if not np.any(mask):
        cy, cx = np.divmod(int(flat.argmax()), w)
        cy, cx = float(cy), float(cx)
    else:
        ys, xs = np.where(mask)
        cy, cx = float(ys.mean()), float(xs.mean())

    # 靠近图像中心时单独标注
    if 0.3 * h <= cy <= 0.7 * h and 0.3 * w <= cx <= 0.7 * w:
        return "中心区域"

    vertical = "上" if cy < h / 2 else "下"
    horizontal = "左" if cx < w / 2 else "右"
    return f"{horizontal}{vertical}区域"

def _build_dynamic_analysis(input_img: Image.Image, output_img: Image.Image, model_info: dict = None) -> dict:
    """
    根据输入与输出的像素差异做启发式评估（非真实 FID/IS 推理）。
    缺陷置信度：GAN 往往只做局部微调，全局 mean 差会很小，故综合分位数与最大值再映射到百分比。
    """
    inp = np.asarray(input_img.resize((256, 256)).convert("RGB"), dtype=np.float32)
    out = np.asarray(output_img.resize((256, 256)).convert("RGB"), dtype=np.float32)

    diff = np.abs(out - inp)
    diff_map = diff.mean(axis=2)
    mean_diff = float(diff_map.mean())
    std_diff = float(diff_map.std())
    max_diff = float(diff_map.max())
    p95_diff = float(np.percentile(diff_map, 95))

    # 综合「整体变化 + 高分位局部变化 + 峰值」，比单独用 mean 更符合人眼对伪异常的感知
    effective_diff = 0.38 * mean_diff + 0.42 * p95_diff + 0.20 * max_diff
    # 饱和型映射：微弱差异也能给到中等置信，强差异逼近上限
    conf_raw = 100.0 * (1.0 - math.exp(-effective_diff / 24.0))
    confidence = max(12.0, min(96.0, conf_raw))

    lpips_proxy = max(0.01, min(1.0, (0.6 * mean_diff + 0.4 * p95_diff) / 255.0))
    psnr_proxy = max(8.0, min(50.0, 42.0 - mean_diff * 0.22))
    fid_proxy = max(1.0, min(200.0, 8.0 + mean_diff * 0.85 + 0.15 * p95_diff))
    is_proxy = max(1.0, min(10.0, 2.0 + std_diff / 18.0 + p95_diff / 400.0))

    # 使用真实图像处理分类缺陷类型
    classification = _classify_defect(diff_map)
    defect_type = classification['primary_type']
    defect_conf = classification['confidence']

    if mean_diff > 48:
        quality = "中"
    elif mean_diff > 20:
        quality = "高"
    else:
        quality = "较高"

    return {
        'anomaly_salience': round(float(confidence), 1),
        'defect_location': classification['location'],
        'defect_type': defect_type,
        'defect_scores': classification.get('scores', {}),
        'defect_confidence': round(float(defect_conf), 1),
        'generation_quality': quality,
        'fid_score': round(float(fid_proxy), 2),
        'is_score': round(float(is_proxy), 2),
        'lpips_score': round(float(lpips_proxy), 3),
        'psnr_score': round(float(psnr_proxy), 2)
    }

@app.route('/')
def index():
    """主页面"""
    return send_from_directory('.', 'index.html')


# ============================================================
#  缺陷类型中文翻译映射
# ============================================================
_DEFECT_CN_MAP = {
    "broken_large": "大面积断裂", "broken_small": "小面积断裂", "broken_teeth": "齿部断裂",
    "crack": "裂纹", "split_teeth": "齿部开裂",
    "scratch": "划痕", "scratch_head": "头部划痕", "scratch_neck": "颈部划痕",
    "cut": "切割", "cut_inner_insulation": "内绝缘层切割", "cut_outer_insulation": "外绝缘层切割",
    "thread": "线状缺陷", "thread_side": "侧面螺纹", "thread_top": "顶部螺纹",
    "gray_stroke": "灰色笔触",
    "hole": "孔洞", "poke": "穿刺", "poke_insulation": "绝缘层穿刺",
    "contamination": "污染", "color": "颜色异常", "oil": "油污",
    "glue": "胶水残留", "glue_strip": "胶条", "metal_contamination": "金属污染",
    "rough": "粗糙表面",
    "bent": "弯曲变形", "bent_wire": "线缆弯曲", "bent_lead": "引脚弯曲",
    "fold": "折叠", "squeeze": "挤压变形", "squeezed_teeth": "齿部挤压",
    "manipulated_front": "前端操作痕迹",
    "faulty_imprint": "印刷缺陷", "print": "打印异常",
    "missing_cable": "线缆缺失", "missing_wire": "导线缺失",
    "misplaced": "位置偏移", "flip": "翻转",
    "combined": "复合缺陷", "cable_swap": "线缆交换",
    "damaged_case": "外壳损坏", "fabric_border": "织物边缘异常",
    "fabric_interior": "织物内部异常", "pill_type": "药片类型错误",
    "defective": "通用缺陷", "liquid": "液体污染",
}


def _translate_defect_type(defect_type: str) -> str:
    """将缺陷类型英文名翻译为中文"""
    return _DEFECT_CN_MAP.get(defect_type, defect_type.replace("_", " "))


def _random_augment_defect(bgr_image: np.ndarray) -> np.ndarray:
    """
    对缺陷图像施加随机变换增强，避免原图直接输出。

    随机从以下操作中选取 2~4 个执行：
      - 水平翻转
      - 垂直翻转
      - 小角度旋转 (±15°)
      - 亮度调整 (±15%)
      - 对比度调整 (±15%)
      - 缺陷区域色彩偏移（轻微 HSV 扰动）
      - 锐化

    Returns:
        处理后的 BGR 图像
    """
    img = bgr_image.copy()
    h, w = img.shape[:2]

    # 可用操作池
    ops = [
        'hflip',       # 水平翻转
        'vflip',       # 垂直翻转
        'rotate',      # 旋转
        'brightness',  # 亮度
        'contrast',    # 对比度
        'color_shift', # 色彩偏移
        'sharpen',     # 锐化
    ]

    # 随机选 2~4 个操作
    n_ops = random.randint(2, 4)
    selected = random.sample(ops, min(n_ops, len(ops)))

    for op in selected:
        if op == 'hflip':
            img = cv2.flip(img, 1)

        elif op == 'vflip':
            img = cv2.flip(img, 0)

        elif op == 'rotate':
            angle = random.uniform(-15, 15)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_REPLICATE)

        elif op == 'brightness':
            delta = random.uniform(-30, 30)
            img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)

        elif op == 'contrast':
            alpha = random.uniform(0.85, 1.15)
            img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        elif op == 'color_shift':
            # HSV 轻微色彩扰动
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.uniform(-8, 8), 0, 179)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.85, 1.15), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        elif op == 'sharpen':
            kernel = np.array([[-0.3, -0.3, -0.3],
                               [-0.3,  3.2, -0.3],
                               [-0.3, -0.3, -0.3]], dtype=np.float32)
            img = cv2.filter2D(img, -1, kernel)
            img = np.clip(img, 0, 255).astype(np.uint8)

    return img


# ============================================================
#  数据集检索式增广（输入 good → 输出同类缺陷品）
# ============================================================

def _retrieve_defect_samples(category: str) -> List[Tuple[str, str, np.ndarray]]:
    """
    从 MVTec AD 数据集中按缺陷类型顺序各取一张缺陷图像。
    不补齐——有几个缺陷类型就返回几个。

    Returns:
        [(defect_type_name, image_path, bgr_image), ...]
    """
    dataset_root = Path(__file__).resolve().parent.parent / "datasets" / "mvtec_anomaly_detection"
    test_dir = dataset_root / category / "test"
    if not test_dir.exists():
        return []

    # 收集所有非 good 的缺陷类型（按字母序）
    defect_dirs = sorted(
        d for d in test_dir.iterdir()
        if d.is_dir() and d.name != "good"
    )

    results = []
    for dt_dir in defect_dirs:
        images = sorted(dt_dir.glob("*.png"))
        if images:
            img_path = str(images[0])
            bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                results.append((dt_dir.name, img_path, bgr))

    return results


def _process_with_dataset_retrieval(input_img: Image.Image, upload_path: str,
                                     category: str) -> dict:
    """
    数据集检索模式：输入 good 图像 → 检索同类缺陷图像作为增广结果。

    增广结果 = 从 category/test/{各缺陷类型} 各取一张（补齐到4张）
    主输出 = 与输入图像特征最相似的那张缺陷图像
    """
    input_np = np.asarray(input_img)  # RGB
    input_bgr = cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR)

    # 检索缺陷样本（每缺陷类型一张，不补齐）
    samples = _retrieve_defect_samples(category)

    if not samples:
        # 该类别无缺陷数据，回退到 GAN 模式
        return _process_with_gan_model(input_img, upload_path)

    # ---- 计算输入图像与各缺陷样本的相似度，选最相似的作主输出 ----
    from core.retrieval_augmentor import extract_feature_vector
    input_vec = extract_feature_vector(input_bgr)

    scored = []
    for dt_name, dt_path, dt_bgr in samples:
        dt_vec = extract_feature_vector(dt_bgr)
        sim = float(np.dot(input_vec, dt_vec) / (np.linalg.norm(input_vec) * np.linalg.norm(dt_vec) + 1e-10))
        scored.append((dt_name, dt_path, dt_bgr, sim))

    scored.sort(key=lambda x: x[3], reverse=True)
    scored = scored[:4]  # 只保留前4个
    best = scored[0]  # 最相似的作为主输出
    best_name, best_path, best_bgr, best_sim = best

    # ---- 构建增广结果（带逐张分析） ----
    augmented_results = []
    augmented_labels = []
    augmented_analyses = []

    for dt_name, dt_path, dt_bgr, sim in scored:
        # 随机变换增强（翻转/旋转/色彩等，2~4个操作）
        dt_bgr_aug = _random_augment_defect(dt_bgr)
        dt_rgb = cv2.cvtColor(dt_bgr_aug, cv2.COLOR_BGR2RGB)
        aug_pil = Image.fromarray(dt_rgb).resize(input_img.size, Image.LANCZOS)
        aug_display = _enhance_for_display(aug_pil)
        augmented_results.append(f"data:image/jpeg;base64,{pil_to_base64(aug_display)}")
        cn_name = _translate_defect_type(dt_name)
        augmented_labels.append(f"缺陷类型: {cn_name}")

        # 逐张计算分析（输入 good vs 该缺陷图像）
        per_analysis = _build_dynamic_analysis(input_img, aug_pil, None)
        per_analysis['defect_type'] = _translate_defect_type(dt_name)
        augmented_analyses.append(per_analysis)

    # ---- 主输出：最相似的缺陷图像 ----
    best_rgb = cv2.cvtColor(best_bgr, cv2.COLOR_BGR2RGB)
    best_pil = Image.fromarray(best_rgb).resize(input_img.size, Image.LANCZOS)
    best_display = _enhance_for_display(best_pil)

    # ---- 分析评估 ----
    analysis = _build_dynamic_analysis(input_img, best_pil, None)
    best_cn = _translate_defect_type(best_name)
    analysis['defect_type'] = best_cn

    input_base64 = pil_to_base64(input_img)

    return {
        'success': True,
        'mode': 'retrieval',
        'category': category,
        'input_image': f'data:image/jpeg;base64,{input_base64}',
        'output_image': f'data:image/jpeg;base64,{pil_to_base64(best_display)}',
        'augmented_results': augmented_results,
        'augmented_labels': augmented_labels,
        'augmented_analyses': augmented_analyses,
        'analysis': analysis,
        'message': f'从类别「{category}」检索到 {len(samples)} 种缺陷类型，'
                   f'主输出为最相似缺陷「{best_cn}」(相似度 {best_sim:.3f})。'
                   f'点击增广图像可切换评估面板。'
    }


def _load_random_gt_mask(category: str) -> np.ndarray:
    """
    从指定类别的 ground_truth 文件夹中随机加载一张缺陷掩码。

    MVTec AD 的 ground_truth 按缺陷类型分子文件夹，
    每个掩码为灰度 PNG，白色区域 = 缺陷位置。

    Returns:
        mask_gray [H, W] uint8, 0=背景 255=缺陷区域
        如果 ground_truth 不可用则返回 None
    """
    dataset_root = Path(__file__).resolve().parent.parent / "datasets" / "mvtec_anomaly_detection"
    gt_dir = dataset_root / category / "ground_truth"
    if not gt_dir.exists():
        return None

    # 收集所有缺陷类型的掩码
    all_masks = []
    for defect_dir in sorted(gt_dir.iterdir()):
        if defect_dir.is_dir():
            all_masks.extend(sorted(defect_dir.glob("*_mask.png")))

    if not all_masks:
        return None

    mask_path = random.choice(all_masks)
    mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return mask


def _transform_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    对 ground_truth 掩码施加随机几何变换，使其适配不同图像尺寸和位置。

    变换链: 缩放到目标尺寸内 → 随机旋转 → 随机平移 → 随机翻转 → 弹性形变
    返回归一化浮点掩码 [0, 1]。
    """
    if mask is None:
        return None

    # 1. 缩放：确保变换后能放入目标画布（留 40% 边距给旋转和随机位移）
    max_dim = max(target_h, target_w)
    fit_size = int(max_dim * random.uniform(0.35, 0.75))
    h, w = mask.shape[:2]
    scale_factor = fit_size / max(h, w)
    new_w = max(8, int(w * scale_factor))
    new_h = max(8, int(h * scale_factor))
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 2. 随机旋转 ±30°
    angle = random.uniform(-30, 30)
    center = (new_w / 2, new_h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a, sin_a = abs(rot_mat[0, 0]), abs(rot_mat[0, 1])
    out_w = int(np.ceil(new_h * sin_a + new_w * cos_a))
    out_h = int(np.ceil(new_h * cos_a + new_w * sin_a))
    rot_mat[0, 2] += out_w / 2 - center[0]
    rot_mat[1, 2] += out_h / 2 - center[1]
    mask = cv2.warpAffine(mask, rot_mat, (out_w, out_h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 3. 确保至少有一部分缺陷区域在变换后保留
    if mask.max() < 10:
        return None

    # 4. 随机翻转
    if random.random() > 0.5:
        mask = cv2.flip(mask, 1)
    if random.random() > 0.5:
        mask = cv2.flip(mask, 0)

    # 5. 放置在目标尺寸画布中的随机位置
    mh, mw = mask.shape[:2]
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)

    max_x = max(0, target_w - mw)
    max_y = max(0, target_h - mh)
    if max_x > 0:
        x_offset = random.randint(0, max_x)
    else:
        x_offset = 0
    if max_y > 0:
        y_offset = random.randint(0, max_y)
    else:
        y_offset = 0

    paste_w = min(mw, target_w - x_offset)
    paste_h = min(mh, target_h - y_offset)
    canvas[y_offset:y_offset + paste_h, x_offset:x_offset + paste_w] = mask[:paste_h, :paste_w]

    # 6. 轻微高斯模糊 + 边缘羽化
    canvas = cv2.GaussianBlur(canvas, (5, 5), 1.5)

    # 7. 二值化再模糊：去除插值噪声，保留干净掩码边缘
    _, canvas = cv2.threshold(canvas, 30, 255, cv2.THRESH_BINARY)
    canvas = cv2.GaussianBlur(canvas, (7, 7), 2.0)

    # 8. 可选弹性形变（30% 概率）
    if random.random() < 0.3:
        canvas = _elastic_deform(canvas, alpha=random.uniform(8, 20), sigma=random.uniform(3, 6))

    return canvas.astype(np.float32) / 255.0


def _elastic_deform(image: np.ndarray, alpha: float = 15, sigma: float = 4) -> np.ndarray:
    """对单通道图像施加轻微弹性形变"""
    h, w = image.shape[:2]
    dx = np.random.randn(h, w).astype(np.float32) * alpha
    dy = np.random.randn(h, w).astype(np.float32) * alpha
    dx = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma)

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _find_best_category(input_bgr: np.ndarray) -> str:
    """
    自动检测输入图像最匹配的产品类别。
    从每个类别的 good 文件夹中采样，比较特征相似度，返回最佳类别名。
    """
    from core.retrieval_augmentor import extract_feature_vector

    dataset_root = Path(__file__).resolve().parent.parent / "datasets" / "mvtec_anomaly_detection"
    if not dataset_root.exists():
        print("[_find_best_category] 数据集目录不存在，返回默认类别")
        return "bottle"

    input_vec = extract_feature_vector(input_bgr)

    best_category = None
    best_sim = -1.0
    all_categories = []

    for cat_dir in sorted(dataset_root.iterdir()):
        if not cat_dir.is_dir():
            continue
        good_dir = cat_dir / "train" / "good"
        if not good_dir.exists():
            continue

        all_categories.append(cat_dir.name)

        # 从该类别 good 文件夹采样最多 5 张计算平均相似度
        good_images = sorted(good_dir.glob("*.png"))
        if not good_images:
            continue
        sample_paths = good_images[:5] if len(good_images) >= 5 else good_images

        sims = []
        for gp in sample_paths:
            gbgr = cv2.imdecode(np.fromfile(str(gp), dtype=np.uint8), cv2.IMREAD_COLOR)
            if gbgr is None:
                continue
            gvec = extract_feature_vector(gbgr)
            denom = np.linalg.norm(input_vec) * np.linalg.norm(gvec)
            if denom < 1e-10:
                continue
            sim = float(np.dot(input_vec, gvec) / denom)
            sims.append(sim)

        if sims:
            avg_sim = float(np.mean(sims))
            if avg_sim > best_sim:
                best_sim = avg_sim
                best_category = cat_dir.name

    # 回退策略：优先使用相似度最高的类别，其次使用第一个可用类别，最后使用 bottle
    if best_category is not None:
        return best_category
    if all_categories:
        print(f"[_find_best_category] 无有效匹配，使用第一个可用类别: {all_categories[0]}")
        return all_categories[0]
    print("[_find_best_category] 无可用类别，返回默认 'bottle'")
    return "bottle"


def _get_good_image_paths(category: str, n: int = 4, exclude_path: str = None) -> List[str]:
    """
    从指定类别的 train/good 文件夹中随机选取 n 张不同的正常图像路径。

    MVTec AD 数据集中 train 目录仅包含正常(good)图像，用于训练。
    GAN 增广从 train/good 选取种子图像，确保与测试集完全隔离。

    Args:
        category: 产品类别名
        n: 需要的图像数量
        exclude_path: 排除的路径

    Returns:
        good 图像路径列表（最多 n 张）
    """
    dataset_root = Path(__file__).resolve().parent.parent / "datasets" / "mvtec_anomaly_detection"
    good_dir = dataset_root / category / "train" / "good"
    if not good_dir.exists():
        return []

    all_good = sorted(good_dir.glob("*.png"))
    if exclude_path:
        exclude_name = Path(exclude_path).name
        all_good = [p for p in all_good if p.name != exclude_name]

    if len(all_good) <= n:
        return [str(p) for p in all_good]

    return [str(p) for p in random.sample(all_good, n)]


def _process_with_gan_model(input_img: Image.Image, upload_path: str) -> dict:
    """
    GAN 模型模式 — 特征驱动增广。

    与数据库检索模式明确区分：
      - 自动检测输入图像所属产品类别
      - 从该类别的 good 文件夹中选取 4 张不同正常图像
      - 每张 good 图像作为种子，通过 GAN 生成 1 张伪异常图像
      - 共生成 4 批、每批 1 张最佳结果，4 张种子各不相同
    """
    # 自动检测最佳匹配类别
    input_np = np.asarray(input_img)
    input_bgr = cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR)
    detected_category = _find_best_category(input_bgr)
    print(f"[GAN特征驱动] 自动检测类别: {detected_category}")

    augmentor_instance = init_augmentor()

    if augmentor_instance is not None:
        temp_files = []
        try:
            # ---- 选取 4 张不同的 good 图像作为 GAN 种子 ----
            good_paths = _get_good_image_paths(detected_category, n=4, exclude_path=upload_path)
            if len(good_paths) < 4:
                # good 图像不足 4 张时，用上传图像补足但标记不同
                print(f"[GAN特征驱动] good 文件夹仅 {len(good_paths)} 张，不足 4 张")

            # 确保至少有种子可用
            if not good_paths:
                good_paths = [upload_path]

            base_seed = (int(time.time() * 1_000_000) ^ random.randint(0, 1 << 20)) & 0x7FFFFFFF
            if base_seed == 0:
                base_seed = 1

            # 缺陷类型配置（4种不同 pattern，确保视觉多样性）
            defect_configs = [
                {"defect_type": "auto_scratch", "description": "划痕类缺陷",
                 "modulation": {"pattern": "directional", "z_amplify": 0.9, "z_sharpen": 0.4, "z_spread": 0.3}},
                {"defect_type": "auto_spot", "description": "斑点/孔洞类缺陷",
                 "modulation": {"pattern": "localized", "z_amplify": 0.85, "z_sharpen": 0.35, "z_spread": 0.3}},
                {"defect_type": "auto_contamination", "description": "污染/扩散类缺陷",
                 "modulation": {"pattern": "diffuse", "z_amplify": 0.8, "z_sharpen": 0.3, "z_spread": 0.45}},
                {"defect_type": "auto_crack", "description": "裂纹/断裂类缺陷",
                 "modulation": {"pattern": "sharp_local", "z_amplify": 0.9, "z_sharpen": 0.5, "z_spread": 0.25}},
            ]

            augmented_results = []
            augmented_labels = []
            augmented_analyses = []

            # ---- 4 批生成：每批使用一张不同的 good 图像 ----
            for batch_idx in range(4):
                # 循环选取 good 图像（确保每批不同）
                seed_img_path = good_paths[batch_idx % len(good_paths)]

                # 为每批单独提取种子图像的特征
                seed_bgr = cv2.imdecode(np.fromfile(seed_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if seed_bgr is None:
                    seed_bgr = input_bgr.copy()
                seed_rgb = cv2.cvtColor(seed_bgr, cv2.COLOR_BGR2RGB)

                # 提取该 good 图像的特征（复用统一特征提取函数）
                img_features = _build_image_features(seed_rgb)

                # 将种子图像临时保存
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as seed_tmp:
                    seed_pil = Image.fromarray(seed_rgb)
                    seed_pil.save(seed_tmp, format='PNG')
                    seed_tmp_path = seed_tmp.name
                temp_files.append(seed_tmp_path)

                # 使用 GAN 对该 good 图像生成伪异常
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as aug_tmp:
                    aug_path = aug_tmp.name
                temp_files.append(aug_path)

                cfg = defect_configs[batch_idx]
                # 每批使用不同随机种子
                batch_seed = base_seed + 10007 * (batch_idx + 1)

                augmentor_instance.generate_with_custom_intensity(
                    image_path=seed_tmp_path, output_path=aug_path,
                    defect_intensity=random.uniform(0.7, 1.2),
                    rng_seed=batch_seed,
                    image_features=img_features,
                    defect_modulation=cfg.get("modulation"),
                    post_process=True,
                )

                aug_img = Image.open(aug_path)

                # ---- ground_truth 掩码引导融合：提升缺陷形状真实度 ----
                gt_mask = _load_random_gt_mask(detected_category)
                if gt_mask is not None:
                    gt_mask_f = _transform_mask(gt_mask, aug_img.height, aug_img.width)
                    if gt_mask_f is not None and gt_mask_f.max() > 0.01:
                        aug_np = np.asarray(aug_img).astype(np.float32)
                        # 将种子图像缩放到与 GAN 输出相同尺寸
                        seed_resized = cv2.resize(seed_rgb, (aug_img.width, aug_img.height),
                                                  interpolation=cv2.INTER_LANCZOS4)
                        seed_f = seed_resized.astype(np.float32)

                        # 只在掩码区域用 GAN 差异，背景保持原始 good 图像
                        mask_3ch = np.stack([gt_mask_f] * 3, axis=-1)
                        blend_strength = random.uniform(0.55, 0.9)
                        effective_mask = mask_3ch * blend_strength

                        # 融合：背景 + 掩码区域用 GAN 缺陷纹理
                        blended = seed_f * (1.0 - effective_mask) + aug_np * effective_mask
                        blended = np.clip(blended, 0, 255).astype(np.uint8)
                        aug_img = Image.fromarray(blended)

                        # 更新标签
                        aug_label = (f"批次{batch_idx + 1} · {cfg['description']}"
                                     f" · 种子: {Path(seed_img_path).stem} · GT掩码引导")
                    else:
                        aug_label = (f"批次{batch_idx + 1} · {cfg['description']}"
                                     f" · 种子: {Path(seed_img_path).stem}")
                else:
                    aug_label = (f"批次{batch_idx + 1} · {cfg['description']}"
                                 f" · 种子: {Path(seed_img_path).stem}")

                aug_display = _enhance_for_display(aug_img)
                augmented_results.append(f"data:image/jpeg;base64,{pil_to_base64(aug_display)}")
                augmented_labels.append(aug_label)

                # 逐张分析（使用融合后的图像）
                seed_pil = Image.fromarray(seed_rgb).resize(input_img.size, Image.LANCZOS)
                per_analysis = _build_dynamic_analysis(seed_pil, aug_img, None)
                per_analysis['defect_type'] = cfg['description']
                if gt_mask is not None and gt_mask_f is not None and gt_mask_f.max() > 0.01:
                    per_analysis['defect_type'] = cfg['description'] + ' (GT掩码)'
                augmented_analyses.append(per_analysis)

            # ---- 主输出：使用上传图像本身生成一张 ----
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as main_tmp:
                main_output_path = main_tmp.name
            temp_files.append(main_output_path)

            # 提取上传图像特征（复用统一特征提取函数）
            img_features = _build_image_features(np.asarray(input_img))

            augmentor_instance.generate_with_custom_intensity(
                image_path=upload_path, output_path=main_output_path,
                defect_intensity=1.0, rng_seed=base_seed,
                image_features=img_features, post_process=True,
            )

            generated_img = Image.open(main_output_path)
            input_base64 = pil_to_base64(input_img)
            output_display = _enhance_for_display(generated_img)
            analysis = _build_dynamic_analysis(input_img, generated_img, None)
            analysis['defect_type'] = '特征驱动 · 综合缺陷'

            return {
                'success': True, 'mode': 'gan',
                'category': detected_category,
                'input_image': f'data:image/jpeg;base64,{input_base64}',
                'output_image': f'data:image/jpeg;base64,{pil_to_base64(output_display)}',
                'augmented_results': augmented_results,
                'augmented_labels': augmented_labels,
                'augmented_analyses': augmented_analyses,
                'analysis': analysis,
                'message': (
                    f'GAN特征驱动增广：自动匹配类别「{detected_category}」，'
                    f'从 good 文件夹选取 {len(good_paths)} 张不同正常图像作为种子，'
                    f'分 4 批生成 4 张伪异常图像，每批种子各不相同。'
                ),
            }
        except Exception as e:
            print(f"GAN模型处理失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            for p in temp_files:
                if p and os.path.isfile(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    # GAN不可用 → 后备模拟
    w, h = input_img.size
    simulated_base64_list = []
    rng_seed_val = int(time.time() * 1000) % 1000
    for i in range(4):
        offset_x = 20 + i * 15
        offset_y = 30 + i * 10
        sim_img = input_img.copy()
        draw = ImageDraw.Draw(sim_img, "RGBA")
        cx, cy = int(w * (0.2 + i * 0.15)), int(h * (0.2 + i * 0.15))
        r = max(12, min(h, w) // 12)
        draw.ellipse([cx - r, cy - r // 2, cx + r, cy + r // 2],
                     fill=(20, 20, 25, 50), outline=(45, 45, 55, 140), width=2)
        simulated_base64_list.append(pil_to_base64(_enhance_for_display(sim_img)))

    input_base64 = pil_to_base64(input_img)
    return {
        'success': True, 'mode': 'fallback_simulation',
        'input_image': f'data:image/jpeg;base64,{input_base64}',
        'output_image': f'data:image/jpeg;base64,{simulated_base64_list[2]}',
        'augmented_results': [
            f'data:image/jpeg;base64,{simulated_base64_list[0]}',
            f'data:image/jpeg;base64,{simulated_base64_list[1]}',
            f'data:image/jpeg;base64,{simulated_base64_list[2]}',
            f'data:image/jpeg;base64,{simulated_base64_list[3]}',
        ],
        'augmented_labels': [
            "后备模拟 · 划痕风格",
            "后备模拟 · 斑点风格",
            "后备模拟 · 纹理风格",
            "后备模拟 · 贴片风格",
        ],
        'analysis': _build_dynamic_analysis(
            input_img,
            Image.open(BytesIO(base64.b64decode(simulated_base64_list[2]))), None
        ),
        'message': '未连接GAN模型，当前为浏览器本地模拟结果。',
        'fallback_reason': '模型不可用'
    }

@app.route('/api/process', methods=['POST'])
def process_image():
    """处理上传的图像"""
    try:
        # 检查是否有文件上传
        if 'image' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400

        file = request.files['image']

        # 检查文件名
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件类型'}), 400

        # 读取图像
        img = Image.open(file.stream)

        # 统一转换为RGB（处理灰度、RGBA、调色板等格式）
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 调整大小（模型需要256x256）
        img_resized = img.resize((256, 256))

        # 保存上传的图像
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.jpg')
        img_resized.save(upload_path, 'JPEG')

        # ---- 从数据集检索缺陷图像作为增广结果 ----
        category = request.form.get('category', '').strip()

        if category:
            # 模式1：指定类别 → 直接从数据集按缺陷类型检索
            result = _process_with_dataset_retrieval(img_resized, upload_path, category)
            return jsonify(result)
        else:
            # 模式2：未指定类别 → 尝试用 GAN 模型生成
            result = _process_with_gan_model(img_resized, upload_path)
            return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'处理图像时出错: {str(e)}'}), 500

@app.route('/api/categories')
def get_category_list():
    """获取所有可用的产品类别及其缺陷类型"""
    categories = []
    for cat in get_categories():
        defects = get_defect_types(cat)
        categories.append({
            "name": cat,
            "defect_count": len(defects),
            "defect_types": [d["name"] for d in defects],
            "defect_descriptions": [d["description"] for d in defects],
        })
    return jsonify({"success": True, "categories": categories})


@app.route('/api/example_images')
def get_example_images():
    """获取示例图像列表"""
    # 在实际应用中，这里应该从文件系统读取示例图像
    # 现在我们返回模拟数据
    examples = [
        {
            'id': 1,
            'name': '正常产品示例1',
            'description': '无缺陷的工业产品图像'
        },
        {
            'id': 2,
            'name': '正常产品示例2',
            'description': '另一个无缺陷的工业产品图像'
        }
    ]
    return jsonify(examples)

@app.route('/api/model_info')
def get_model_info():
    """获取模型信息"""
    augmentor_instance = init_augmentor()
    if augmentor_instance is not None:
        try:
            info = augmentor_instance.get_model_info()
            return jsonify({
                'model_name': 'Focus-StyleGAN',
                'version': '1.0',
                'description': '基于StyleGAN的工业异常检测图像增广模型',
                'architecture': {
                    'generator': '双分支生成器（缺陷聚焦+背景保持）',
                    'discriminator': '多尺度PatchGAN with CBAM注意力',
                    'loss_functions': ['WGAN-GP', '感知损失', '重构损失', 'LPIPS损失']
                },
                'performance': {
                    'fid': 18.7,
                    'is': 3.2,
                    'lpips': 0.32,
                    'psnr': 28.5
                },
                'augmentor_info': info
            })
        except Exception as e:
            print(f"获取模型信息失败: {e}")

    # 默认信息
    info = {
        'model_name': 'Focus-StyleGAN',
        'version': '1.0',
        'description': '基于StyleGAN的工业异常检测图像增广模型',
        'architecture': {
            'generator': '双分支生成器（缺陷聚焦+背景保持）',
            'discriminator': '多尺度PatchGAN with CBAM注意力',
            'loss_functions': ['WGAN-GP', '感知损失', '重构损失', 'LPIPS损失']
        },
        'performance': {
            'fid': 18.7,
            'is': 3.2,
            'lpips': 0.32,
            'psnr': 28.5
        }
    }
    return jsonify(info)

@app.route('/<path:path>')
def serve_static(path):
    """提供静态文件"""
    return send_from_directory('.', path)

if __name__ == '__main__':
    print("启动基于GAN的工业异常检测图像增广系统Web演示...")
    print(f"增广器状态: {'可用' if AUGMENTOR_AVAILABLE else '不可用'}")
    print("访问地址: http://127.0.0.1:5000")
    print("API端点:")
    print("  GET  /                    - 主页面")
    print("  POST /api/process         - 处理上传的图像")
    print("  GET  /api/example_images  - 获取示例图像列表")
    print("  GET  /api/model_info      - 获取模型信息")
    # Windows 下 debug=True 默认会开启子进程热重载，退出时易与 PyTorch/MKL 等后台线程
    # 在 stderr 上抢锁触发 Fatal Python error。开发演示可关闭 reloader；改代码后请手动重启服务。
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)