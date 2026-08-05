#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实缺陷迁移增广器
从 MVTec AD 数据集提取真实缺陷区域，通过几何增强、色彩匹配、
Poisson/Alpha 融合叠加到正常(good)图像上，生成高保真缺陷增广样本。
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# GPU 超分辨率（可选，云环境安装 realesrgan 后自动启用）
_GPU_SR_AVAILABLE = False
_SR_MODULE = None
try:
    from core.super_resolution import SuperResolution
    _GPU_SR_AVAILABLE = True
    _SR_MODULE = SuperResolution
except ImportError:
    pass


def _imread_unicode(path: str) -> np.ndarray:
    """OpenCV imread 不支持 Windows 中文路径，用 imdecode 代替"""
    # 先用 utf-8 路径尝试
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    # 回退：二进制读取 + imdecode
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imread_gray_unicode(path: str) -> np.ndarray:
    """同上，灰度版本"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def _imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """OpenCV imwrite 在 Windows 中文路径下的安全写入"""
    ext = Path(path).suffix
    success, buf = cv2.imencode(ext, img)
    if not success:
        return False
    with open(path, 'wb') as f:
        f.write(buf.tobytes())
    return True


# 数据集根目录
_DATASET_ROOT = Path(__file__).parent.parent / "datasets" / "mvtec_anomaly_detection"

# 支持的缺陷类别及其缺陷类型
_SUPPORTED_CATEGORIES: Dict[str, List[str]] = {}


def _init_categories():
    """扫描数据集，缓存可用的类别和缺陷类型"""
    global _SUPPORTED_CATEGORIES
    if _SUPPORTED_CATEGORIES:
        return
    if not _DATASET_ROOT.exists():
        return
    for cat_dir in sorted(_DATASET_ROOT.iterdir()):
        if not cat_dir.is_dir():
            continue
        gt_dir = cat_dir / "ground_truth"
        test_dir = cat_dir / "test"
        if not gt_dir.exists() or not test_dir.exists():
            continue
        defect_types = []
        for d in sorted(test_dir.iterdir()):
            if d.is_dir() and d.name != "good":
                # 确认有对应的 ground_truth
                if (gt_dir / d.name).exists():
                    defect_types.append(d.name)
        if defect_types:
            _SUPPORTED_CATEGORIES[cat_dir.name] = defect_types


def get_categories() -> List[str]:
    """返回所有可用的产品类别"""
    _init_categories()
    return sorted(_SUPPORTED_CATEGORIES.keys())


def get_defect_types(category: str) -> List[str]:
    """返回指定类别的缺陷类型列表"""
    _init_categories()
    return _SUPPORTED_CATEGORIES.get(category, [])


def get_random_defect_sample(category: str, defect_type: str = None
                             ) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    随机获取一个缺陷样本及其 mask

    Returns:
        (defect_image_rgb, mask_gray, defect_type_name)
        defect_image_rgb: [H, W, 3] uint8
        mask_gray: [H, W] uint8, 0=背景 255=缺陷
    """
    _init_categories()
    if category not in _SUPPORTED_CATEGORIES:
        raise ValueError(f"不支持的类别: {category}，可用: {list(_SUPPORTED_CATEGORIES.keys())}")

    if defect_type is None:
        defect_type = random.choice(_SUPPORTED_CATEGORIES[category])

    test_dir = _DATASET_ROOT / category / "test" / defect_type
    gt_dir = _DATASET_ROOT / category / "ground_truth" / defect_type

    if not test_dir.exists() or not gt_dir.exists():
        raise ValueError(f"缺陷类型 {defect_type} 不存在于类别 {category}")

    test_images = sorted(test_dir.glob("*.png"))
    if not test_images:
        raise ValueError(f"类别 {category}/{defect_type} 没有缺陷图像")

    # 随机选一张
    img_path = random.choice(test_images)
    # 查找对应的 mask
    mask_name = img_path.stem.replace(".", "") + "_mask.png"
    mask_path = gt_dir / mask_name

    defect_img = _imread_unicode(str(img_path))
    defect_img = cv2.cvtColor(defect_img, cv2.COLOR_BGR2RGB)

    if mask_path.exists():
        mask = _imread_gray_unicode(str(mask_path))
    else:
        # 没有 mask 时，用差分法估算
        good_dir = _DATASET_ROOT / category / "test" / "good"
        good_images = sorted(good_dir.glob("*.png"))
        if good_images:
            ref = _imread_unicode(str(good_images[0]))
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
            diff = cv2.absdiff(defect_img, ref)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        else:
            mask = np.ones(defect_img.shape[:2], dtype=np.uint8) * 255

    return defect_img, mask, defect_type


class RealDefectBlender:
    """真实缺陷融合器 —— 将真实缺陷迁移到正常图像上"""

    def __init__(self, category: str = None, rng_seed: int = None,
                 use_gpu_sr: bool = True, gpu_sr_device: str = "auto",
                 gpu_sr_model: str = "RealESRGAN_x4plus"):
        """
        Args:
            category: 产品类别（如 'bottle'），None 则自动选择
            rng_seed: 随机种子
            use_gpu_sr: 是否启用 GPU 超分辨率（需要 realesrgan）
            gpu_sr_device: GPU 设备 'auto' | 'cuda' | 'cpu'
            gpu_sr_model: Real-ESRGAN 模型名
        """
        _init_categories()
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        if category is None:
            categories = get_categories()
            if not categories:
                raise ValueError("未找到任何可用的数据集类别")
            self.category = random.choice(categories)
        else:
            if category not in _SUPPORTED_CATEGORIES:
                raise ValueError(f"不支持的类别: {category}")
            self.category = category

        # GPU 超分辨率
        self._sr = None
        if use_gpu_sr and _GPU_SR_AVAILABLE:
            try:
                self._sr = _SR_MODULE(device=gpu_sr_device, model_name=gpu_sr_model)
                print(f"  GPU 超分辨率已启用: {gpu_sr_model} @ {self._sr.device}")
            except Exception as e:
                print(f"  GPU 超分辨率初始化失败 ({e})，回退到 Lanczos")
                self._sr = None

    @staticmethod
    def extract_defect_roi(defect_img: np.ndarray, mask: np.ndarray,
                           padding: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        从缺陷图像中提取缺陷 ROI 区域（委托 core.blending_utils 统一实现）

        Returns:
            (defect_roi_rgb, mask_roi)
        """
        from core.blending_utils import extract_roi
        return extract_roi(defect_img, mask, padding)

    @staticmethod
    def random_geometric_transform(defect_roi: np.ndarray, mask_roi: np.ndarray,
                                   target_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """随机几何增强：缩放、旋转、翻转（委托 core.blending_utils 统一实现）"""
        from core.blending_utils import random_geometric_transform
        return random_geometric_transform(
            defect_roi, mask_roi, scale_range=(0.6, 1.8), angle_range=30.0
        )

    @staticmethod
    def color_match(source: np.ndarray, target_bg: np.ndarray, mask: np.ndarray,
                    strength: float = 0.5) -> np.ndarray:
        """
        将 source 的均值匹配到 target_bg，保留缺陷全部纹理、对比度。
        （委托 core.blending_utils 统一实现）
        strength=0 不做匹配，strength=1 完全均值迁移。
        """
        from core.blending_utils import color_match
        return color_match(source, target_bg, mask, strength)

    @staticmethod
    def alpha_blend(defect: np.ndarray, mask: np.ndarray,
                    background: np.ndarray, position: Tuple[int, int],
                    feather: int = 1) -> np.ndarray:
        """
        将缺陷 alpha 融合到背景图像上。feather 控制在 mask 边缘做几像素高斯羽化。
        （委托 core.blending_utils 统一实现）
        """
        from core.blending_utils import alpha_blend
        return alpha_blend(defect, mask, background, position, feather=float(feather))

    def blend_single_defect(self, good_image: np.ndarray,
                            defect_img: np.ndarray, mask: np.ndarray,
                            intensity: float = 1.0,
                            use_color_match: bool = None) -> np.ndarray:
        """
        将单个缺陷融合到 good 图像上。
        use_color_match=None 时随机决定（70%不做色彩匹配，保留原始缺陷外观）。
        """
        result = good_image.copy()
        h, w = good_image.shape[:2]

        # 1. 提取缺陷 ROI
        defect_roi, mask_roi = self.extract_defect_roi(defect_img, mask)
        if defect_roi.size == 0 or (mask_roi > 30).sum() < 20:
            return result

        # 2. 随机几何变换
        defect_roi, mask_roi = self.random_geometric_transform(defect_roi, mask_roi)
        if defect_roi.size == 0 or (mask_roi > 30).sum() < 20:
            return result

        # 3. 约束缺陷尺寸（不超过图像 75%，至少 5%）
        dh, dw = defect_roi.shape[:2]
        max_dim = max(dh, dw)
        min_dim_img = int(min(h, w) * 0.05)
        max_dim_img = int(min(h, w) * 0.75)

        if max_dim > max_dim_img:
            scale = max_dim_img / max_dim
            defect_roi = cv2.resize(defect_roi, (max(8, int(dw * scale)), max(8, int(dh * scale))),
                                    interpolation=cv2.INTER_LANCZOS4)
            mask_roi = cv2.resize(mask_roi, (max(8, int(dw * scale)), max(8, int(dh * scale))),
                                  interpolation=cv2.INTER_NEAREST)
        elif max_dim < min_dim_img:
            scale = min_dim_img / max_dim
            defect_roi = cv2.resize(defect_roi, (int(dw * scale), int(dh * scale)),
                                    interpolation=cv2.INTER_LANCZOS4)
            mask_roi = cv2.resize(mask_roi, (int(dw * scale), int(dh * scale)),
                                  interpolation=cv2.INTER_NEAREST)

        dh, dw = defect_roi.shape[:2]

        # 4. 放置位置（40% 概率靠近中心，更显眼）
        margin = 5
        if random.random() < 0.4:
            # 靠近中心，但确保缺陷完全在图像内
            x_lo = max(margin, w // 4)
            x_hi = min(w - dw - margin, w - w // 4)
            x = random.randint(x_lo, max(x_lo, x_hi)) if x_hi >= x_lo else margin
            y_lo = max(margin, h // 4)
            y_hi = min(h - dh - margin, h - h // 4)
            y = random.randint(y_lo, max(y_lo, y_hi)) if y_hi >= y_lo else margin
        else:
            max_x = max(margin, w - dw - margin)
            max_y = max(margin, h - dh - margin)
            x = random.randint(margin, max_x) if max_x > margin else margin
            y = random.randint(margin, max_y) if max_y > margin else margin

        # 最终钳制，防止浮点累积导致越界
        x = max(0, min(x, w - dw))
        y = max(0, min(y, h - dh))

        # 5. 色彩匹配（均值迁移，保留缺陷全部纹理/对比度）
        if use_color_match is None:
            use_color_match = random.random() < 0.5

        if use_color_match:
            bg_roi = good_image[y:y + dh, x:x + dw]
            cm_strength = random.uniform(0.3, 0.7)
            defect_roi = self.color_match(defect_roi, bg_roi, mask_roi, strength=cm_strength)

        # 6. Alpha 融合（几乎不羽化，保持缺陷锐利）
        feather = max(0, min(1, int(intensity * 0.5)))
        result = self.alpha_blend(defect_roi, mask_roi, result, (x, y), feather=feather)

        return result

    def _detail_preserving_upscale(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        保留细节的超分辨率放大。
        GPU 可用时使用 Real-ESRGAN（神经网络超分，质量最佳），
        否则回退到 Lanczos4 + CLAHE + 锐化。
        """
        # GPU 超分辨率（Real-ESRGAN）
        if self._sr is not None:
            try:
                return self._sr.upscale(image, factor=int(factor))
            except Exception as e:
                print(f"  GPU 超分不可用 ({e})，回退到 Lanczos")
                self._sr = None  # 仅报错一次

        # CPU 回退：Lanczos4 + CLAHE + 锐化
        h, w = image.shape[:2]
        new_w, new_h = int(w * factor), int(h * factor)

        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        lab = cv2.cvtColor(upscaled, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

        sigma = max(0.5, min(new_h, new_w) / 512.0)
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigma)
        sharpened = cv2.addWeighted(enhanced, 1.35, blur, -0.35, 0)

        noise = np.random.randn(*sharpened.shape).astype(np.float32) * 0.6
        return np.clip(sharpened.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def _realistic_postprocess(image: np.ndarray) -> np.ndarray:
        """
        照片级真实感后处理：
        1. CLAHE 局部对比度增强（LAB L通道），恢复纹理
        2. 锐化（轻微 Unsharp Mask）
        3. 传感器噪声模拟（亮度 + 色度），匹配真实相机特性
        """
        h, w = image.shape[:2]
        resolution = min(h, w)

        # ---- CLAHE 局部对比度增强 ----
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

        # ---- 锐化 ----
        sigma = max(0.5, resolution / 640.0)
        strength = 0.22
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigma)
        sharpened = cv2.addWeighted(enhanced, 1.0 + strength, blur, -strength, 0)

        # ---- 传感器噪声模拟（亮度 + 色度） ----
        # 亮度噪声
        luma_noise_sigma = 0.7
        luma_noise = np.random.randn(h, w).astype(np.float32) * luma_noise_sigma
        # 色度噪声（更低强度，真实相机色度噪声远弱于亮度噪声）
        chroma_noise_sigma = 0.35
        chroma_noise = np.random.randn(h, w, 2).astype(np.float32) * chroma_noise_sigma

        noisy_f = sharpened.astype(np.float32)
        noisy_f[:, :, 0] += luma_noise * 0.7 + chroma_noise[:, :, 0] * 0.3
        noisy_f[:, :, 1] += luma_noise * 0.6 + chroma_noise[:, :, 1] * 0.3
        noisy_f[:, :, 2] += luma_noise * 0.5 + chroma_noise[:, :, 0] * 0.4

        return np.clip(noisy_f, 0, 255).astype(np.uint8)

    def generate(self, good_image_path: str, output_path: str,
                 n_defects: int = None, category: str = None,
                 intensity: float = 1.0, rng_seed: int = None,
                 upscale_factor: float = 1.0) -> str:
        """
        生成缺陷增广图像

        Args:
            good_image_path: 输入正常图像路径
            output_path: 输出路径
            n_defects: 叠加的缺陷数量 (1~5)，None 则随机
            category: 产品类别，None 则使用初始化时的类别
            intensity: 缺陷强度
            rng_seed: 随机种子
            upscale_factor: 超分放大倍率 (>1.0 启用保留细节放大)

        Returns:
            输出图像路径
        """
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        if category is None:
            category = self.category

        if n_defects is None:
            n_defects = random.randint(1, 3)

        # 读取 good 图像
        good_img = _imread_unicode(good_image_path)
        if good_img is None:
            raise ValueError(f"无法读取图像: {good_image_path}")
        good_img = cv2.cvtColor(good_img, cv2.COLOR_BGR2RGB)

        # 超分辨率放大（缺陷融合前，使缺陷在高分辨率画布上更清晰）
        if upscale_factor > 1.0:
            good_img = self._detail_preserving_upscale(good_img, upscale_factor)

        defect_types = get_defect_types(category)
        if not defect_types:
            raise ValueError(f"类别 {category} 没有可用的缺陷类型")

        result = good_img.copy()

        for i in range(n_defects):
            try:
                defect_type = random.choice(defect_types)
                defect_img, mask, dt_name = get_random_defect_sample(category, defect_type)

                result = self.blend_single_defect(
                    result, defect_img, mask,
                    intensity=intensity * random.uniform(0.7, 1.3)
                )
            except Exception as e:
                print(f"  叠加缺陷 {i + 1} 失败: {e}")
                continue

        # 后处理：微锐化 + 传感器噪声模拟 → 照片级真实感
        result = self._realistic_postprocess(result)

        # 保存
        _imwrite_unicode(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        return output_path

    def generate_batch(self, good_image_path: str, output_dir: str,
                       n_variations: int = 5, n_defects: int = None,
                       intensity: float = 1.0,
                       upscale_factor: float = 1.0) -> List[str]:
        """
        为一张 good 图像生成多个缺陷变体

        Args:
            upscale_factor: 超分放大倍率 (>1.0 启用，云 GPU 环境推荐 4.0)

        Returns:
            生成图像路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(good_image_path).stem
        paths = []
        for i in range(n_variations):
            out_path = output_dir / f"{stem}_defect_{i:02d}.png"
            self.generate(
                good_image_path, str(out_path),
                n_defects=n_defects,
                intensity=intensity,
                rng_seed=random.randint(0, 2**31 - 1),
                upscale_factor=upscale_factor,
            )
            paths.append(str(out_path))
        return paths


# ============================================================
# 命令行接口
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="真实缺陷迁移增广器")
    parser.add_argument("--input", type=str, required=True, help="输入正常图像路径")
    parser.add_argument("--output", type=str, required=True, help="输出图像路径")
    parser.add_argument("--category", type=str, default=None, help="产品类别 (如 bottle)")
    parser.add_argument("--n_defects", type=int, default=None, help="叠加缺陷数量")
    parser.add_argument("--intensity", type=float, default=1.0, help="缺陷强度")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--batch", type=int, default=0, help="批量生成数量(>0启用批量)")
    parser.add_argument("--upscale", type=float, default=1.0,
                        help="超分放大倍率 (2.0=2x, 4.0=4x, 需 realesrgan)")
    parser.add_argument("--gpu-sr", action="store_true", default=True,
                        help="启用 GPU 超分辨率 (默认)")
    parser.add_argument("--no-gpu-sr", action="store_true",
                        help="禁用 GPU 超分辨率，使用 Lanczos")
    parser.add_argument("--sr-device", type=str, default="auto",
                        help="超分设备: auto/cuda/cpu")
    parser.add_argument("--sr-model", type=str, default="RealESRGAN_x4plus",
                        help="Real-ESRGAN 模型名")

    args = parser.parse_args()

    use_gpu = not args.no_gpu_sr and args.gpu_sr
    blender = RealDefectBlender(
        category=args.category, rng_seed=args.seed,
        use_gpu_sr=use_gpu, gpu_sr_device=args.sr_device,
        gpu_sr_model=args.sr_model,
    )

    if args.batch > 0:
        output_dir = Path(args.output)
        paths = blender.generate_batch(
            args.input, str(output_dir),
            n_variations=args.batch,
            n_defects=args.n_defects,
            intensity=args.intensity,
            upscale_factor=args.upscale,
        )
        print(f"生成 {len(paths)} 张缺陷增广图像 -> {output_dir}")
    else:
        blender.generate(
            args.input, args.output,
            n_defects=args.n_defects,
            intensity=args.intensity,
            rng_seed=args.seed,
            upscale_factor=args.upscale,
        )
        print(f"生成缺陷增广图像 -> {args.output}")


if __name__ == "__main__":
    main()
