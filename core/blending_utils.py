#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缺陷融合共享工具（单一实现，供全项目共用）

RealDefectBlender 与 DefectStackingAugmentor 共用的底层图像操作：
ROI 提取、随机几何变换、色彩匹配、Alpha 融合、重叠检测。

两个增广器只保留各自策略（随机选取 vs pattern 互补堆叠），
底层操作统一维护在本模块，避免参数漂移。
"""

from typing import List, Tuple

import cv2
import numpy as np
import random


def extract_roi(img: np.ndarray, mask: np.ndarray, padding: int = 4
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    按 mask 非零区域提取 ROI（带 padding）

    Returns:
        (img_roi, mask_roi)
    """
    ys, xs = np.where(mask > 30)
    if len(ys) < 10:
        return img, mask
    y1 = max(0, ys.min() - padding)
    y2 = min(img.shape[0], ys.max() + padding + 1)
    x1 = max(0, xs.min() - padding)
    x2 = min(img.shape[1], xs.max() + padding + 1)
    return img[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy()


def random_geometric_transform(defect_roi: np.ndarray, mask_roi: np.ndarray,
                               scale_range: Tuple[float, float] = (0.5, 1.6),
                               angle_range: float = 25.0
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    随机缩放 + 旋转 + 翻转。

    先 resize 再旋转（scale=1.0），避免 warpAffine 双重缩放；
    图像用 BORDER_REPLICATE 防止黑边污染，mask 用 BORDER_CONSTANT(0)。
    """
    h, w = defect_roi.shape[:2]

    # 1. 随机缩放
    scale = random.uniform(*scale_range)
    new_w = max(6, int(w * scale))
    new_h = max(6, int(h * scale))
    img = cv2.resize(defect_roi, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    msk = cv2.resize(mask_roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 2. 随机旋转（输出尺寸扩展以容纳旋转后内容）
    angle = random.uniform(-angle_range, angle_range)
    center = (new_w / 2, new_h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a, sin_a = abs(rot_mat[0, 0]), abs(rot_mat[0, 1])
    out_w = int(np.ceil(new_h * sin_a + new_w * cos_a))
    out_h = int(np.ceil(new_h * cos_a + new_w * sin_a))
    rot_mat[0, 2] += out_w / 2 - center[0]
    rot_mat[1, 2] += out_h / 2 - center[1]

    img = cv2.warpAffine(img, rot_mat, (out_w, out_h),
                         flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    msk = cv2.warpAffine(msk, rot_mat, (out_w, out_h),
                         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 3. 随机翻转
    if random.random() > 0.5:
        img, msk = cv2.flip(img, 1), cv2.flip(msk, 1)
    if random.random() > 0.5:
        img, msk = cv2.flip(img, 0), cv2.flip(msk, 0)

    return img, msk


def constrain_roi_size(defect_roi: np.ndarray, mask_roi: np.ndarray,
                       img_h: int, img_w: int,
                       min_ratio: float = 0.04, max_ratio: float = 0.6
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 ROI 尺寸约束到图像尺寸的 [min_ratio, max_ratio] 范围内
    """
    dh, dw = defect_roi.shape[:2]
    max_dim_img = int(min(img_h, img_w) * max_ratio)
    min_dim_img = int(min(img_h, img_w) * min_ratio)
    cur = max(dh, dw)

    if cur > max_dim_img:
        s = max_dim_img / cur
    elif cur < min_dim_img:
        s = min_dim_img / cur
    else:
        return defect_roi, mask_roi

    new_w = max(6, int(dw * s))
    new_h = max(6, int(dh * s))
    defect_roi = cv2.resize(defect_roi, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    mask_roi = cv2.resize(mask_roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return defect_roi, mask_roi


def color_match(source: np.ndarray, target_bg: np.ndarray, mask: np.ndarray,
                strength: float = 0.45) -> np.ndarray:
    """
    将 source 的 mask 区域均值匹配到 target_bg，保留缺陷纹理与对比度。
    strength=0 不做匹配，strength=1 完全均值迁移。
    """
    if strength <= 0.01:
        return source.astype(np.uint8)

    mask_f = (mask > 30).astype(np.float32)
    s = mask_f.sum()
    if s < 10:
        return source.astype(np.uint8)

    src_f = source.astype(np.float32)
    tgt_f = target_bg.astype(np.float32)
    src_mean = np.sum(src_f * mask_f[..., None], axis=(0, 1)) / (s + 1e-6)
    tgt_mean = np.sum(tgt_f * mask_f[..., None], axis=(0, 1)) / (s + 1e-6)

    matched = src_f - src_mean + tgt_mean
    blend = mask_f[..., None] * strength
    return np.clip(src_f * (1 - blend) + matched * blend, 0, 255).astype(np.uint8)


def alpha_blend(defect: np.ndarray, mask: np.ndarray, background: np.ndarray,
                position: Tuple[int, int], feather: float = 0.0) -> np.ndarray:
    """
    将缺陷 alpha 融合到背景图像上（自动裁剪越界部分）。

    Args:
        defect: 缺陷 ROI [dh, dw, 3] uint8
        mask: 缺陷 mask [dh, dw] uint8, 255=缺陷
        background: 背景图像 [H, W, 3] uint8
        position: 左上角 (x, y)
        feather: mask 边缘高斯羽化强度（0 表示不羽化）
    """
    dh, dw = defect.shape[:2]
    bh, bw = background.shape[:2]
    x, y = position

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + dw), min(bh, y + dh)
    dx1, dy1 = max(0, -x), max(0, -y)
    dx2, dy2 = dx1 + (x2 - x1), dy1 + (y2 - y1)

    if dx2 <= dx1 or dy2 <= dy1:
        return background

    dp = defect[dy1:dy2, dx1:dx2].astype(np.float32)
    mp = mask[dy1:dy2, dx1:dx2].astype(np.float32) / 255.0
    bp = background[y1:y2, x1:x2].astype(np.float32)

    # 仅在边缘做少量羽化（保持内部纹理锐利）
    if feather > 0 and mp.max() > 0.1:
        ksize = 3 if feather <= 1 else 5
        mp = cv2.GaussianBlur(mp, (ksize, ksize), max(feather, 0.5))

    result = background.copy()
    result[y1:y2, x1:x2] = np.clip(bp * (1 - mp[..., None]) + dp * mp[..., None], 0, 255).astype(np.uint8)
    return result


def has_overlap(rect: Tuple[int, int, int, int],
                rects: List[Tuple[int, int, int, int]],
                iou_threshold: float = 0.2) -> bool:
    """检测矩形 rect 是否与 rects 中任一矩形的 IoU 超过阈值"""
    x1, y1, x2, y2 = rect
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return True
    for rx1, ry1, rx2, ry2 in rects:
        ox1, oy1 = max(x1, rx1), max(y1, ry1)
        ox2, oy2 = min(x2, rx2), min(y2, ry2)
        if ox1 >= ox2 or oy1 >= oy2:
            continue
        inter = (ox2 - ox1) * (oy2 - oy1)
        iou = inter / (area + (rx2 - rx1) * (ry2 - ry1) - inter + 1e-8)
        if iou > iou_threshold:
            return True
    return False
