#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像特征提取（单一实现，供全项目共用）

6 维手工特征向量：edge_density, texture_complexity, brightness_mean,
                brightness_std, dominant_orientation, surface_type

此前该逻辑在 augmentor / retrieval_augmentor / web/app.py 中重复维护三份，
现已统一到本模块。所有特征分量归一化到 [0, 1]，
surface_type 使用 float 编码：0=smooth, 1/3=textured, 2/3=structured, 1=reflective。
"""

from typing import Dict

import cv2
import numpy as np

FEATURE_KEYS = [
    'edge_density', 'texture_complexity', 'brightness_mean',
    'brightness_std', 'dominant_orientation', 'surface_type',
]


def extract_feature_vector(image_bgr: np.ndarray) -> np.ndarray:
    """
    从 BGR uint8 图像提取归一化特征向量。

    Args:
        image_bgr: [H, W, 3] uint8, BGR 通道序

    Returns:
        float32 ndarray, shape (6,)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 1. 边缘密度
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edge_density = float(edges.sum()) / float(edges.size * 255)

    # 2. 纹理复杂度 (局部标准差均值)
    kernel = np.ones((7, 7), dtype=np.float32) / 49
    local_mean = cv2.filter2D(gray, -1, kernel)
    local_sq_mean = cv2.filter2D(gray * gray, -1, kernel)
    local_var = np.maximum(local_sq_mean - local_mean * local_mean, 0)
    texture_complexity = float(np.sqrt(local_var).mean() / 128.0)

    # 3. 亮度分布
    brightness_mean = float(gray.mean() / 255.0)
    brightness_std = float(gray.std() / 255.0)

    # 4. 主方向（强梯度区域的梯度方向直方图峰值）
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi
    strong_mask = magnitude > np.percentile(magnitude, 70)
    if strong_mask.sum() > 100:
        hist, _ = np.histogram(orientation[strong_mask], bins=18, range=(-180, 180))
        dominant_orientation = float(np.argmax(hist)) / 18.0
    else:
        dominant_orientation = 0.5

    # 5. 表面类型 (0=smooth, 1/3=textured, 2/3=structured, 1=reflective)
    if edge_density > 0.15:
        surface_type = 2.0 / 3.0
    elif texture_complexity > 0.35:
        surface_type = 1.0 / 3.0
    elif brightness_std > 0.25:
        surface_type = 1.0
    else:
        surface_type = 0.0

    return np.array([
        edge_density,
        texture_complexity,
        brightness_mean,
        brightness_std,
        dominant_orientation,
        surface_type,
    ], dtype=np.float32)


def features_vector_to_dict(vec: np.ndarray) -> Dict[str, float]:
    """将 (6,) 特征向量转换为特征字典"""
    return {k: float(vec[i]) for i, k in enumerate(FEATURE_KEYS)}


def features_dict_to_vector(features: Dict[str, float]) -> np.ndarray:
    """将特征字典转换回 (6,) 向量"""
    return np.array([float(features.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float32)


def build_features_from_bgr(bgr_np: np.ndarray) -> Dict[str, float]:
    """从 BGR uint8 图像一步构建特征字典"""
    return features_vector_to_dict(extract_feature_vector(bgr_np))


def build_features_from_rgb(rgb_np: np.ndarray) -> Dict[str, float]:
    """从 RGB uint8 图像一步构建特征字典"""
    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    return build_features_from_bgr(bgr)
