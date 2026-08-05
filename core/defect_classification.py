#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缺陷类型识别（单一实现，供全项目共用）

基于真实图像处理技术分析差异图：
Hough线检测(划痕)、多尺度LoG Blob检测(斑点)、局部纹理对比(纹理)、
轮廓复杂度(不规则缺陷)。返回置信度分数而非硬分类。

此前该逻辑在 augmentor 与 web/app.py 中重复维护且参数已漂移，现已统一。
"""

from typing import Dict

import cv2
import numpy as np

TYPE_NAMES = {
    'scratch': '划痕',
    'spot': '斑点/凹陷',
    'texture': '纹理异常',
    'irregular': '不规则缺陷',
    'mixed': '混合型缺陷',
}


def classify_defect_from_diff(diff_map: np.ndarray, min_confidence: float = 0.05) -> Dict[str, any]:
    """
    分析差异图，识别缺陷类型。

    Args:
        diff_map: 差异图 [H, W] float, 0-255
        min_confidence: 最小置信度阈值

    Returns:
        {'primary_type', 'confidence', 'scores': {各类型分数}, 'location'}
    """
    h, w = diff_map.shape
    diff_u8 = np.clip(diff_map, 0, 255).astype(np.uint8)
    _, diff_bin = cv2.threshold(diff_u8, int(np.percentile(diff_u8, 85)), 255, cv2.THRESH_BINARY)

    if diff_bin.sum() < 50:
        return {'primary_type': '无明显缺陷', 'confidence': 0.0,
                'scores': {'scratch': 0, 'spot': 0, 'texture': 0, 'irregular': 0},
                'location': '无明显缺陷区域'}

    # ---- 1. 划痕检测：概率Hough线 ----
    edges = cv2.Canny(diff_u8, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=15, maxLineGap=5)
    scratch_score = 0.0
    if lines is not None and len(lines) > 0:
        lines = np.asarray(lines).reshape(-1, 4)  # 兼容 OpenCV 4.x (N,1,4) 与 5.x (N,4)
        lengths = np.sqrt((lines[:, 2] - lines[:, 0]) ** 2 + (lines[:, 3] - lines[:, 1]) ** 2)
        total_len = float(lengths.sum())
        diag = np.sqrt(h ** 2 + w ** 2)
        line_density = min(len(lines), 80) / 80.0
        scratch_score = min(1.0, total_len / (diag * 2.0) + line_density * 0.3)

    # ---- 2. 斑点检测：多尺度 Laplacian of Gaussian ----
    diff_smooth = cv2.GaussianBlur(diff_u8.astype(np.float32), (5, 5), 1.0)
    spot_scores = []
    for sigma in [3, 5, 7, 10]:
        log = cv2.GaussianBlur(diff_smooth, (0, 0), sigma) - cv2.GaussianBlur(diff_smooth, (0, 0), sigma * 1.6)
        log_abs = np.abs(log)
        threshold = np.percentile(log_abs, 92)
        blobs = log_abs > threshold
        n_labels, labels = cv2.connectedComponents(blobs.astype(np.uint8))
        if n_labels > 1:
            areas = [np.sum(labels == i) for i in range(1, n_labels)]
            valid = [a for a in areas if 10 < a < (h * w * 0.15)]
            spot_scores.append(len(valid) * sigma / 18.0)

    spot_score = min(1.0, sum(spot_scores) / max(len(spot_scores), 1) * 0.5)

    # ---- 3. 纹理异常检测：局部纹理对比 ----
    diff_f = diff_u8.astype(np.float32)
    texture_score = 0.0
    if diff_bin.sum() > 200:
        kernel_size = 11
        local_mean = cv2.blur(diff_f, (kernel_size, kernel_size))
        local_sq = cv2.blur(diff_f ** 2, (kernel_size, kernel_size))
        local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))

        mask = diff_bin > 0
        if mask.sum() > 0:
            texture_score = min(1.0, np.mean(local_std[mask]) / 55.0)

    # ---- 4. 不规则缺陷：轮廓复杂度 ----
    contours, _ = cv2.findContours(diff_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    irregular_score = 0.0
    if contours:
        contour_scores = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue
            # 圆形度：4π*area/perimeter²，越低越不规则
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
            # 凸性：area/convex_hull_area
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            convexity = area / (hull_area + 1e-8)
            irr = (1.0 - circularity) * 0.5 + (1.0 - convexity) * 0.5
            contour_scores.append(irr)
        if contour_scores:
            irregular_score = min(1.0, np.mean(contour_scores) * 1.2)

    # ---- 5. 综合判断 ----
    scores = {
        'scratch': round(scratch_score, 3),
        'spot': round(spot_score, 3),
        'texture': round(texture_score, 3),
        'irregular': round(irregular_score, 3),
    }
    primary = max(scores, key=scores.get)
    confidence = scores[primary]

    if confidence < min_confidence:
        primary = 'mixed'
        confidence = max(scores.values()) if scores else 0.0

    # 定位缺陷区域
    mask = diff_bin > 0
    ys, xs = np.where(mask)
    if len(ys) > 0:
        cy, cx = float(ys.mean()), float(xs.mean())
        if 0.3 * h <= cy <= 0.7 * h and 0.3 * w <= cx <= 0.7 * w:
            location = "中心区域"
        else:
            v = "上" if cy < h / 2 else "下"
            hz = "左" if cx < w / 2 else "右"
            location = f"{hz}{v}区域"
    else:
        location = "不明显"

    return {
        'primary_type': TYPE_NAMES.get(primary, primary),
        'confidence': round(confidence * 100, 1),
        'scores': scores,
        'location': location,
    }
