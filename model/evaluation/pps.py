#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
物理合理性得分 (Physical Plausibility Score, PPS)
评估生成缺陷是否符合材料物理规律

PPS = 0.5 * S_geo + 0.5 * S_illum
其中:
  S_geo:  几何一致性得分 (KS检验比较生成缺陷与真实缺陷的几何属性分布)
  S_illum: 光照一致性得分 (Phong光照模型下的阴影一致性)
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy import stats
from typing import Dict, Tuple, Optional, List


class PhysicalPlausibilityScore:
    """物理合理性得分计算器"""

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        curvature_bins: int = 50,
        ks_alpha: float = 0.05
    ):
        """
        Args:
            canny_low: Canny边缘检测低阈值
            canny_high: Canny边缘检测高阈值
            curvature_bins: 曲率分布直方图的bin数量
            ks_alpha: KS检验的显著性水平
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.curvature_bins = curvature_bins
        self.ks_alpha = ks_alpha

    def compute(
        self,
        generated_images: torch.Tensor,
        real_images: torch.Tensor,
        real_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        计算物理合理性得分

        Args:
            generated_images: 生成图像 [N, 3, H, W], 范围[-1, 1]
            real_images: 真实缺陷图像 [N, 3, H, W]
            real_masks: 真实缺陷掩码 [N, 1, H, W], 可选

        Returns:
            包含PPS及子指标的字典
        """
        n_gen = generated_images.shape[0]

        # 转换到[0, 255]的numpy格式
        gen_np = self._tensor_to_numpy(generated_images)
        real_np = self._tensor_to_numpy(real_images)

        # 1. 几何一致性得分
        s_geo = self._compute_geometric_consistency(gen_np, real_np, real_masks)

        # 2. 光照一致性得分
        s_illum = self._compute_illumination_consistency(gen_np)

        # 综合得分
        pps = 0.5 * s_geo + 0.5 * s_illum

        return {
            'pps': float(pps),
            's_geo': float(s_geo),
            's_illum': float(s_illum)
        }

    def _tensor_to_numpy(self, images: torch.Tensor) -> np.ndarray:
        """将张量转换为numpy数组 [N, H, W, C], uint8"""
        imgs = images.detach().cpu()
        # 从[-1, 1]转换到[0, 255]
        imgs = ((imgs + 1) / 2 * 255).clamp(0, 255)
        imgs = imgs.permute(0, 2, 3, 1).numpy().astype(np.uint8)
        return imgs

    def _compute_geometric_consistency(
        self,
        gen_images: np.ndarray,
        real_images: np.ndarray,
        real_masks: Optional[torch.Tensor] = None
    ) -> float:
        """
        计算几何一致性得分 S_geo

        使用Canny边缘检测提取缺陷边缘轮廓，
        计算曲率分布、长宽比、边缘锐度等几何属性，
        通过KS检验比较生成缺陷与真实缺陷的统计分布差异。
        """
        gen_geometric_features = []
        real_geometric_features = []

        # 从生成图像提取几何特征
        for img in gen_images:
            features = self._extract_geometric_features(img)
            gen_geometric_features.extend(features)

        # 从真实图像提取几何特征
        for img in real_images[:len(gen_images)]:  # 匹配数量
            features = self._extract_geometric_features(img)
            real_geometric_features.extend(features)

        if len(gen_geometric_features) == 0 or len(real_geometric_features) == 0:
            return 0.0

        gen_features = np.array(gen_geometric_features)
        real_features = np.array(real_geometric_features)

        # 对每个几何属性做KS检验
        p_values = []
        for dim in range(gen_features.shape[1]):
            if np.std(gen_features[:, dim]) < 1e-8 and np.std(real_features[:, dim]) < 1e-8:
                p_values.append(1.0)
                continue
            try:
                ks_stat, p_value = stats.ks_2samp(
                    gen_features[:, dim],
                    real_features[:, dim]
                )
                p_values.append(p_value)
            except Exception:
                p_values.append(0.0)

        # KS检验的p值作为几何一致性度量
        s_geo = np.mean(p_values)
        return float(np.clip(s_geo, 0.0, 1.0))

    def _extract_geometric_features(self, img: np.ndarray) -> List[np.ndarray]:
        """
        从单张图像提取几何特征

        Returns:
            特征向量列表，每个元素为[曲率均值, 曲率方差, 边缘密度, 边缘锐度, 轮廓长宽比]
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Canny边缘检测
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features_list = []
        for contour in contours:
            if len(contour) < 5:
                continue

            # 1. 曲率分布统计
            curvatures = self._compute_curvature(contour)
            curvature_mean = np.mean(curvatures) if len(curvatures) > 0 else 0
            curvature_var = np.var(curvatures) if len(curvatures) > 0 else 0

            # 2. 边缘密度
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # 3. 边缘锐度 (用Sobel梯度幅值近似)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            # 只在边缘位置计算锐度
            edge_sharpness = np.mean(gradient_magnitude[edges > 0]) if np.any(edges > 0) else 0

            # 4. 轮廓长宽比
            rect = cv2.minAreaRect(contour)
            (w, h) = rect[1]
            if min(w, h) > 0:
                aspect_ratio = max(w, h) / min(w, h)
            else:
                aspect_ratio = 1.0

            features_list.append(np.array([
                curvature_mean,
                curvature_var,
                edge_density,
                edge_sharpness,
                aspect_ratio
            ]))

        # 如果没有有效轮廓，返回默认零特征
        if len(features_list) == 0:
            features_list.append(np.zeros(5))

        return features_list

    def _compute_curvature(self, contour: np.ndarray) -> np.ndarray:
        """
        计算轮廓各点的曲率

        Args:
            contour: OpenCV轮廓, shape [N, 1, 2]

        Returns:
            离散曲率数组
        """
        pts = contour.squeeze()
        if pts.ndim != 2 or pts.shape[0] < 3:
            return np.array([0.0])

        curvatures = []
        for i in range(1, len(pts) - 1):
            p_prev = pts[i - 1].astype(np.float64)
            p_curr = pts[i].astype(np.float64)
            p_next = pts[i + 1].astype(np.float64)

            # 离散曲率: 使用三点确定圆的曲率倒数
            v1 = p_curr - p_prev
            v2 = p_next - p_curr

            # 一阶导数和二阶导数的离散近似
            dx = (p_next[0] - p_prev[0]) / 2.0
            dy = (p_next[1] - p_prev[1]) / 2.0
            ddx = p_next[0] - 2 * p_curr[0] + p_prev[0]
            ddy = p_next[1] - 2 * p_curr[1] + p_prev[1]

            denominator = (dx**2 + dy**2) ** 1.5
            if denominator > 1e-8:
                curvature = abs(dx * ddy - dy * ddx) / denominator
                curvatures.append(curvature)

        return np.array(curvatures) if curvatures else np.array([0.0])

    def _compute_illumination_consistency(self, gen_images: np.ndarray) -> float:
        """
        计算光照一致性得分 S_illum

        对生成图像中的缺陷区域施加简化的Phong光照模型，
        根据背景图像估计主光源方向，
        计算缺陷区域表面法向量与光源方向的角度偏差。
        """
        scores = []

        for img in gen_images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # 1. 估计主光源方向 (使用图像梯度)
            sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

            # 梯度方向反推光源方向 (假设朗伯表面)
            mean_grad_x = np.mean(sobel_x)
            mean_grad_y = np.mean(sobel_y)
            norm = np.sqrt(mean_grad_x**2 + mean_grad_y**2 + 1.0)
            if norm < 1e-8:
                light_dir = np.array([0.0, 0.0, 1.0])
            else:
                light_dir = np.array([-mean_grad_x / norm, -mean_grad_y / norm, 1.0 / norm])
                light_dir = light_dir / np.linalg.norm(light_dir)

            # 2. 估计表面法向量 (使用梯度)
            h, w = gray.shape
            # 采样点
            step = max(1, min(h, w) // 32)
            ys, xs = np.meshgrid(
                np.arange(0, h, step),
                np.arange(0, w, step),
                indexing='ij'
            )

            deviations = []
            for y, x in zip(ys.flat, xs.flat):
                if y < 1 or y >= h - 1 or x < 1 or x >= w - 1:
                    continue
                gx = sobel_x[y, x]
                gy = sobel_y[y, x]
                normal = np.array([-gx, -gy, 1.0])
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-8:
                    continue
                normal = normal / normal_norm

                # 3. 计算法向量与光源方向的角度偏差
                cos_angle = np.clip(np.dot(normal, light_dir), -1.0, 1.0)
                angle_deviation = np.arccos(cos_angle)
                deviations.append(angle_deviation)

            if deviations:
                # 使用角度偏差的标准差的倒数映射为一致性得分
                deviation_std = np.std(deviations)
                # 映射: 标准差越小, 得分越高
                score = np.exp(-deviation_std)
                scores.append(score)
            else:
                scores.append(0.0)

        s_illum = np.mean(scores) if scores else 0.0
        return float(np.clip(s_illum, 0.0, 1.0))

    def compute_class_wise(
        self,
        generated_by_class: Dict[str, torch.Tensor],
        real_by_class: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """
        按类别计算PPS

        Args:
            generated_by_class: {类别名: 生成图像张量}
            real_by_class: {类别名: 真实图像张量}

        Returns:
            {类别名: {pps, s_geo, s_illum}}
        """
        results = {}
        for category in generated_by_class:
            if category in real_by_class:
                results[category] = self.compute(
                    generated_by_class[category],
                    real_by_class[category]
                )
        return results
