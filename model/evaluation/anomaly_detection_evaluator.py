#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
异常检测评估模块
计算Pixel-AUC和PRO-AUC指标
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import cv2


class AnomalyDetectionEvaluator:
    """异常检测评估器"""

    def __init__(
        self,
        device: torch.device,
        logger: logging.Logger,
        backbone: str = "resnet18",
        feature_layers: List[str] = None
    ):
        """
        初始化评估器

        Args:
            device: 计算设备
            logger: 日志记录器
            backbone: 骨干网络类型
            feature_layers: 使用的特征层
        """
        self.device = device
        self.logger = logger

        # 加载预训练模型
        self.model = self._load_pretrained_model(backbone)
        self.model.eval()
        self.model.to(device)

        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 特征层
        if feature_layers is None:
            self.feature_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        else:
            self.feature_layers = feature_layers

        # 注册钩子以获取特征
        self.features = {}
        self._register_hooks()

    def _load_pretrained_model(self, backbone: str) -> nn.Module:
        """加载预训练模型，优先从本地加载"""
        from model.utils.weights import find_weight, load_state_dict_from_local

        if backbone not in ("resnet18", "resnet50", "wide_resnet50"):
            raise ValueError(f"不支持的骨干网络: {backbone}")

        model_map = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
            "wide_resnet50": models.wide_resnet50_2,
        }

        try:
            state_dict = load_state_dict_from_local(backbone, map_location="cpu")
            model = model_map[backbone](pretrained=False)
            model.load_state_dict(state_dict)
        except (FileNotFoundError, RuntimeError):
            model = model_map[backbone](pretrained=True)

        # 保留完整模块结构：hook 依赖 layer1~layer4 的模块名注册，
        # 不能用 nn.Sequential 拍平（否则 named_modules 只剩数字索引）
        return model

    def _register_hooks(self):
        """注册钩子以获取特征"""
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        registered = []
        for name, module in self.model.named_modules():
            if name in self.feature_layers:
                module.register_forward_hook(get_features(name))
                registered.append(name)

        missing = [l for l in self.feature_layers if l not in registered]
        if missing:
            raise RuntimeError(
                f"特征层hook注册失败: {missing}。"
                f"可用模块: {[n for n, _ in self.model.named_modules()][:20]}"
            )

    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取图像特征

        Args:
            images: 输入图像 [B, 3, H, W]

        Returns:
            各层特征字典
        """
        self._last_input_size = (images.shape[2], images.shape[3])
        with torch.no_grad():
            # 前向传播
            _ = self.model(images.to(self.device))
            # 复制特征并清除
            features = {k: v.cpu() for k, v in self.features.items()}
            self.features.clear()

        return features

    def compute_patch_distribution(
        self,
        normal_features: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        计算正常图像的块分布（均值和协方差）

        Args:
            normal_features: 正常图像特征

        Returns:
            每层的分布统计量
        """
        distributions = {}

        for layer_name, features in normal_features.items():
            # 特征形状: [B, C, H, W]
            batch_size, channels, height, width = features.shape

            # 重塑为 [B*H*W, C]
            features_flat = features.permute(0, 2, 3, 1).contiguous()
            features_flat = features_flat.view(-1, channels)

            # 计算均值和协方差
            mean = torch.mean(features_flat, dim=0)
            cov = torch.cov(features_flat.T)

            # 添加小的正则化项确保可逆
            cov = cov + 1e-6 * torch.eye(cov.shape[0])

            distributions[layer_name] = {
                'mean': mean,
                'cov': cov,
                'inv_cov': torch.inverse(cov)
            }

        return distributions

    def compute_anomaly_map(
        self,
        features: Dict[str, torch.Tensor],
        distributions: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        计算异常图

        Args:
            features: 测试图像特征
            distributions: 正常图像分布

        Returns:
            异常图 [H, W]
        """
        anomaly_maps = []
        # 异常图上采样的目标尺寸：使用最近一次 extract_features 的输入尺寸
        target_size = getattr(self, '_last_input_size', (256, 256))

        for layer_name in self.feature_layers:
            if layer_name not in features or layer_name not in distributions:
                continue

            feat = features[layer_name]
            dist = distributions[layer_name]

            # 特征形状: [1, C, H, W]
            batch_size, channels, height, width = feat.shape

            # 重塑为 [H*W, C]
            feat_flat = feat.permute(0, 2, 3, 1).contiguous()
            feat_flat = feat_flat.view(-1, channels)

            # 计算马氏距离
            diff = feat_flat - dist['mean']
            # 计算 (x - μ)^T Σ^{-1} (x - μ)
            mahalanobis = torch.sum(
                diff @ dist['inv_cov'] * diff,
                dim=1
            )
            mahalanobis = torch.sqrt(torch.clamp(mahalanobis, min=0.0))

            # 重塑为 [H, W]
            anomaly_map = mahalanobis.view(height, width)

            # 上采样到原始图像大小
            anomaly_map = F.interpolate(
                anomaly_map.unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze()

            anomaly_maps.append(anomaly_map)

        # 合并多尺度异常图
        if anomaly_maps:
            combined_map = torch.stack(anomaly_maps).mean(dim=0)
        else:
            self.logger.warning("未提取到任何特征层，异常图为空，请检查hook注册")
            combined_map = torch.zeros(target_size)

        # 高斯平滑
        combined_map_np = combined_map.cpu().numpy()
        combined_map_np = gaussian_filter(combined_map_np, sigma=4)
        combined_map = torch.from_numpy(combined_map_np)

        return combined_map

    def evaluate_dataset(
        self,
        normal_dataset: Dataset,
        anomaly_dataset: Dataset,
        n_normal_samples: int = 100,
        n_anomaly_samples: int = 100
    ) -> Dict[str, float]:
        """
        评估数据集

        Args:
            normal_dataset: 正常图像数据集
            anomaly_dataset: 异常图像数据集
            n_normal_samples: 用于训练的正常样本数
            n_anomaly_samples: 用于测试的异常样本数

        Returns:
            评估指标字典
        """
        self.logger.info("开始异常检测评估...")

        # 1. 提取正常图像特征以构建分布
        self.logger.info("提取正常图像特征...")
        normal_loader = DataLoader(
            normal_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0
        )

        normal_features_by_layer = {layer: [] for layer in self.feature_layers}
        normal_count = 0

        for batch in tqdm(normal_loader, desc="处理正常图像"):
            if normal_count >= n_normal_samples:
                break

            images = batch['image']
            features = self.extract_features(images)

            for layer_name, feat in features.items():
                normal_features_by_layer[layer_name].append(feat)

            normal_count += 1

        # 合并特征
        for layer_name in self.feature_layers:
            if normal_features_by_layer[layer_name]:
                normal_features_by_layer[layer_name] = torch.cat(
                    normal_features_by_layer[layer_name], dim=0
                )

        # 2. 计算正常分布
        self.logger.info("计算正常分布...")
        distributions = self.compute_patch_distribution(normal_features_by_layer)

        # 3. 评估异常图像
        self.logger.info("评估异常图像...")
        anomaly_loader = DataLoader(
            anomaly_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0
        )

        pixel_labels = []
        pixel_scores = []
        region_labels = []
        region_scores = []
        anomaly_maps_all = []
        masks_all = []

        anomaly_count = 0

        for batch in tqdm(anomaly_loader, desc="处理异常图像"):
            if anomaly_count >= n_anomaly_samples:
                break

            images = batch['image']
            # 假设数据集中有mask
            if 'mask' in batch:
                mask = batch['mask'].squeeze().cpu().numpy()
            else:
                # 如果没有mask，创建全零mask（正常区域）
                mask = np.zeros((images.shape[2], images.shape[3]))

            # 提取特征
            features = self.extract_features(images)

            # 计算异常图
            anomaly_map = self.compute_anomaly_map(features, distributions)

            # 准备Pixel-AUC数据
            map_np = anomaly_map.cpu().numpy().flatten()
            mask_flat = mask.flatten()

            # 确保数据有效
            valid_indices = ~np.isnan(map_np)
            if np.sum(valid_indices) > 0:
                pixel_scores.extend(map_np[valid_indices].tolist())
                pixel_labels.extend(mask_flat[valid_indices].tolist())

                # 存储完整的异常图和掩码用于PRO-AUC计算
                anomaly_maps_all.append(anomaly_map.cpu().numpy())
                masks_all.append(mask)

                # 计算区域级分数（平均异常分数）
                if np.any(mask > 0):
                    # 异常区域
                    anomaly_region_score = np.mean(map_np[mask_flat > 0])
                    region_scores.append(anomaly_region_score)
                    region_labels.append(1)

                    # 正常区域（从图像中随机采样）
                    normal_indices = np.where(mask_flat == 0)[0]
                    if len(normal_indices) > 0:
                        sampled_normal = np.random.choice(
                            normal_indices,
                            min(1000, len(normal_indices)),
                            replace=False
                        )
                        normal_region_score = np.mean(map_np[sampled_normal])
                        region_scores.append(normal_region_score)
                        region_labels.append(0)

            anomaly_count += 1

        self.logger.info(f"处理了 {anomaly_count} 张异常图像")
        self.logger.info(f"像素级样本数: {len(pixel_scores)}")
        self.logger.info(f"区域级样本数: {len(region_scores)}")

        # 4. 计算指标
        metrics = {}

        if len(pixel_scores) > 0 and len(np.unique(pixel_labels)) > 1:
            pixel_auc = roc_auc_score(pixel_labels, pixel_scores)
            metrics['pixel_auc'] = pixel_auc
            self.logger.info(f"Pixel-AUC: {pixel_auc:.4f}")
        else:
            metrics['pixel_auc'] = 0.0
            self.logger.warning("Pixel-AUC计算失败，数据不足")

        if len(region_scores) > 0 and len(np.unique(region_labels)) > 1:
            region_auc = roc_auc_score(region_labels, region_scores)
            metrics['region_auc'] = region_auc
            self.logger.info(f"Region-AUC: {region_auc:.4f}")
        else:
            metrics['region_auc'] = 0.0
            self.logger.warning("Region-AUC计算失败，数据不足")

        # 5. 计算PRO (Per-Region Overlap) 分数
        metrics['pro_auc'] = self._compute_pro_score(
            anomaly_maps_all, masks_all, integration_limit=0.3
        )

        self.logger.info("异常检测评估完成!")
        return metrics

    def _compute_pro_score(
        self,
        anomaly_maps: List[np.ndarray],
        ground_truth_masks: List[np.ndarray],
        integration_limit: float = 0.3,
        n_thresholds: int = 200
    ) -> float:
        """
        计算PRO (Per-Region Overlap) 分数

        PRO-AUC计算每个连通异常区域的检测率，然后对所有区域取平均。
        这是MVTec AD官方推荐的指标，比Pixel-AUC更能反映实际检测效果。

        Args:
            anomaly_maps: 异常图列表 [H, W]
            ground_truth_masks: 真值掩码列表 [H, W]，值为0或1
            integration_limit: FPR积分上限，默认0.3
            n_thresholds: 阈值数量

        Returns:
            PRO-AUC值
        """
        if len(anomaly_maps) == 0 or len(ground_truth_masks) == 0:
            return 0.0

        # 收集所有连通区域
        all_region_scores = []  # 每个区域在阈值下的检测率

        for anomaly_map, gt_mask in zip(anomaly_maps, ground_truth_masks):
            # 确保尺寸一致
            if anomaly_map.shape != gt_mask.shape:
                from scipy.ndimage import zoom
                scale_h = gt_mask.shape[0] / anomaly_map.shape[0]
                scale_w = gt_mask.shape[1] / anomaly_map.shape[1]
                anomaly_map = zoom(anomaly_map, (scale_h, scale_w), order=1)

            # 查找真值中的连通区域
            from scipy.ndimage import label
            labeled, n_regions = label(gt_mask > 0.5)

            if n_regions == 0:
                continue

            # 对每个连通区域计算检测率曲线
            thresholds = np.linspace(anomaly_map.min(), anomaly_map.max(), n_thresholds)

            for region_id in range(1, n_regions + 1):
                region_mask = (labeled == region_id)
                region_size = np.sum(region_mask)

                if region_size < 4:  # 忽略过小的区域
                    continue

                # 在不同阈值下计算该区域的检测率
                detection_rates = []
                for thresh in thresholds:
                    binary_pred = (anomaly_map >= thresh).astype(np.float64)
                    # 区域内的正确检测比例
                    detected = np.sum(binary_pred * region_mask)
                    detection_rate = detected / region_size
                    detection_rates.append(detection_rate)

                all_region_scores.append(np.array(detection_rates))

        if len(all_region_scores) == 0:
            return 0.0

        # 计算全局FPR (基于所有像素)
        # 对每个阈值，计算平均检测率和全局FPR
        thresholds_global = np.linspace(
            np.mean([am.min() for am in anomaly_maps]),
            np.mean([am.max() for am in anomaly_maps]),
            n_thresholds
        )

        mean_detection_rates = []
        fpr_values = []

        # 收集所有正常像素
        all_normal_mask = np.concatenate([
            (1 - gt).flatten() for gt in ground_truth_masks
        ])
        all_anomaly_scores = np.concatenate([
            am.flatten() for am in anomaly_maps
        ])

        for thresh in thresholds_global:
            # 平均每个区域的检测率
            region_detection_sum = 0.0
            for region_scores in all_region_scores:
                # 找到最接近的阈值索引
                idx = np.argmin(np.abs(thresholds - thresh))
                region_detection_sum += region_scores[min(idx, len(region_scores) - 1)]

            mean_dr = region_detection_sum / len(all_region_scores)
            mean_detection_rates.append(mean_dr)

            # 全局FPR
            binary_global = (all_anomaly_scores >= thresh).astype(np.float64)
            fp = np.sum(binary_global * all_normal_mask)
            tn = np.sum((1 - binary_global) * all_normal_mask)
            fpr = fp / (fp + tn + 1e-8)
            fpr_values.append(fpr)

        mean_detection_rates = np.array(mean_detection_rates)
        fpr_values = np.array(fpr_values)

        # 对有限制的FPR范围积分 (0 到 integration_limit)
        valid_indices = fpr_values <= integration_limit
        if np.sum(valid_indices) < 2:
            return 0.0

        fpr_limited = fpr_values[valid_indices]
        dr_limited = mean_detection_rates[valid_indices]

        # 归一化积分
        if fpr_limited[-1] > 0:
            pro_auc = np.trapz(dr_limited, fpr_limited) / integration_limit
        else:
            pro_auc = 0.0

        return float(np.clip(pro_auc, 0.0, 1.0))

    def evaluate_augmentation_effect(
        self,
        original_dataset: Dataset,
        augmented_dataset: Dataset,
        anomaly_dataset: Dataset,
        n_samples: int = 50
    ) -> Dict[str, Dict[str, float]]:
        """
        评估增广效果

        Args:
            original_dataset: 原始数据集（正常图像）
            augmented_dataset: 增广数据集（包含增广的正常图像）
            anomaly_dataset: 异常数据集
            n_samples: 每类样本数

        Returns:
            评估结果对比
        """
        self.logger.info("评估增广效果...")

        results = {}

        # 1. 仅使用原始数据集训练
        self.logger.info("基准模型（仅原始数据）...")
        baseline_metrics = self.evaluate_dataset(
            original_dataset, anomaly_dataset,
            n_normal_samples=n_samples, n_anomaly_samples=n_samples
        )
        results['baseline'] = baseline_metrics

        # 2. 使用增广后数据集训练
        self.logger.info("增广后模型（原始+增广数据）...")

        # 合并原始和增广数据集
        class CombinedDataset(Dataset):
            def __init__(self, dataset1, dataset2):
                self.dataset1 = dataset1
                self.dataset2 = dataset2

            def __len__(self):
                return len(self.dataset1) + len(self.dataset2)

            def __getitem__(self, idx):
                if idx < len(self.dataset1):
                    return self.dataset1[idx]
                else:
                    return self.dataset2[idx - len(self.dataset1)]

        combined_dataset = CombinedDataset(original_dataset, augmented_dataset)

        augmented_metrics = self.evaluate_dataset(
            combined_dataset, anomaly_dataset,
            n_normal_samples=n_samples, n_anomaly_samples=n_samples
        )
        results['augmented'] = augmented_metrics

        # 3. 计算改进百分比
        improvement = {}
        for metric in ['pixel_auc', 'region_auc', 'pro_auc']:
            if metric in baseline_metrics and metric in augmented_metrics:
                baseline_val = baseline_metrics[metric]
                augmented_val = augmented_metrics[metric]
                if baseline_val > 0:
                    improvement[metric] = (augmented_val - baseline_val) / baseline_val * 100
                else:
                    improvement[metric] = 0.0

        results['improvement_percent'] = improvement

        self.logger.info("增广效果评估完成!")
        return results