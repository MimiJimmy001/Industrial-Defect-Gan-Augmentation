#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MVTec-AD数据集加载器
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MVTecADDataset(Dataset):
    """MVTec-AD数据集类"""

    def __init__(
        self,
        root_path: Union[str, Path],
        image_size: int = 256,
        augment: bool = True,
        mode: str = "train",
        category: str = None
    ):
        """
        初始化数据集

        Args:
            root_path: 数据集根目录
            image_size: 图像大小
            augment: 是否进行数据增强
            mode: 模式，"train"或"test"
            category: 类别名称，如果为None则加载所有类别
        """
        self.root_path = Path(root_path)
        self.image_size = image_size
        self.augment = augment
        self.mode = mode
        self.category = category

        # 检查数据集是否存在
        if not self.root_path.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.root_path}")

        # 获取所有类别
        self.categories = self._get_categories()

        # 如果指定了类别，只加载该类别
        if category is not None:
            if category not in self.categories:
                raise ValueError(f"类别不存在: {category}")
            self.categories = [category]

        # 加载图像路径和标签
        self.image_paths, self.labels = self._load_data()

        # 设置数据增强
        self.transform = self._get_transform()

    def _get_categories(self) -> List[str]:
        """获取所有类别"""
        categories = []
        for item in self.root_path.iterdir():
            if item.is_dir():
                categories.append(item.name)
        return sorted(categories)

    def _load_data(self) -> Tuple[List[Path], List[int]]:
        """
        加载图像路径和标签

        Returns:
            (图像路径列表, 标签列表), 0表示正常，1表示异常
        """
        image_paths = []
        labels = []

        for category in self.categories:
            category_path = self.root_path / category

            # 正常图像
            normal_path = category_path / self.mode / "good"
            if normal_path.exists():
                for img_file in normal_path.glob("*.png"):
                    image_paths.append(img_file)
                    labels.append(0)  # 正常

            # 异常图像（仅在test模式下）
            if self.mode == "test":
                anomaly_path = category_path / self.mode
                for anomaly_type in anomaly_path.iterdir():
                    if anomaly_type.name == "good":
                        continue
                    if anomaly_type.is_dir():
                        for img_file in anomaly_type.glob("*.png"):
                            image_paths.append(img_file)
                            labels.append(1)  # 异常

        return image_paths, labels

    def _get_transform(self) -> A.Compose:
        """获取数据增强变换"""
        if self.augment and self.mode == "train":
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])

    def __len__(self) -> int:
        """数据集大小"""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据项

        Args:
            idx: 索引

        Returns:
            包含图像和标签的字典
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 读取图像
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 应用变换
        transformed = self.transform(image=image)
        image_tensor = transformed["image"]

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "path": str(img_path)
        }

    def get_class_distribution(self) -> Dict[int, int]:
        """获取类别分布"""
        from collections import Counter
        return dict(Counter(self.labels))

    def split_dataset(self, train_ratio: float = 0.8, seed: int = 42) -> Tuple['MVTecADDataset', 'MVTecADDataset']:
        """
        分割数据集

        Args:
            train_ratio: 训练集比例
            seed: 随机种子（固定划分，保证实验可复现）

        Returns:
            (训练集, 验证集)
        """
        if self.mode != "train":
            raise ValueError("只能在train模式下分割数据集")

        n_total = len(self)
        n_train = int(n_total * train_ratio)
        indices = list(range(n_total))
        rng = random.Random(seed)
        rng.shuffle(indices)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # 创建训练集
        train_dataset = self._create_subset(train_indices)
        # 创建验证集（关闭数据增强）
        val_dataset = self._create_subset(val_indices)
        val_dataset.augment = False
        val_dataset.transform = val_dataset._get_transform()

        return train_dataset, val_dataset

    def _create_subset(self, indices: List[int]) -> 'MVTecADDataset':
        """创建子集"""
        subset = MVTecADDataset.__new__(MVTecADDataset)
        subset.root_path = self.root_path
        subset.image_size = self.image_size
        subset.augment = self.augment
        subset.mode = self.mode
        subset.category = self.category
        subset.categories = self.categories

        # 根据索引选择数据
        subset.image_paths = [self.image_paths[i] for i in indices]
        subset.labels = [self.labels[i] for i in indices]
        subset.transform = self.transform

        return subset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = None
) -> DataLoader:
    """
    创建数据加载器

    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        drop_last: 是否丢弃最后不足一批的样本；None 时跟随 shuffle
            （训练集丢弃以稳定 WGAN-GP 批量统计，验证集保留全部样本）

    Returns:
        数据加载器
    """
    if drop_last is None:
        drop_last = shuffle

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last
    )