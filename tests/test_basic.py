#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基本功能测试
"""

import sys
import os
import torch
import yaml
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.models.focus_stylegan import FocusStyleGAN
from model.data.loader import MVTecADDataset
from model.utils.config import Config
from model.utils.logger import setup_logger


def test_config():
    """测试配置加载"""
    print("测试配置加载...")
    try:
        config = Config(str(Path(__file__).resolve().parent.parent / "core" / "integrated_config.yaml"))
        print(f"✓ 配置加载成功")
        print(f"  图像尺寸: {config.data.image_size}")
        print(f"  批大小: {config.data.batch_size}")
        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False


def test_model():
    """测试模型创建"""
    print("测试模型创建...")
    try:
        config = Config(str(Path(__file__).resolve().parent.parent / "core" / "integrated_config.yaml"))
        model = FocusStyleGAN(config=config.model)

        # 测试前向传播
        batch_size = 2
        real_images = torch.randn(batch_size, 3, 256, 256)
        z_defect = torch.randn(batch_size, config.model.generator.latent_dim)
        z_background = torch.randn(batch_size, config.model.generator.latent_dim)

        outputs = model(real_images, z_defect, z_background)

        print(f"✓ 模型创建成功")
        print(f"  生成图像形状: {outputs['fused_images'].shape}")
        print(f"  判别器输出形状: {outputs['d_real'].shape}")
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False


def test_dataset():
    """测试数据集加载"""
    print("测试数据集加载...")
    try:
        # 创建测试数据目录结构
        test_data_dir = Path("test_data")
        test_data_dir.mkdir(exist_ok=True)

        # 创建最小化的数据集结构
        category_dir = test_data_dir / "bottle" / "train" / "good"
        category_dir.mkdir(parents=True, exist_ok=True)

        # 创建虚拟图像文件
        import numpy as np
        from PIL import Image

        for i in range(5):
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(category_dir / f"test_{i}.png")

        # 测试数据集
        dataset = MVTecADDataset(
            root_path=test_data_dir,
            image_size=256,
            augment=False
        )

        print(f"✓ 数据集加载成功")
        print(f"  数据集大小: {len(dataset)}")

        # 测试数据项
        item = dataset[0]
        print(f"  图像形状: {item['image'].shape}")
        print(f"  标签: {item['label']}")

        # 清理测试数据
        import shutil
        shutil.rmtree(test_data_dir)

        return True
    except Exception as e:
        print(f"✗ 数据集测试失败: {e}")
        return False


def test_training_components():
    """测试训练组件"""
    print("测试训练组件...")
    try:
        config = Config(str(Path(__file__).resolve().parent.parent / "core" / "integrated_config.yaml"))

        # 测试损失函数
        from model.training.trainer import PerceptualLoss

        perceptual_loss = PerceptualLoss()
        img1 = torch.randn(1, 3, 256, 256)
        img2 = torch.randn(1, 3, 256, 256)
        loss = perceptual_loss(img1, img2)

        print(f"✓ 训练组件测试成功")
        print(f"  感知损失值: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ 训练组件测试失败: {e}")
        return False


def test_logger():
    """测试日志系统"""
    print("测试日志系统...")
    try:
        logger = setup_logger("test_logger")
        logger.info("测试日志信息")
        logger.warning("测试日志警告")
        logger.error("测试日志错误")

        print(f"✓ 日志系统测试成功")
        return True
    except Exception as e:
        print(f"✗ 日志系统测试失败: {e}")
        return False


def test_augmentation():
    """测试增广模块"""
    print("测试增广模块...")
    try:
        config = Config(str(Path(__file__).resolve().parent.parent / "core" / "integrated_config.yaml"))

        # 测试增广器创建
        from model.augmentation.augmentor import Augmentor
        import logging

        logger = logging.getLogger("test")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        augmentor = Augmentor(config, device, logger)

        print(f"✓ 增广模块测试成功")
        print(f"  设备: {device}")
        return True
    except Exception as e:
        print(f"✗ 增广模块测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Focus-StyleGAN 增广系统测试")
    print("=" * 60)

    tests = [
        ("配置加载", test_config),
        ("模型创建", test_model),
        ("数据集加载", test_dataset),
        ("训练组件", test_training_components),
        ("日志系统", test_logger),
        ("增广模块", test_augmentation),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            results.append((test_name, False))

    # 总结结果
    print("\n" + "=" * 60)
    print("测试总结:")
    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {test_name}: {status}")

    print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n所有测试通过! 系统基本功能正常。")
        return 0
    else:
        print(f"\n{total - passed} 个测试失败，请检查问题。")
        return 1


if __name__ == "__main__":
    sys.exit(main())