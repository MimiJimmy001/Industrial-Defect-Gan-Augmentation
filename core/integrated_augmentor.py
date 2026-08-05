#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
集成增广器
直接使用训练好的Focus-StyleGAN模型进行图像增广
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Union

# Windows + Anaconda 下可能出现 OpenMP 运行时冲突，导致导入 torch/cv2 直接崩溃。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import cv2
import numpy as np
from PIL import Image

# 导入项目模块
try:
    from model.augmentation.augmentor import Augmentor
    from model.utils.config import Config
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"导入项目模块失败: {e}")
    print("请确保项目结构完整，且work目录包含src模块")
    IMPORT_SUCCESS = False

# 导入真实缺陷迁移模块
try:
    from core.real_defect_blender import RealDefectBlender, get_categories as get_real_categories
    REAL_DEFECT_AVAILABLE = True
except ImportError as e:
    print(f"真实缺陷模块导入失败: {e}")
    REAL_DEFECT_AVAILABLE = False


class IntegratedAugmentor:
    """集成增广器 - 简化接口"""

    def __init__(self, config_path: str = None, checkpoint_path: str = None, device: str = None):
        """
        初始化集成增广器

        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
            checkpoint_path: 检查点路径，如果为None则使用checkpoints/final_model.pth
            device: 设备类型，'cuda'或'cpu'，如果为None则自动选择
        """
        if not IMPORT_SUCCESS:
            raise ImportError("无法导入项目模块，请检查项目结构")

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")

        # 加载配置
        if config_path is None:
            # 尝试多个可能的配置路径
            config_candidates = [
                "integrated_config.yaml",
                "core/integrated_config.yaml",
                "work/configs/default.yaml",
                "configs/default.yaml"
            ]

            for candidate in config_candidates:
                if Path(candidate).exists():
                    config_path = candidate
                    break

            if config_path is None:
                raise FileNotFoundError("未找到配置文件，请指定config_path参数")

        print(f"加载配置: {config_path}")
        self.config = Config(config_path)

        # 设置检查点路径
        if checkpoint_path is None:
            checkpoint_candidates = [
                "checkpoints/final_model.pth",
                "checkpoints/best_model.pth",
                "work/checkpoints/final_model.pth"
            ]

            for candidate in checkpoint_candidates:
                if Path(candidate).exists():
                    checkpoint_path = candidate
                    break

            if checkpoint_path is None:
                raise FileNotFoundError("未找到模型检查点，请指定checkpoint_path参数")

        self.checkpoint_path = checkpoint_path

        # 初始化增广器
        logger = logging.getLogger("IntegratedAugmentor")
        logger.setLevel(logging.INFO)

        self.augmentor = Augmentor(
            config=self.config,
            device=self.device,
            logger=logger
        )

        # 加载模型
        print(f"加载模型: {self.checkpoint_path}")
        self.augmentor.load_model(Path(self.checkpoint_path))
        print("模型加载完成!")

    def augment_single_image(self, image_path: str, output_dir: str = None,
                           n_variations: int = 5, defect_intensity_range: tuple = (0.5, 1.5)) -> List[str]:
        """
        增广单张图像

        Args:
            image_path: 输入图像路径
            output_dir: 输出目录，如果为None则创建'outputs/augmented_single'目录
            n_variations: 生成的变体数量
            defect_intensity_range: 缺陷强度范围(min, max)

        Returns:
            生成的图像路径列表
        """
        if output_dir is None:
            output_dir = Path("outputs") / "augmented_single"
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # 使用原增广器的augment_image方法
        generated_paths = self.augmentor.augment_image(
            image_path=Path(image_path),
            output_dir=output_dir,
            n_variations=n_variations
        )

        print(f"生成 {len(generated_paths)} 张增广图像到: {output_dir}")
        return [str(p) for p in generated_paths]

    def augment_single_image_real(self, image_path: str, output_dir: str = None,
                                  n_variations: int = 5, category: str = None,
                                  n_defects: int = None, intensity: float = 1.0) -> List[str]:
        """
        使用真实 MVTec AD 缺陷图像进行增广（推荐）
        从数据集提取真实缺陷区域，经几何变换后融合到正常图像上

        Args:
            image_path: 输入正常(good)图像路径
            output_dir: 输出目录
            n_variations: 生成的变体数量
            category: 产品类别（如'bottle'），None则自动选择
            n_defects: 每张图像叠加的缺陷数量，None则随机 1~3
            intensity: 缺陷强度 (0.5~1.5)

        Returns:
            生成的图像路径列表
        """
        if not REAL_DEFECT_AVAILABLE:
            raise ImportError("真实缺陷模块不可用，请确保 real_defect_blender.py 存在")

        if output_dir is None:
            output_dir = Path("outputs") / "augmented_real"
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        blender = RealDefectBlender(category=category)
        generated_paths = blender.generate_batch(
            good_image_path=image_path,
            output_dir=str(output_dir),
            n_variations=n_variations,
            n_defects=n_defects,
            intensity=intensity,
        )

        print(f"真实缺陷增广: 生成 {len(generated_paths)} 张图像到: {output_dir}")
        print(f"  使用类别: {blender.category}")
        return [str(p) for p in generated_paths]

    def augment_single_retrieval(self, image_path: str, category: str,
                                  output_dir: str = None,
                                  n_variations: int = 5) -> List[str]:
        """
        检索式增广：从数据集同类型中按特征相似度选取最相近图像。

        Args:
            image_path: 输入图像路径
            category: 产品类别（如'bottle'）——必填
            output_dir: 输出目录
            n_variations: 检索数量

        Returns:
            输出图像路径列表
        """
        if output_dir is None:
            output_dir = Path("outputs") / "augmented_retrieval"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_paths = self.augmentor.augment_by_retrieval(
            image_path=Path(image_path),
            output_dir=output_dir,
            category=category,
            n_variations=n_variations,
        )

        print(f"检索式增广: 检索 {len(generated_paths)} 张图像到: {output_dir}")
        print(f"  使用类别: {category}")
        return [str(p) for p in generated_paths]

    def augment_single_stacking(self, image_path: str, category: str,
                                 output_dir: str = None,
                                 n_variations: int = 5,
                                 n_defects: int = None,
                                 intensity: float = 1.0) -> List[str]:
        """
        堆叠式增广：从同类型中选取不同属性缺陷，几何变换后堆叠融合。

        Args:
            image_path: 输入正常(good)图像路径
            category: 产品类别（如'bottle'）——必填
            output_dir: 输出目录
            n_variations: 生成变体数量
            n_defects: 每张图像堆叠缺陷数，None则随机1~3
            intensity: 缺陷强度

        Returns:
            输出图像路径列表
        """
        if output_dir is None:
            output_dir = Path("outputs") / "augmented_stacking"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_paths = self.augmentor.augment_by_stacking(
            image_path=Path(image_path),
            output_dir=output_dir,
            category=category,
            n_variations=n_variations,
            n_defects=n_defects,
            intensity=intensity,
        )

        print(f"堆叠式增广: 生成 {len(generated_paths)} 张图像到: {output_dir}")
        print(f"  使用类别: {category}")
        return [str(p) for p in generated_paths]

    def augment_dataset(self, input_dir: str, output_dir: str = None,
                       n_samples: int = None) -> Dict[str, int]:
        """
        增广整个数据集目录

        Args:
            input_dir: 输入目录
            output_dir: 输出目录，如果为None则创建'outputs/augmented_dataset'目录
            n_samples: 总样本数，如果为None则使用配置中的值

        Returns:
            增广统计信息
        """
        if output_dir is None:
            output_dir = Path("outputs") / "augmented_dataset"
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # 使用原增广器的augment_dataset方法
        stats = self.augmentor.augment_dataset(
            input_dir=Path(input_dir),
            output_dir=output_dir,
            n_samples=n_samples
        )

        print(f"数据集增广完成!")
        print(f"  输入目录: {input_dir}")
        print(f"  输出目录: {output_dir}")
        print(f"  生成总数: {stats.get('total_generated', 0)}")

        return stats

    def generate_with_custom_intensity(
        self,
        image_path: str,
        output_path: str,
        defect_intensity: float = 1.0,
        rng_seed: Optional[int] = None,
        image_features: Optional[Dict[str, float]] = None,
        defect_modulation: Optional[Dict[str, any]] = None,
        post_process: bool = True,
    ) -> str:
        """
        使用自定义缺陷强度生成伪异常图像（数据集缺陷类型驱动版）

        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            defect_intensity: 缺陷强度(0.0-2.0)
            rng_seed: 随机种子
            image_features: extract_image_features()输出，None则自动提取
            defect_modulation: defect_registry.get_modulation_params()输出
            post_process: 是否启用后处理

        Returns:
            输出图像路径
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_size = self.config.data.image_size
        image = cv2.resize(image, (target_size, target_size))

        image_tensor = torch.from_numpy(image).float() / 127.5 - 1.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        if image_features is None:
            image_features = self.augmentor.extract_image_features(image_tensor)

        with torch.no_grad():
            pseudo_anomaly = self.augmentor.generate_pseudo_anomaly(
                real_image=image_tensor,
                defect_intensity=defect_intensity,
                rng_seed=rng_seed,
                image_features=image_features,
                defect_modulation=defect_modulation,
            )

            pseudo_np = pseudo_anomaly.squeeze(0).cpu().numpy()
            pseudo_np = (pseudo_np + 1.0) / 2.0 * 255.0
            pseudo_np = pseudo_np.transpose(1, 2, 0).astype(np.uint8)

        # ---- 后处理：边缘柔化 + 微噪声（让结果更自然） ----
        if post_process and defect_intensity > 0.2:
            pseudo_np = self._post_process(pseudo_np, defect_intensity)

        cv2.imwrite(output_path, cv2.cvtColor(pseudo_np, cv2.COLOR_RGB2BGR))
        print(f"生成图像保存到: {output_path}")
        return output_path

    def _post_process(self, image_np: np.ndarray, intensity: float) -> np.ndarray:
        """
        后处理管线：边缘柔化 + 微量噪声 + 轻微锐化恢复

        Args:
            image_np: RGB图像 [H, W, 3], uint8
            intensity: 缺陷强度

        Returns:
            处理后的图像
        """
        from cv2 import bilateralFilter, GaussianBlur, addWeighted

        # 轻量双边滤波：保边去噪，仅轻度柔化缺陷边缘
        d = 5
        sigma_color = max(5, int(8 * intensity))
        sigma_space = max(3, int(5 * intensity))
        smoothed = bilateralFilter(image_np, d, sigma_color, sigma_space)

        # 微噪声注入（模拟传感器噪声，降低幅度避免掩盖缺陷）
        noise = np.random.randn(*image_np.shape).astype(np.float32) * (1.2 * intensity)
        noisy = np.clip(smoothed.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 轻微锐化恢复细节（增强纹理对比度）
        blur = GaussianBlur(noisy, (3, 3), 0)
        sharpened = addWeighted(noisy, 1.0 + 0.12 * intensity, blur, -0.12 * intensity, 0)

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def batch_augment(self, image_paths: List[str], output_dir: str,
                     n_variations_per_image: int = 3) -> Dict[str, List[str]]:
        """
        批量增广多张图像

        Args:
            image_paths: 输入图像路径列表
            output_dir: 输出目录
            n_variations_per_image: 每张图像的变体数量

        Returns:
            每张输入图像对应的生成图像路径字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for i, img_path in enumerate(image_paths):
            try:
                img_name = Path(img_path).stem
                img_output_dir = output_dir / img_name
                img_output_dir.mkdir(exist_ok=True)

                generated_paths = self.augment_single_image(
                    image_path=img_path,
                    output_dir=str(img_output_dir),
                    n_variations=n_variations_per_image
                )

                results[img_path] = generated_paths
                print(f"处理完成 [{i+1}/{len(image_paths)}]: {img_path} -> {len(generated_paths)} 个变体")

            except Exception as e:
                print(f"处理图像失败 {img_path}: {e}")
                results[img_path] = []

        return results

    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息"""
        config_path = "unknown"
        if hasattr(self.config, 'config_path'):
            config_path = str(self.config.config_path)

        return {
            "device": str(self.device),
            "config_path": config_path,
            "checkpoint_path": str(self.checkpoint_path),
            "image_size": self.config.data.image_size,
            "latent_dim": self.config.model.generator.latent_dim,
            "model_loaded": True
        }


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description="集成增广器 - 基于Focus-StyleGAN的图像增广系统")
    parser.add_argument("--mode", type=str, default="single",
                       choices=["single", "dataset", "batch", "info", "real", "retrieval", "stacking"],
                       help="运行模式: single(GAN), dataset(数据集), batch(批量), info(信息), "
                            "real(真实缺陷迁移), retrieval(相似度检索), stacking(缺陷堆叠)")
    parser.add_argument("--input", type=str, help="输入图像路径或目录")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径")
    parser.add_argument("--variations", type=int, default=5, help="每张图像的变体数量")
    parser.add_argument("--samples", type=int, default=None, help="数据集增广的样本数")
    parser.add_argument("--device", type=str, default=None, help="设备: cuda或cpu")
    parser.add_argument("--category", type=str, default=None, help="产品类别(真实缺陷模式, 如bottle)")
    parser.add_argument("--n_defects", type=int, default=None, help="叠加缺陷数量(真实缺陷模式)")
    parser.add_argument("--intensity", type=float, default=1.0, help="缺陷强度")

    args = parser.parse_args()

    try:
        # 检索/堆叠模式不需要GAN模型，直接处理
        if args.mode == "retrieval":
            if not args.input or not args.category:
                print("错误: 检索模式需要 --input 和 --category 参数")
                return
            from core.retrieval_augmentor import RetrievalAugmentor
            aug = RetrievalAugmentor()
            aug.build_index()
            paths = aug.augment_by_retrieval(
                args.input, args.output or "outputs/augmented_retrieval",
                args.category, n_variations=args.variations,
            )
            print(f"检索增广完成: {len(paths)} 张")
            return

        if args.mode == "stacking":
            if not args.input or not args.category:
                print("错误: 堆叠模式需要 --input 和 --category 参数")
                return
            from core.retrieval_augmentor import DefectStackingAugmentor
            aug = DefectStackingAugmentor(category=args.category)
            paths = aug.generate_batch(
                args.input, args.output or "outputs/augmented_stacking",
                n_variations=args.variations,
                n_defects=args.n_defects,
                intensity=args.intensity,
            )
            print(f"堆叠增广完成: {len(paths)} 张")
            return

        # 初始化增广器（GAN模式需要模型）
        augmentor = IntegratedAugmentor(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device
        )

        # 根据模式执行
        if args.mode == "single":
            if not args.input:
                print("错误: 单张图像模式需要--input参数指定图像路径")
                return

            augmentor.augment_single_image(
                image_path=args.input,
                output_dir=args.output,
                n_variations=args.variations
            )

        elif args.mode == "dataset":
            if not args.input:
                print("错误: 数据集模式需要--input参数指定目录路径")
                return

            augmentor.augment_dataset(
                input_dir=args.input,
                output_dir=args.output,
                n_samples=args.samples
            )

        elif args.mode == "batch":
            if not args.input:
                print("错误: 批量模式需要--input参数指定包含图像路径的文本文件")
                return

            # 从文本文件读取图像路径
            if Path(args.input).exists():
                with open(args.input, 'r') as f:
                    image_paths = [line.strip() for line in f if line.strip()]

                if not image_paths:
                    print("错误: 文本文件中没有有效的图像路径")
                    return

                augmentor.batch_augment(
                    image_paths=image_paths,
                    output_dir=args.output if args.output else "outputs/batch_augmented",
                    n_variations_per_image=args.variations
                )
            else:
                print(f"错误: 输入文件不存在: {args.input}")

        elif args.mode == "real":
            if not args.input:
                print("错误: 真实缺陷模式需要--input参数指定图像路径")
                return

            if not REAL_DEFECT_AVAILABLE:
                print("错误: 真实缺陷模块不可用")
                return

            augmentor.augment_single_image_real(
                image_path=args.input,
                output_dir=args.output,
                n_variations=args.variations,
                category=args.category,
                n_defects=args.n_defects,
                intensity=args.intensity,
            )

        elif args.mode == "info":
            info = augmentor.get_model_info()
            print("模型信息:")
            for key, value in info.items():
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()