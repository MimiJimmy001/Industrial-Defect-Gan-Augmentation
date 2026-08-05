#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增广器模块
生成伪异常图像并增广数据集
"""

import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# 可选导入
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    lpips = None

from model.models.focus_stylegan import FocusStyleGAN
from model.utils.config import Config

# 导入缺陷类型注册表
_root = Path(__file__).parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
try:
    from core.defect_registry import DEFECT_MODULATION
except ImportError:
    DEFECT_MODULATION = {}


class Augmentor:
    """图像增广器"""

    def __init__(
        self,
        config: Config,
        device: torch.device,
        logger: logging.Logger
    ):
        """
        初始化增广器

        Args:
            config: 配置
            device: 设备
            logger: 日志记录器
        """
        self.config = config
        self.device = device
        self.logger = logger

        # 创建模型
        self.model = FocusStyleGAN(config=config.model).to(device)
        self.model.eval()

        # 增广参数
        self.n_samples_per_class = config.augmentation.n_samples_per_class

        # 预计算高斯核（设备无关，后续 .to(device) 即可）
        self._gaussian_kernels: Dict[int, torch.Tensor] = {}

    def _get_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """获取或创建高斯模糊核"""
        cache_key = kernel_size * 100 + int(sigma * 10)
        if cache_key not in self._gaussian_kernels:
            ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
            gauss_1d = gauss_1d / gauss_1d.sum()
            kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]
            kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
            self._gaussian_kernels[cache_key] = kernel_2d
        return self._gaussian_kernels[cache_key]

    def _gaussian_blur(self, x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        """对单通道特征图做高斯模糊"""
        kernel = self._get_gaussian_kernel(kernel_size, sigma).to(x.device)
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding)

    def _compute_edge_map(self, x: torch.Tensor) -> torch.Tensor:
        """Sobel边缘检测，返回 [0,1] 边缘强度图"""
        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            device=x.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
            device=x.device
        ).view(1, 1, 3, 3)

        # 转灰度
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        edges_x = F.conv2d(gray, sobel_x, padding=1)
        edges_y = F.conv2d(gray, sobel_y, padding=1)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-8)
        # 自适应归一化
        edge_max = edges.max()
        if edge_max > 1e-6:
            edges = edges / edge_max
        return torch.tanh(edges * 2.0)  # 压缩到 [0, ~0.96]

    @staticmethod
    def classify_defect_from_diff(diff_map: np.ndarray, min_confidence: float = 0.05) -> Dict[str, any]:
        """
        基于真实图像处理技术分析差异图，识别缺陷类型。

        统一实现见 core/defect_classification.py（本方法为兼容包装）。

        Args:
            diff_map: 差异图 [H, W] float, 0-255
            min_confidence: 最小置信度阈值

        Returns:
            {'primary_type', 'confidence', 'scores': {各类型分数}, 'location'}
        """
        try:
            from core.defect_classification import classify_defect_from_diff as _classify
        except ImportError:
            _root = Path(__file__).parents[2]
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from core.defect_classification import classify_defect_from_diff as _classify
        return _classify(diff_map, min_confidence=min_confidence)

    @staticmethod
    def extract_image_features(image_tensor: torch.Tensor) -> Dict[str, float]:
        """
        从输入图像提取特征，用于指导缺陷生成方向。

        统一实现见 core/features.py（本方法为张量输入的兼容包装）。

        Args:
            image_tensor: [1, 3, H, W] 范围 [-1, 1]

        Returns:
            {'edge_density', 'texture_complexity', 'brightness_mean',
             'brightness_std', 'dominant_orientation', 'surface_type'}
        """
        try:
            from core.features import build_features_from_rgb
        except ImportError:
            _root = Path(__file__).parents[2]
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from core.features import build_features_from_rgb

        img = image_tensor.detach().clone()
        # 转到 [0, 1] 范围方便处理
        img_01 = (img * 0.5 + 0.5).clamp(0, 1)
        np_img = (img_01.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        features = build_features_from_rgb(np_img)
        return {k: round(v, 4) for k, v in features.items()}

    def _modulate_latent_by_features(
        self,
        z: torch.Tensor,
        features: Dict[str, float],
        defect_modulation: Optional[Dict[str, any]] = None,
    ) -> torch.Tensor:
        """
        缺陷类型作为 PRIMARY 信号，图像特征仅做 mild fine-tuning。
        不同 pattern 激活 z 的完全不同的区域，确保生成结果肉眼可区分。

        Args:
            z: 原始随机潜在向量 [1, latent_dim]
            features: extract_image_features 的输出
            defect_modulation: 来自 defect_registry.get_modulation_params()
        """
        z_mod = z.clone()
        D = z.shape[1]
        # 将 latent 分成 5 段
        seg = D // 5

        # ---- 1. 缺陷类型主导调制（不同 pattern 激活动不同段） ----
        if defect_modulation:
            pattern = defect_modulation.get('pattern', 'mixed')
            amp = float(defect_modulation.get('z_amplify', 1.1))
            sharp = float(defect_modulation.get('z_sharpen', 0.55))
            spread = float(defect_modulation.get('z_spread', 0.4))
        else:
            pattern = 'mixed'
            amp = 1.1
            sharp = 0.55
            spread = 0.4

        # 每种 pattern 在不同段上施加完全不同的放大/抑制
        if pattern == 'directional':
            # 划痕/切割：强化高频方向性，抑制扩散
            z_mod[:, :seg]           *= 1.6 + sharp * 2.0
            z_mod[:, seg:2*seg]      *= 1.8 + amp * 1.6
            z_mod[:, 2*seg:3*seg]    *= 0.4
            z_mod[:, 3*seg:4*seg]    *= 0.45
            z_mod[:, 4*seg:]         *= 0.5

        elif pattern == 'localized':
            # 孔洞/穿刺：局部高振幅斑点
            z_mod[:, :seg]           *= 0.45
            z_mod[:, seg:2*seg]      *= 0.45
            z_mod[:, 2*seg:3*seg]    *= 1.8 + amp * 2.0
            z_mod[:, 3*seg:4*seg]    *= 1.6 + sharp * 1.5
            z_mod[:, 4*seg:]         *= 0.5

        elif pattern == 'diffuse':
            # 污染/颜色/油污：宽扩散，低振幅
            z_mod[:, :seg]           *= 0.55
            z_mod[:, seg:2*seg]      *= 0.55
            z_mod[:, 2*seg:3*seg]    *= 0.55
            z_mod[:, 3*seg:4*seg]    *= 1.6 + spread * 2.0
            z_mod[:, 4*seg:]         *= 1.3 + amp * 1.3

        elif pattern == 'structural':
            # 弯曲/折叠/挤压：宽域结构变化
            z_mod[:, :seg]           *= 1.4 + amp * 1.4
            z_mod[:, seg:2*seg]      *= 0.6
            z_mod[:, 2*seg:3*seg]    *= 1.6 + spread * 1.6
            z_mod[:, 3*seg:4*seg]    *= 1.8 + amp * 1.4
            z_mod[:, 4*seg:]         *= 0.65

        elif pattern == 'sharp_local':
            # 断裂/裂纹：锐利局部变化
            z_mod[:, :seg]           *= 1.8 + sharp * 2.2
            z_mod[:, seg:2*seg]      *= 1.6 + amp * 1.6
            z_mod[:, 2*seg:3*seg]    *= 0.4
            z_mod[:, 3*seg:4*seg]    *= 0.4
            z_mod[:, 4*seg:]         *= 1.6 + amp * 1.3

        elif pattern == 'surface':
            # 印刷/表面：细微表面变化
            z_mod[:, :seg]           *= 0.6
            z_mod[:, seg:2*seg]      *= 0.6
            z_mod[:, 2*seg:3*seg]    *= 0.65
            z_mod[:, 3*seg:4*seg]    *= 1.3 + spread * 1.3
            z_mod[:, 4*seg:]         *= 0.6

        elif pattern == 'missing':
            # 缺失：大面积结构空洞
            z_mod[:, :seg]           *= 0.5
            z_mod[:, seg:2*seg]      *= 0.5
            z_mod[:, 2*seg:3*seg]    *= 1.8 + amp * 2.0
            z_mod[:, 3*seg:4*seg]    *= 1.6 + spread * 1.6
            z_mod[:, 4*seg:]         *= 1.6 + amp * 1.4

        else:  # mixed
            z_mod[:, :seg]           *= 1.0 + amp * 0.7
            z_mod[:, seg:2*seg]      *= 1.0 + sharp * 0.9
            z_mod[:, 2*seg:3*seg]    *= 1.0 + spread * 0.9
            z_mod[:, 3*seg:4*seg]    *= 1.0 + amp * 0.6
            z_mod[:, 4*seg:]         *= 1.0 + sharp * 0.6

        # ---- 2. 图像特征微调（仅 10-20% 的额外调制） ----
        edge = float(features.get('edge_density', 0.1))
        tex = float(features.get('texture_complexity', 0.2))
        z_mod[:, :seg]           *= 1.0 + edge * 0.15
        z_mod[:, 3*seg:4*seg]    *= 1.0 + tex * 0.15

        return z_mod

    def load_model(self, checkpoint_path: Path):
        """
        加载训练好的模型

        Args:
            checkpoint_path: 检查点路径
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"模型检查点不存在: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # strict=False: 允许加载旧版检查点（例如不含CBAM的判别器权重）
        # 缺失的权重（如新增的CBAM层）将使用随机初始化，不影响生成器推理
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            self.logger.warning(f"检查点中缺少的权重键 ({len(missing_keys)} 个)，将使用随机初始化: "
                               f"{missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            self.logger.warning(f"检查点中多余的权重键 ({len(unexpected_keys)} 个)，已忽略: "
                               f"{unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

        self.logger.info(f"加载模型: {checkpoint_path}")

    def generate_pseudo_anomaly(
        self,
        real_image: torch.Tensor,
        defect_intensity: float = 1.0,
        rng_seed: Optional[int] = None,
        image_features: Optional[Dict[str, float]] = None,
        defect_modulation: Optional[Dict[str, any]] = None,
    ) -> torch.Tensor:
        """
        生成伪异常图像（特征 + 数据集缺陷类型联合调制版）

        Args:
            real_image: 真实图像 [1, 3, H, W]
            defect_intensity: 缺陷强度 (0.0-2.0)
            rng_seed: 随机种子
            image_features: extract_image_features()的输出
            defect_modulation: defect_registry.get_modulation_params()的输出

        Returns:
            伪异常图像
        """
        with torch.no_grad():
            if image_features is None:
                image_features = self.extract_image_features(real_image)

            if rng_seed is not None:
                g = torch.Generator()
                g.manual_seed(int(rng_seed) % (2**31 - 1))
                z_defect = torch.randn(
                    1, self.config.model.generator.latent_dim,
                    generator=g,
                ).to(self.device)
                z_background = torch.randn(
                    1, self.config.model.generator.latent_dim,
                    generator=g,
                ).to(self.device)
            else:
                z_defect = torch.randn(1, self.config.model.generator.latent_dim, device=self.device)
                z_background = torch.randn(1, self.config.model.generator.latent_dim, device=self.device)

            z_defect = self._modulate_latent_by_features(z_defect, image_features, defect_modulation)

            # ---- 2. 模型前向 ----
            defect_image = self.model.defect_branch(z_defect)
            background_image = self.model.background_branch(real_image, z_background)
            fused_image = self.model.fusion_module(defect_image, background_image)

            # ---- 3. 改进的缺陷遮罩计算 ----
            if defect_intensity <= 0.05:
                return background_image

            defect_for_mask = defect_image
            if defect_for_mask.shape[-2:] != background_image.shape[-2:]:
                defect_for_mask = F.interpolate(
                    defect_for_mask,
                    size=background_image.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            channel_w = torch.tensor([0.299, 0.587, 0.114], device=self.device).view(1, 3, 1, 1)
            diff = (defect_for_mask - background_image).abs()
            weighted_diff = (diff * channel_w).sum(dim=1, keepdim=True)
            smoothed_diff = self._gaussian_blur(weighted_diff, kernel_size=7, sigma=1.5)

            temperature = 0.03 + 0.08 / max(defect_intensity, 0.1)
            defect_mask = torch.sigmoid((smoothed_diff - 0.05) / temperature)

            edge_map = self._compute_edge_map(real_image)
            edge_protection = 1.0 - edge_map * 0.12
            defect_mask = defect_mask * edge_protection

            effective_intensity = float(np.clip(defect_intensity, 0.1, 1.8))
            defect_mask = torch.clamp(defect_mask * effective_intensity, 0.0, 0.95)

            # ---- 4. 三路混合 ----
            defect_aligned = defect_image
            if defect_aligned.shape[-2:] != background_image.shape[-2:]:
                defect_aligned = F.interpolate(
                    defect_aligned,
                    size=background_image.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            blend_alpha = torch.clamp(defect_mask * 1.15, 0.0, 0.92)
            pseudo_anomaly = (
                background_image * (1.0 - blend_alpha)
                + fused_image * blend_alpha * 0.78
                + defect_aligned * blend_alpha * 0.22
            )

            # ---- 5. 纹理噪声 ----
            if defect_intensity > 0.3:
                noise_level = 0.015 * defect_intensity
                texture_noise = torch.randn_like(pseudo_anomaly) * noise_level
                noise_mask = blend_alpha.repeat(1, 3, 1, 1)
                pseudo_anomaly = pseudo_anomaly + texture_noise * noise_mask

            # ---- 6. 颜色校正：仅均值微调，保留缺陷纹理的色彩和对比度 ----
            input_mean = real_image.mean(dim=[2, 3], keepdim=True)  # [1, 3, 1, 1]
            output_mean = pseudo_anomaly.mean(dim=[2, 3], keepdim=True)
            # 80% 保留生成结果，20% 拉回输入均值（仅校正极端色偏）
            pseudo_anomaly = pseudo_anomaly * 0.80 + (pseudo_anomaly - output_mean + input_mean) * 0.20
            pseudo_anomaly = torch.clamp(pseudo_anomaly, -1.0, 1.0)

            return pseudo_anomaly

    def augment_image(
        self,
        image_path: Path,
        output_dir: Path,
        n_variations: int = 5
    ) -> List[Path]:
        """
        增广单张图像

        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            n_variations: 生成变体数量

        Returns:
            生成的图像路径列表
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.warning(f"无法读取图像: {image_path}")
            return []

        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整大小
        target_size = self.config.data.image_size
        image = cv2.resize(image, (target_size, target_size))

        # 转换为张量
        image_tensor = torch.from_numpy(image).float() / 127.5 - 1.0  # [-1, 1]
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        generated_paths = []

        # 提取一次图像特征，多张增广共用
        features = self.extract_image_features(image_tensor)

        defect_type_names = list(DEFECT_MODULATION.keys()) if DEFECT_MODULATION else []
        for i in range(n_variations):
            defect_intensity = random.uniform(0.6, 1.4)
            # 随机选择缺陷类型，让每种变体产生不同视觉特征
            defect_modulation = None
            if defect_type_names:
                defect_modulation = DEFECT_MODULATION[random.choice(defect_type_names)]

            pseudo_anomaly = self.generate_pseudo_anomaly(
                image_tensor, defect_intensity,
                image_features=features,
                defect_modulation=defect_modulation,
            )

            # 转换为图像
            pseudo_np = pseudo_anomaly.squeeze(0).cpu().numpy()
            pseudo_np = (pseudo_np + 1.0) / 2.0 * 255.0  # 从[-1, 1]到[0, 255]
            pseudo_np = pseudo_np.transpose(1, 2, 0).astype(np.uint8)

            # 保存图像
            output_path = output_dir / f"{image_path.stem}_aug_{i}.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(pseudo_np, cv2.COLOR_RGB2BGR))

            generated_paths.append(output_path)

        return generated_paths

    def augment_by_retrieval(
        self,
        image_path: Path,
        output_dir: Path,
        category: str,
        n_variations: int = 5,
    ) -> List[Path]:
        """
        检索式增广：从数据集中按特征相似度选最相近的同类型图像直接输出。

        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            category: 产品类别 (如 'bottle')
            n_variations: 检索数量

        Returns:
            输出图像路径列表
        """
        try:
            from core.retrieval_augmentor import RetrievalAugmentor
        except ImportError:
            _root = Path(__file__).parents[2]
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from core.retrieval_augmentor import RetrievalAugmentor

        ret_aug = RetrievalAugmentor()
        ret_aug.build_index()
        paths = ret_aug.augment_by_retrieval(
            str(image_path), str(output_dir), category, n_variations=n_variations,
        )
        return [Path(p) for p in paths]

    def augment_by_stacking(
        self,
        image_path: Path,
        output_dir: Path,
        category: str,
        n_variations: int = 5,
        n_defects: Optional[int] = None,
        intensity: float = 1.0,
    ) -> List[Path]:
        """
        缺陷堆叠式增广：从同类型中选取不同属性缺陷，几何变换后堆叠融合。

        Args:
            image_path: 输入正常(good)图像路径
            output_dir: 输出目录
            category: 产品类别
            n_variations: 生成变体数量
            n_defects: 每张图像堆叠的缺陷数，None则随机1~3
            intensity: 缺陷强度

        Returns:
            输出图像路径列表
        """
        try:
            from core.retrieval_augmentor import DefectStackingAugmentor
        except ImportError:
            _root = Path(__file__).parents[2]
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from core.retrieval_augmentor import DefectStackingAugmentor

        stack_aug = DefectStackingAugmentor(category=category)
        paths = stack_aug.generate_batch(
            str(image_path), str(output_dir),
            n_variations=n_variations, n_defects=n_defects, intensity=intensity,
        )
        return [Path(p) for p in paths]

    def augment_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        n_samples: Optional[int] = None
    ) -> Dict[str, int]:
        """
        增广整个数据集

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            n_samples: 每类样本数，如果为None则使用配置中的值

        Returns:
            增广统计信息
        """
        if n_samples is None:
            n_samples = self.n_samples_per_class

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 查找所有图像
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_dir.glob(f"*{ext}"))
            image_paths.extend(input_dir.glob(f"*{ext.upper()}"))

        if not image_paths:
            self.logger.warning(f"在目录中未找到图像: {input_dir}")
            return {'total_generated': 0}

        self.logger.info(f"找到 {len(image_paths)} 张图像")

        # 限制每类样本数
        if len(image_paths) > n_samples:
            image_paths = random.sample(image_paths, n_samples)
            self.logger.info(f"随机选择 {n_samples} 张图像进行增广")

        # 增广图像
        total_generated = 0
        progress_bar = tqdm(image_paths, desc="增广图像")

        for img_path in progress_bar:
            try:
                # 为每张图像生成多个变体
                n_variations = max(1, n_samples // len(image_paths))
                generated_paths = self.augment_image(img_path, output_dir, n_variations)

                total_generated += len(generated_paths)
                progress_bar.set_postfix({'generated': total_generated})

            except Exception as e:
                self.logger.error(f"增广图像失败 {img_path}: {e}")

        self.logger.info(f"增广完成! 生成 {total_generated} 张图像")
        return {'total_generated': total_generated}

    def create_mixed_dataset(
        self,
        original_dir: Path,
        augmented_dir: Path,
        output_dir: Path,
        ratio: float = 0.5
    ) -> Dict[str, int]:
        """
        创建混合数据集（原始+增广）

        Args:
            original_dir: 原始图像目录
            augmented_dir: 增广图像目录
            output_dir: 输出目录
            ratio: 增广图像比例

        Returns:
            统计信息
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 复制原始图像
        original_count = 0
        for img_path in original_dir.glob("*.png"):
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), cv2.imread(str(img_path)))
            original_count += 1

        # 复制增广图像
        augmented_count = 0
        for img_path in augmented_dir.glob("*.png"):
            output_path = output_dir / f"aug_{img_path.name}"
            cv2.imwrite(str(output_path), cv2.imread(str(img_path)))
            augmented_count += 1

        total_count = original_count + augmented_count
        actual_ratio = augmented_count / total_count if total_count > 0 else 0

        self.logger.info(f"创建混合数据集:")
        self.logger.info(f"  原始图像: {original_count}")
        self.logger.info(f"  增广图像: {augmented_count}")
        self.logger.info(f"  总计: {total_count}")
        self.logger.info(f"  增广比例: {actual_ratio:.2f} (目标: {ratio:.2f})")

        return {
            'original_count': original_count,
            'augmented_count': augmented_count,
            'total_count': total_count,
            'augmented_ratio': actual_ratio
        }

    def evaluate_augmentation_quality(
        self,
        original_dir: Path,
        augmented_dir: Path
    ) -> Dict[str, float]:
        """
        评估增广质量

        Args:
            original_dir: 原始图像目录
            augmented_dir: 增广图像目录

        Returns:
            质量指标
        """
        # 尝试导入评估指标
        try:
            from model.evaluation.evaluator import FIDScore, InceptionScore
            EVALUATOR_AVAILABLE = True
        except ImportError as e:
            self.logger.warning(f"评估模块导入失败: {e}")
            EVALUATOR_AVAILABLE = False

        # 加载图像
        original_images = self._load_images_from_dir(original_dir)
        augmented_images = self._load_images_from_dir(augmented_dir)

        if len(original_images) == 0 or len(augmented_images) == 0:
            self.logger.warning("没有找到足够的图像进行评估")
            return {}

        # 转换为张量
        original_tensor = torch.stack(original_images)
        augmented_tensor = torch.stack(augmented_images)

        # 计算指标
        metrics = {}

        # FID
        if EVALUATOR_AVAILABLE:
            try:
                fid_calculator = FIDScore(self.device)
                fid_score = fid_calculator.compute_score(original_tensor, augmented_tensor)
                metrics['fid'] = fid_score
            except Exception as e:
                self.logger.warning(f"FID计算失败: {e}")
                metrics['fid'] = float('nan')
        else:
            self.logger.warning("FID指标不可用")

        # Inception Score
        if EVALUATOR_AVAILABLE:
            try:
                is_calculator = InceptionScore(self.device)
                is_mean, is_std = is_calculator.compute_score(augmented_tensor)
                metrics['is_mean'] = is_mean
                metrics['is_std'] = is_std
            except Exception as e:
                self.logger.warning(f"Inception Score计算失败: {e}")
                metrics['is_mean'] = float('nan')
                metrics['is_std'] = float('nan')
        else:
            self.logger.warning("Inception Score指标不可用")

        # LPIPS（多样性）
        if LPIPS_AVAILABLE and lpips is not None:
            try:
                from model.utils.weights import find_weight
                vgg16_path = find_weight("vgg16")
                if vgg16_path is not None:
                    try:
                        lpips_calculator = lpips.LPIPS(net='vgg', model_path=vgg16_path).to(self.device)
                    except TypeError:
                        lpips_calculator = lpips.LPIPS(net='vgg').to(self.device)
                else:
                    lpips_calculator = lpips.LPIPS(net='vgg').to(self.device)
                lpips_values = []

                # 随机采样计算多样性
                n_samples = min(100, len(augmented_images))
                indices = random.sample(range(len(augmented_images)), n_samples)

                for i in range(0, n_samples, 2):
                    if i + 1 < n_samples:
                        img1 = augmented_tensor[indices[i]].unsqueeze(0).to(self.device)
                        img2 = augmented_tensor[indices[i + 1]].unsqueeze(0).to(self.device)
                        lpips_val = lpips_calculator(img1, img2)
                        lpips_values.append(lpips_val.item())

                metrics['lpips_diversity'] = np.mean(lpips_values) if lpips_values else 0
            except Exception as e:
                self.logger.warning(f"LPIPS计算失败: {e}")
                metrics['lpips_diversity'] = float('nan')
        else:
            self.logger.warning("LPIPS指标不可用")

        self.logger.info("增广质量评估:")
        if 'fid' in metrics:
            self.logger.info(f"  FID: {metrics['fid']:.4f}")
        if 'is_mean' in metrics:
            self.logger.info(f"  IS: {metrics['is_mean']:.4f} ± {metrics.get('is_std', 0):.4f}")
        if 'lpips_diversity' in metrics:
            self.logger.info(f"  LPIPS多样性: {metrics['lpips_diversity']:.4f}")

        return metrics

    def _load_images_from_dir(self, directory: Path) -> List[torch.Tensor]:
        """从目录加载图像"""
        images = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

        for ext in image_extensions:
            for img_path in directory.glob(f"*{ext}"):
                try:
                    # 读取图像
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.config.data.image_size, self.config.data.image_size))

                    # 转换为张量
                    img_tensor = torch.from_numpy(img).float() / 127.5 - 1.0
                    img_tensor = img_tensor.permute(2, 0, 1)

                    images.append(img_tensor)

                    # 限制数量
                    if len(images) >= 1000:
                        break

                except Exception as e:
                    self.logger.warning(f"加载图像失败 {img_path}: {e}")

            if len(images) >= 1000:
                break

        return images

    def generate_latent_space_exploration(
        self,
        base_image_path: Path,
        output_dir: Path,
        n_steps: int = 10
    ):
        """
        潜在空间探索可视化
        沿缺陷/背景风格方向的插值，展示可控生成能力

        Args:
            base_image_path: 基准图像路径
            output_dir: 输出目录
            n_steps: 步数
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 读取基准图像
        base_image = cv2.imread(str(base_image_path))
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
        base_image = cv2.resize(base_image, (self.config.data.image_size, self.config.data.image_size))

        base_tensor = torch.from_numpy(base_image).float() / 127.5 - 1.0
        base_tensor = base_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 生成基准潜在向量并固定种子以保证可复现
            g = torch.Generator()
            g.manual_seed(42)
            z_base_defect = torch.randn(1, self.config.model.generator.latent_dim, generator=g).to(self.device)
            z_base_background = torch.randn(1, self.config.model.generator.latent_dim, generator=g).to(self.device)

            # 生成方向向量
            g.manual_seed(123)
            z_dir_defect = torch.randn(1, self.config.model.generator.latent_dim, generator=g).to(self.device)
            z_dir_background = torch.randn(1, self.config.model.generator.latent_dim, generator=g).to(self.device)

            # 归一化方向
            z_dir_defect = F.normalize(z_dir_defect, dim=1)
            z_dir_background = F.normalize(z_dir_background, dim=1)

            # 探索潜在空间：沿两个方向独立插值
            for i in range(-n_steps, n_steps + 1):
                alpha = i / n_steps

                # 插值：分别操纵缺陷和背景的潜在向量
                z_defect = z_base_defect + alpha * z_dir_defect
                z_background = z_base_background + alpha * z_dir_background

                # 使用插值后的潜在向量生成缺陷图像
                defect_image = self.model.defect_branch(z_defect)
                # 使用插值后的潜在向量生成背景图像
                background_image = self.model.background_branch(base_tensor, z_background)
                # 融合
                fused_image = self.model.fusion_module(defect_image, background_image)

                # 保存
                pseudo_np = fused_image.squeeze(0).cpu().numpy()
                pseudo_np = (pseudo_np + 1.0) / 2.0 * 255.0
                pseudo_np = pseudo_np.transpose(1, 2, 0).astype(np.uint8)

                output_path = output_dir / f"latent_exploration_{i + n_steps:03d}.png"
                cv2.imwrite(str(output_path), cv2.cvtColor(pseudo_np, cv2.COLOR_RGB2BGR))

        self.logger.info(f"潜在空间探索完成: {output_dir}")