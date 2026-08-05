#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Focus-StyleGAN模型
包含缺陷聚焦分支和背景保持分支
"""

import math
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    """自适应实例归一化"""

    def __init__(self, channels: int, style_dim: int, eps: float = 1e-8):
        """
        Args:
            channels: 输入通道数
            style_dim: 风格向量维度
            eps: 数值稳定性系数
        """
        super().__init__()
        self.channels = channels
        self.eps = eps

        # 风格向量的线性变换
        self.style_scale = nn.Linear(style_dim, channels)
        self.style_bias = nn.Linear(style_dim, channels)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [B, C, H, W]
            style: 风格向量 [B, style_dim]

        Returns:
            归一化后的张量
        """
        batch_size, channels, height, width = x.shape

        # 计算实例统计量
        instance_mean = x.mean(dim=(2, 3), keepdim=True)
        instance_std = x.std(dim=(2, 3), keepdim=True) + self.eps

        # 计算风格统计量
        style_scale = self.style_scale(style).view(batch_size, channels, 1, 1)
        style_bias = self.style_bias(style).view(batch_size, channels, 1, 1)

        # 应用AdaIN
        normalized = (x - instance_mean) / instance_std
        styled = normalized * style_scale + style_bias

        return styled


class CBAM(nn.Module):
    """卷积块注意力模块"""

    def __init__(self, channels: int, reduction_ratio: int = 16):
        """
        Args:
            channels: 输入通道数
            reduction_ratio: 通道注意力中的降维比例
        """
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [B, C, H, W]

        Returns:
            注意力加权后的张量
        """
        # 通道注意力
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights

        # 空间注意力
        spatial_avg = torch.mean(x_channel, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_concat = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_weights = self.spatial_attention(spatial_concat)

        # 应用空间注意力
        x_weighted = x_channel * spatial_weights

        return x_weighted


class StyleMappingNetwork(nn.Module):
    """风格映射网络"""

    def __init__(self, latent_dim: int = 512, style_dim: int = 512, n_mlp: int = 8):
        """
        Args:
            latent_dim: 潜在向量维度
            style_dim: 风格向量维度
            n_mlp: MLP层数
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim

        layers = []
        # 输入层
        layers.append(nn.Linear(latent_dim, style_dim))
        layers.append(nn.LeakyReLU(0.2))

        # 中间层
        for _ in range(n_mlp - 2):
            layers.append(nn.Linear(style_dim, style_dim))
            layers.append(nn.LeakyReLU(0.2))

        # 输出层
        layers.append(nn.Linear(style_dim, style_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 潜在向量 [B, latent_dim]

        Returns:
            风格向量 [B, style_dim]
        """
        return self.mlp(z)


class FusionModule(nn.Module):
    """融合模块：结合缺陷聚焦分支和背景保持分支的输出"""

    def __init__(self, channels: int, defect_channels: int = None, background_channels: int = None):
        """
        Args:
            channels: 输出通道数
            defect_channels: 缺陷特征通道数，如果为None则假定与channels相同
            background_channels: 背景特征通道数，如果为None则假定与channels相同
        """
        super().__init__()
        self.channels = channels
        self.defect_channels = defect_channels if defect_channels is not None else channels
        self.background_channels = background_channels if background_channels is not None else channels

        # 静态创建投影层（不能在forward中动态add_module：
        # 动态创建的参数不在优化器内，永远不会被训练）
        if self.defect_channels != channels:
            self.defect_proj = nn.Conv2d(self.defect_channels, channels, 1)
        else:
            self.defect_proj = nn.Identity()
        if self.background_channels != channels:
            self.background_proj = nn.Conv2d(self.background_channels, channels, 1)
        else:
            self.background_proj = nn.Identity()

        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Tanh()
        )

        # 注意力门控
        self.attention_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, defect_feat: torch.Tensor, background_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            defect_feat: 缺陷特征 [B, C_defect, H, W]
            background_feat: 背景特征 [B, C_background, H, W]

        Returns:
            融合后的特征 [B, C, H, W]
        """
        defect_feat = self.defect_proj(defect_feat)
        background_feat = self.background_proj(background_feat)

        # 调整空间尺寸以匹配（将defect_feat上采样到background_feat的尺寸）
        if defect_feat.shape[2:] != background_feat.shape[2:]:
            defect_feat = F.interpolate(defect_feat, size=background_feat.shape[2:], mode='bilinear', align_corners=True)

        # 计算注意力权重
        concat = torch.cat([defect_feat, background_feat], dim=1)
        attention = self.attention_gate(concat)

        # 加权融合
        weighted_defect = defect_feat * attention
        weighted_background = background_feat * (1 - attention)

        # 融合
        fused = weighted_defect + weighted_background

        # 进一步融合
        fused = self.fusion_conv(fused)

        return fused


class GeneratorBlock(nn.Module):
    """生成器块"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        use_adain: bool = True,
        use_cbam: bool = True,
        reduction_ratio: int = 16
    ):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            style_dim: 风格向量维度
            use_adain: 是否使用AdaIN
            use_cbam: 是否使用CBAM
            reduction_ratio: CBAM降维比例
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_adain = use_adain
        self.use_cbam = use_cbam

        # 上采样卷积
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # 注意：不再使用独立的 InstanceNorm —— AdaIN 本身包含实例归一化，
        # 重复归一化会削弱风格调制信号

        # 激活函数
        self.act = nn.LeakyReLU(0.2)

        # AdaIN层
        if use_adain:
            self.adain1 = AdaIN(out_channels, style_dim)
            self.adain2 = AdaIN(out_channels, style_dim)

        # CBAM层
        if use_cbam:
            self.cbam = CBAM(out_channels, reduction_ratio)

        # 残差连接
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, style: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C_in, H, W]
            style: 风格向量 [B, style_dim]，如果为None则不使用AdaIN

        Returns:
            输出特征 [B, C_out, H, W]
        """
        # 残差连接
        residual = self.residual(self.upsample(x))

        # 第一层
        x = self.upsample(x)
        x = self.conv1(x)
        if self.use_adain and style is not None:
            x = self.adain1(x, style)
        x = self.act(x)

        # 第二层
        x = self.conv2(x)
        if self.use_adain and style is not None:
            x = self.adain2(x, style)
        x = self.act(x)

        # CBAM注意力
        if self.use_cbam:
            x = self.cbam(x)

        # 残差连接
        x = x + residual

        return x


class DefectFocusedBranch(nn.Module):
    """缺陷聚焦分支"""

    def __init__(
        self,
        latent_dim: int = 512,
        style_dim: int = 512,
        n_mlp: int = 8,
        channel_multiplier: int = 2
    ):
        """
        Args:
            latent_dim: 潜在向量维度
            style_dim: 风格向量维度
            n_mlp: MLP层数
            channel_multiplier: 通道乘数
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim

        # 风格映射网络
        self.style_mapping = StyleMappingNetwork(latent_dim, style_dim, n_mlp)

        # 初始常数输入
        self.const_input = nn.Parameter(torch.randn(1, 512, 4, 4))

        # 生成器块（6个块 ×2上采样：4x4 -> 256x256，与背景分支输出分辨率一致）
        channels = [512, 512, 256, 128, 64, 32, 16]
        self.blocks = nn.ModuleList()

        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            block = GeneratorBlock(
                in_ch, out_ch, style_dim,
                use_adain=True, use_cbam=True
            )
            self.blocks.append(block)

        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels[-1], 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 潜在向量 [B, latent_dim]

        Returns:
            生成的缺陷图像 [B, 3, H, W]
        """
        batch_size = z.shape[0]

        # 风格映射
        style = self.style_mapping(z)

        # 初始特征
        x = self.const_input.repeat(batch_size, 1, 1, 1)

        # 通过生成器块
        for block in self.blocks:
            x = block(x, style)

        # 输出层
        x = self.output_conv(x)

        return x


class BackgroundPreservingBranch(nn.Module):
    """背景保持分支"""

    def __init__(
        self,
        latent_dim: int = 512,
        style_dim: int = 512,
        n_mlp: int = 8,
        channel_multiplier: int = 2
    ):
        """
        Args:
            latent_dim: 潜在向量维度
            style_dim: 风格向量维度
            n_mlp: MLP层数
            channel_multiplier: 通道乘数
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim

        # 风格映射网络
        self.style_mapping = StyleMappingNetwork(latent_dim, style_dim, n_mlp)

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # 解码器块
        channels = [256, 128, 64]
        self.decoder_blocks = nn.ModuleList()

        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            block = GeneratorBlock(
                in_ch, out_ch, style_dim,
                use_adain=True, use_cbam=True
            )
            self.decoder_blocks.append(block)

        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels[-1], 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            z: 潜在向量 [B, latent_dim]

        Returns:
            生成的背景图像 [B, 3, H, W]
        """
        # 风格映射
        style = self.style_mapping(z)

        # 编码
        encoded = self.encoder(x)

        # 解码
        for block in self.decoder_blocks:
            encoded = block(encoded, style)

        # 输出层
        output = self.output_conv(encoded)

        return output


class MultiScaleDiscriminator(nn.Module):
    """多尺度判别器，集成CBAM注意力"""

    def __init__(
        self,
        n_layers: int = 5,
        channel_multiplier: int = 2,
        use_cbam: bool = True,
        n_scales: int = 3
    ):
        """
        Args:
            n_layers: 每尺度判别器的层数
            channel_multiplier: 通道乘数
            use_cbam: 是否使用CBAM注意力
            n_scales: 尺度数量
        """
        super().__init__()
        self.n_layers = n_layers
        self.n_scales = n_scales

        # 创建多个尺度的判别器，每个带有CBAM注意力
        self.discriminators = nn.ModuleList()
        for scale_idx in range(n_scales):
            disc = DiscriminatorBlock(
                n_layers=n_layers,
                channel_multiplier=channel_multiplier,
                scale=scale_idx,
                use_cbam=use_cbam
            )
            self.discriminators.append(disc)

        # 特征融合：将n_scales个判别器输出拼接后融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(n_scales, 64, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            (融合后的判别器输出, 特征列表)
        """
        features = []
        outputs = []

        for scale_idx, disc in enumerate(self.discriminators):
            # 计算缩放因子：scale=0→1.0, scale=1→0.5, scale=2→0.25
            scale_factor = 0.5 ** scale_idx
            if scale_factor != 1.0:
                scaled_x = F.interpolate(
                    x, scale_factor=scale_factor,
                    mode='bilinear', align_corners=True
                )
            else:
                scaled_x = x

            # 通过判别器（包含CBAM注意力处理）
            output, feat = disc(scaled_x)
            outputs.append(output)
            features.extend(feat)

        # 融合多个尺度的输出
        if len(outputs) > 1:
            # 调整所有输出到相同尺寸（以最大尺寸为基准）
            target_size = outputs[0].shape[2:]
            aligned_outputs = []
            for out in outputs:
                if out.shape[2:] != target_size:
                    out = F.interpolate(
                        out, size=target_size,
                        mode='bilinear', align_corners=True
                    )
                aligned_outputs.append(out)

            # 拼接并融合
            concatenated = torch.cat(aligned_outputs, dim=1)
            fused_output = self.feature_fusion(concatenated)
        else:
            fused_output = outputs[0]

        return fused_output, features


class DiscriminatorBlock(nn.Module):
    """判别器块，集成CBAM注意力机制"""

    def __init__(
        self,
        n_layers: int = 5,
        channel_multiplier: int = 2,
        scale: int = 0,
        use_cbam: bool = True,
        cbam_reduction: int = 16
    ):
        """
        Args:
            n_layers: 层数
            channel_multiplier: 通道乘数
            scale: 尺度级别 (0=原分辨率, 1=半分辨率, 2=四分之一分辨率)
            use_cbam: 是否使用CBAM注意力
            cbam_reduction: CBAM降维比例
        """
        super().__init__()
        self.n_layers = n_layers
        self.scale = scale
        self.use_cbam = use_cbam

        # 通道数序列
        channels = [3, 64, 128, 256, 512, 512]
        # 第一个元素保持为3（输入通道数），其余乘以channel_multiplier
        channels = [channels[0]] + [c * channel_multiplier for c in channels[1:]]

        # 创建层
        # 注意：WGAN-GP 的 critic 不能使用 BatchNorm/InstanceNorm
        # （会破坏梯度惩罚的 Lipschitz 约束假设，见 Gulrajani et al. 2017），
        # 因此判别器不使用任何归一化层
        layers = []
        cbam_modules = []
        for i in range(n_layers):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            stride = 2 if i < 3 else 1  # 前3层下采样
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1),
                    nn.LeakyReLU(0.2)
                )
            )
            # 为每层添加CBAM模块（论文3.3.2：在子判别器特征层中集成CBAM）
            if use_cbam:
                cbam_modules.append(CBAM(out_ch, cbam_reduction))
            else:
                cbam_modules.append(nn.Identity())

        self.layers = nn.ModuleList(layers)
        self.cbam_modules = nn.ModuleList(cbam_modules)

        # 输出层
        self.output_conv = nn.Conv2d(channels[-1], 1, 4, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            (判别器输出, 特征列表)
        """
        features = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 应用CBAM注意力（论文3.3.2节）
            x = self.cbam_modules[i](x)
            features.append(x)

        output = self.output_conv(x)

        return output, features


class FocusStyleGAN(nn.Module):
    """Focus-StyleGAN主模型"""

    def __init__(self, config: dict):
        """
        Args:
            config: 模型配置字典
        """
        super().__init__()
        self.config = config

        # 缺陷聚焦分支
        self.defect_branch = DefectFocusedBranch(
            latent_dim=config.get('generator.latent_dim', 512),
            style_dim=config.get('generator.style_dim', 512),
            n_mlp=config.get('generator.n_mlp', 8),
            channel_multiplier=config.get('generator.channel_multiplier', 2)
        )

        # 背景保持分支
        self.background_branch = BackgroundPreservingBranch(
            latent_dim=config.get('generator.latent_dim', 512),
            style_dim=config.get('generator.style_dim', 512),
            n_mlp=config.get('generator.n_mlp', 8),
            channel_multiplier=config.get('generator.channel_multiplier', 2)
        )

        # 融合模块
        self.fusion_module = FusionModule(channels=3, defect_channels=None, background_channels=None)

        # 多尺度判别器（集成CBAM注意力）
        self.discriminator = MultiScaleDiscriminator(
            n_layers=config.get('discriminator.n_layers', 5),
            channel_multiplier=config.get('discriminator.channel_multiplier', 2),
            use_cbam=config.get('attention.use_cbam', True),
            n_scales=config.get('discriminator.n_scales', 3)
        )

    def forward(
        self,
        real_images: torch.Tensor,
        z_defect: torch.Tensor,
        z_background: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            real_images: 真实图像 [B, 3, H, W]
            z_defect: 缺陷分支潜在向量 [B, latent_dim]
            z_background: 背景分支潜在向量 [B, latent_dim]

        Returns:
            包含生成结果的字典
        """
        # 生成缺陷图像
        defect_images = self.defect_branch(z_defect)

        # 生成背景图像
        background_images = self.background_branch(real_images, z_background)

        # 融合
        fused_images = self.fusion_module(defect_images, background_images)

        # 判别器输出
        d_real, d_real_features = self.discriminator(real_images)
        d_fake, d_fake_features = self.discriminator(fused_images.detach())

        return {
            'defect_images': defect_images,
            'background_images': background_images,
            'fused_images': fused_images,
            'd_real': d_real,
            'd_fake': d_fake,
            'd_real_features': d_real_features,
            'd_fake_features': d_fake_features
        }

    def generate(
        self,
        z_defect: torch.Tensor,
        z_background: torch.Tensor,
        background_images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        生成图像（仅生成器）

        Args:
            z_defect: 缺陷分支潜在向量
            z_background: 背景分支潜在向量
            background_images: 可选的真实背景图像 [B, 3, H, W]，
                传入后背景分支以真实图像为条件（与实际增广管线一致）；
                为None时退化为随机背景（无条件生成）

        Returns:
            生成的图像
        """
        # 生成缺陷图像
        defect_images = self.defect_branch(z_defect)

        # 背景输入：优先使用真实背景图像
        if background_images is None:
            batch_size = z_defect.shape[0]
            image_size = self.config.get('data.image_size', 256) if hasattr(self.config, 'get') else 256
            background_images = torch.randn(batch_size, 3, image_size, image_size).to(z_defect.device)

        # 生成背景图像
        background_images = self.background_branch(background_images, z_background)

        # 融合
        fused_images = self.fusion_module(defect_images, background_images)

        return fused_images

    def discriminate(self, images: torch.Tensor) -> torch.Tensor:
        """
        判别图像（仅判别器）

        Args:
            images: 输入图像

        Returns:
            判别器输出
        """
        output, _ = self.discriminator(images)
        return output