#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验框架
自动化执行论文所述的各项消融实验：
- 损失函数消融 (表4-7)
- 分支结构对比 (表4-8)
- 注意力机制消融 (表4-4)
- 多尺度判别器消融
- 超参数敏感性分析 (表4-5)
"""

import logging
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


@dataclass
class AblationConfig:
    """消融实验配置"""
    name: str
    description: str
    # 模型修改
    use_perceptual_loss: bool = True
    use_reconstruction_loss: bool = True
    use_lpips_loss: bool = True
    use_cbam: bool = True
    use_adain: bool = True
    use_dual_branch: bool = True
    use_multiscale_disc: bool = True
    n_disc_scales: int = 3
    # 超参数（默认值与 core/integrated_config.yaml 保持一致）
    lambda_perceptual: float = 0.1
    lambda_reconstruction: float = 10
    lambda_lpips: float = 1.0
    lambda_gp: float = 10
    g_lr: float = 1e-4
    d_lr: float = 4e-4


# 预定义的消融实验配置
ABLATION_CONFIGS = {
    # === 损失函数消融 (表4-7) ===
    "full_model": AblationConfig(
        name="full_model", description="完整模型（所有损失项启用）",
        use_perceptual_loss=True, use_reconstruction_loss=True, use_lpips_loss=True
    ),
    "no_perceptual": AblationConfig(
        name="no_perceptual", description="移除感知损失",
        use_perceptual_loss=False, use_reconstruction_loss=True, use_lpips_loss=True
    ),
    "no_reconstruction": AblationConfig(
        name="no_reconstruction", description="移除重构损失",
        use_perceptual_loss=True, use_reconstruction_loss=False, use_lpips_loss=True
    ),
    "no_lpips": AblationConfig(
        name="no_lpips", description="移除LPIPS损失",
        use_perceptual_loss=True, use_reconstruction_loss=True, use_lpips_loss=False
    ),
    "adversarial_only": AblationConfig(
        name="adversarial_only", description="仅使用对抗损失",
        use_perceptual_loss=False, use_reconstruction_loss=False, use_lpips_loss=False
    ),

    # === 注意力机制消融 (表4-4) ===
    "no_cbam": AblationConfig(
        name="no_cbam", description="移除CBAM注意力",
        use_cbam=False
    ),
    "no_adain": AblationConfig(
        name="no_adain", description="移除AdaIN风格控制",
        use_adain=False
    ),

    # === 分支结构消融 (表4-8) ===
    "single_branch": AblationConfig(
        name="single_branch", description="单分支生成器",
        use_dual_branch=False
    ),

    # === 判别器消融 ===
    "single_scale_disc": AblationConfig(
        name="single_scale_disc", description="单尺度判别器",
        use_multiscale_disc=False, n_disc_scales=1
    ),
    "dual_scale_disc": AblationConfig(
        name="dual_scale_disc", description="双尺度判别器",
        n_disc_scales=2
    ),
}


class AblationRunner:
    """消融实验运行器"""

    def __init__(
        self,
        base_model,
        base_config: 'Config',
        dataset,
        device: torch.device,
        logger: logging.Logger,
        output_dir: Path
    ):
        """
        Args:
            base_model: 基础FocusStyleGAN模型
            base_config: 基础配置
            dataset: 数据集
            device: 设备
            logger: 日志记录器
            output_dir: 结果输出目录
        """
        self.base_model = base_model
        self.base_config = base_config
        self.dataset = dataset
        self.device = device
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, Dict[str, float]] = {}

    def run_all_ablations(
        self,
        n_epochs: int = 30,
        save_samples: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        运行所有消融实验

        Args:
            n_epochs: 每个消融实验的训练轮数
            save_samples: 是否保存生成样本

        Returns:
            所有消融实验的结果
        """
        self.logger.info("=" * 60)
        self.logger.info("开始消融实验")
        self.logger.info(f"共 {len(ABLATION_CONFIGS)} 组实验")
        self.logger.info("=" * 60)

        for config_name, ablation_config in ABLATION_CONFIGS.items():
            self.logger.info(f"\n{'='*40}")
            self.logger.info(f"实验: {config_name} - {ablation_config.description}")
            self.logger.info(f"{'='*40}")

            try:
                result = self.run_single_ablation(ablation_config, n_epochs, save_samples)
                self.results[config_name] = result
                self.logger.info(f"完成: {config_name} -> FID={result.get('fid', 'N/A')}")
            except Exception as e:
                self.logger.error(f"实验失败 {config_name}: {e}")
                import traceback
                traceback.print_exc()
                self.results[config_name] = {'error': str(e)}

        # 保存结果
        self._save_results()
        return self.results

    def run_single_ablation(
        self,
        ablation_config: AblationConfig,
        n_epochs: int,
        save_samples: bool
    ) -> Dict[str, float]:
        """
        运行单个消融实验

        Args:
            ablation_config: 消融配置
            n_epochs: 训练轮数
            save_samples: 是否保存样本

        Returns:
            实验结果指标
        """
        # 创建修改后的模型配置
        model_config = self._modify_model_config(ablation_config)

        # 构建修改后的模型
        model = self._build_ablated_model(model_config, ablation_config)

        # 训练模型（简化版，用于快速评估）
        metrics = self._train_ablated_model(
            model, ablation_config, n_epochs
        )

        # 生成样本并计算FID/IS/LPIPS
        eval_metrics = self._evaluate_ablated_model(
            model, ablation_config, save_samples
        )
        metrics.update(eval_metrics)

        return metrics

    def _modify_model_config(self, ablation_config: AblationConfig) -> dict:
        """根据消融配置修改模型配置"""
        config = copy.deepcopy(self.base_config._config)

        # 修改注意力配置
        if 'attention' in config.get('model', {}):
            config['model']['attention']['use_cbam'] = ablation_config.use_cbam

        # 修改AdaIN配置
        if 'adain' in config.get('model', {}):
            config['model']['adain']['use'] = ablation_config.use_adain

        # 修改判别器配置
        if 'discriminator' in config.get('model', {}):
            config['model']['discriminator']['n_scales'] = ablation_config.n_disc_scales
            config['model']['discriminator']['use_cbam'] = ablation_config.use_cbam

        # 修改训练超参数
        if 'training' in config:
            config['training']['lambda_perceptual'] = ablation_config.lambda_perceptual
            config['training']['lambda_reconstruction'] = ablation_config.lambda_reconstruction
            config['training']['lambda_gp'] = ablation_config.lambda_gp

        return config

    def _build_ablated_model(self, model_config: dict, ablation_config: AblationConfig):
        """
        构建消融后的模型

        对于单分支消融，创建一个简化的单分支生成器
        """
        from model.models.focus_stylegan import FocusStyleGAN, MultiScaleDiscriminator

        # 创建配置包装器
        class ConfigWrapper:
            def __init__(self, config_dict):
                self._config = config_dict

            def get(self, key, default=None):
                keys = key.split('.')
                value = self._config
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return default
                return value

        wrapped_config = ConfigWrapper(model_config)

        if ablation_config.use_dual_branch:
            # 使用标准双分支模型
            model = FocusStyleGAN(config=wrapped_config).to(self.device)
        else:
            # 构建单分支模型（仅使用缺陷聚焦分支 + 简单解码器）
            model = self._build_single_branch_model(wrapped_config)

        # 如果不使用多尺度判别器
        if not ablation_config.use_multiscale_disc:
            model.discriminator = MultiScaleDiscriminator(
                n_layers=model_config.get('model', {}).get('discriminator', {}).get('n_layers', 5),
                channel_multiplier=model_config.get('model', {}).get('discriminator', {}).get('channel_multiplier', 2),
                use_cbam=ablation_config.use_cbam,
                n_scales=1
            ).to(self.device)

        return model

    def _build_single_branch_model(self, config):
        """
        构建单分支生成器（用于消融实验）
        单分支: 直接从噪声生成完整图像，不分离缺陷和背景
        """
        from model.models.focus_stylegan import FocusStyleGAN

        # 创建标准模型，但修改forward以只用缺陷分支
        model = FocusStyleGAN(config=config).to(self.device)

        # 保存原始的forward
        original_forward = model.forward

        def single_branch_forward(real_images, z_defect, z_background):
            """单分支：只用缺陷分支生成完整图像"""
            # 仅使用缺陷分支生成（缺陷分支从噪声生成完整图像）
            fused = model.defect_branch(z_defect)

            # 判别器输出
            d_real, d_real_features = model.discriminator(real_images)
            d_fake, d_fake_features = model.discriminator(fused.detach())

            return {
                'defect_images': fused,
                'background_images': fused,  # 同一个
                'fused_images': fused,
                'd_real': d_real,
                'd_fake': d_fake,
                'd_real_features': d_real_features,
                'd_fake_features': d_fake_features
            }

        model.forward = single_branch_forward
        return model

    def _train_ablated_model(
        self,
        model,
        ablation_config: AblationConfig,
        n_epochs: int
    ) -> Dict[str, float]:
        """训练消融模型并返回训练指标"""
        from model.training.trainer import PerceptualLoss
        import lpips as lpips_lib
        from torch.utils.data import DataLoader

        # 创建数据加载器
        loader = DataLoader(
            self.dataset,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )

        # 优化器
        g_params = list(model.defect_branch.parameters())
        if ablation_config.use_dual_branch:
            g_params += list(model.background_branch.parameters())
            g_params += list(model.fusion_module.parameters())

        g_optimizer = torch.optim.Adam(
            g_params,
            lr=ablation_config.g_lr,
            betas=(0.5, 0.999)
        )
        d_optimizer = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=ablation_config.d_lr,
            betas=(0.5, 0.999)
        )

        # 损失函数（根据消融配置选择性启用）
        if ablation_config.use_perceptual_loss:
            perceptual_loss_fn = PerceptualLoss().to(self.device)
        if ablation_config.use_lpips_loss:
            lpips_loss_fn = lpips_lib.LPIPS(net='vgg').to(self.device)

        reconstruction_loss_fn = nn.L1Loss()

        # 训练循环
        model.train()
        for epoch in range(n_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0

            progress_bar = tqdm(loader, desc=f"Ablation Epoch {epoch+1}/{n_epochs}")
            for batch_idx, batch in enumerate(progress_bar):
                real_images = batch['image'].to(self.device)
                batch_size = real_images.size(0)

                z_defect = torch.randn(batch_size, 512, device=self.device)
                z_background = torch.randn(batch_size, 512, device=self.device)

                # 训练判别器
                d_optimizer.zero_grad()
                outputs = model(real_images, z_defect, z_background)
                fake_images = outputs['fused_images']

                d_real_loss = -outputs['d_real'].mean()
                d_fake_loss = outputs['d_fake'].mean()
                d_loss = d_real_loss + d_fake_loss

                # 梯度惩罚
                epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device)
                interpolates = epsilon * real_images + (1 - epsilon) * fake_images
                interpolates.requires_grad_(True)
                d_interpolates, _ = model.discriminator(interpolates)
                gradients = torch.autograd.grad(
                    outputs=d_interpolates, inputs=interpolates,
                    grad_outputs=torch.ones_like(d_interpolates),
                    create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                gp = ablation_config.lambda_gp * ((gradients.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
                d_loss = d_loss + gp
                d_loss.backward()
                d_optimizer.step()

                # 训练生成器
                if batch_idx % 5 == 0:
                    g_optimizer.zero_grad()
                    outputs = model(real_images, z_defect, z_background)
                    fake_images = outputs['fused_images']

                    d_fake_for_g, _ = model.discriminator(fake_images)
                    g_loss = -d_fake_for_g.mean()

                    if ablation_config.use_perceptual_loss:
                        g_loss += ablation_config.lambda_perceptual * perceptual_loss_fn(fake_images, real_images)

                    if ablation_config.use_dual_branch and ablation_config.use_reconstruction_loss:
                        defect_imgs = outputs['defect_images']
                        bg_imgs = outputs['background_images']
                        if defect_imgs.shape[-2:] != bg_imgs.shape[-2:]:
                            defect_imgs = F.interpolate(defect_imgs, size=bg_imgs.shape[-2:], mode='bilinear', align_corners=False)
                        g_loss += ablation_config.lambda_reconstruction * reconstruction_loss_fn(defect_imgs, bg_imgs)

                    if ablation_config.use_lpips_loss:
                        g_loss += ablation_config.lambda_lpips * lpips_loss_fn(fake_images, real_images).mean()

                    g_loss.backward()
                    g_optimizer.step()
                    epoch_g_loss += g_loss.item()

                epoch_d_loss += d_loss.item()

            avg_g = epoch_g_loss / max(len(loader) // 5, 1)
            avg_d = epoch_d_loss / len(loader)
            self.logger.info(f"Epoch {epoch+1}: G_loss={avg_g:.4f}, D_loss={avg_d:.4f}")

        return {'train_g_loss': avg_g, 'train_d_loss': avg_d}

    def _evaluate_ablated_model(
        self,
        model,
        ablation_config: AblationConfig,
        save_samples: bool
    ) -> Dict[str, float]:
        """评估消融模型"""
        model.eval()
        metrics = {}

        with torch.no_grad():
            # 生成一批样本
            n_samples = 500
            all_fake = []
            all_real = []

            loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=8, shuffle=True, num_workers=2
            )

            for batch in loader:
                real = batch['image'].to(self.device)
                bs = real.size(0)
                z_d = torch.randn(bs, 512, device=self.device)
                z_b = torch.randn(bs, 512, device=self.device)

                outputs = model(real, z_d, z_b)
                fake = outputs['fused_images']

                all_fake.append(fake.cpu())
                all_real.append(real.cpu())

                if len(all_fake) * bs >= n_samples:
                    break

            all_fake = torch.cat(all_fake, dim=0)[:n_samples]
            all_real = torch.cat(all_real, dim=0)[:n_samples]

            # 计算近似FID (简化版，使用特征均值距离)
            try:
                from model.evaluation.evaluator import FIDScore, InceptionScore
                fid_calc = FIDScore(self.device)
                metrics['fid'] = fid_calc.compute_score(all_real, all_fake)
                is_calc = InceptionScore(self.device)
                is_mean, is_std = is_calc.compute_score(all_fake)
                metrics['is'] = is_mean
            except Exception as e:
                self.logger.warning(f"FID/IS计算失败: {e}")
                metrics['fid'] = float('nan')
                metrics['is'] = float('nan')

            # LPIPS
            try:
                import lpips as lpips_lib
                lpips_calc = lpips_lib.LPIPS(net='vgg').to(self.device)
                lpips_vals = []
                for i in range(0, min(len(all_fake), 100), 16):
                    end = min(i + 16, len(all_fake))
                    lpips_vals.append(lpips_calc(
                        all_fake[i:end].to(self.device),
                        all_real[i:end].to(self.device)
                    ).cpu())
                metrics['lpips'] = float(torch.cat(lpips_vals).mean())
            except Exception as e:
                self.logger.warning(f"LPIPS计算失败: {e}")
                metrics['lpips'] = float('nan')

        return metrics

    def _save_results(self):
        """保存消融实验结果"""
        # 保存为JSON
        results_path = self.output_dir / "ablation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        self.logger.info(f"消融实验结��已保存: {results_path}")

        # 生成对比报告
        report_path = self.output_dir / "ablation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("消融实验报告\n")
            f.write("=" * 60 + "\n\n")

            for config_name, result in self.results.items():
                f.write(f"[{config_name}]\n")
                config = ABLATION_CONFIGS.get(config_name)
                if config:
                    f.write(f"  描述: {config.description}\n")
                for metric, value in result.items():
                    if metric != 'error':
                        f.write(f"  {metric}: {value}\n")
                f.write("\n")

            # 损失函数消融汇总
            f.write("\n--- 损失函数消融汇总 (表4-7) ---\n")
            loss_ablations = ['full_model', 'no_perceptual', 'no_reconstruction', 'no_lpips', 'adversarial_only']
            f.write(f"{'配置':<25} {'FID':<10} {'IS':<10}\n")
            f.write("-" * 45 + "\n")
            for name in loss_ablations:
                if name in self.results:
                    r = self.results[name]
                    f.write(f"{ABLATION_CONFIGS[name].description:<25} {r.get('fid', 'N/A'):<10} {r.get('is', 'N/A'):<10}\n")

            # 注意力机制消融汇总
            f.write("\n--- 注意力机制消融汇总 (表4-4) ---\n")
            attn_ablations = ['full_model', 'no_cbam', 'no_adain']
            f.write(f"{'配置':<25} {'FID':<10} {'相对退化率':<10}\n")
            f.write("-" * 45 + "\n")
            base_fid = self.results.get('full_model', {}).get('fid', 1.0)
            for name in attn_ablations:
                if name in self.results:
                    r = self.results[name]
                    fid_val = r.get('fid', float('nan'))
                    if isinstance(fid_val, (int, float)) and isinstance(base_fid, (int, float)) and base_fid > 0:
                        degradation = (fid_val - base_fid) / base_fid * 100
                        f.write(f"{ABLATION_CONFIGS[name].description:<25} {fid_val:<10.1f} {degradation:<10.1f}%\n")

        self.logger.info(f"消融实验报告已保存: {report_path}")


def run_hyperparameter_sensitivity(
    model_class,
    config,
    dataset,
    device,
    logger,
    output_dir: Path
) -> Dict[str, Any]:
    """
    运行超参数敏感性分析 (表4-5、图4-4)

    Args:
        model_class: 模型类
        config: 基础配置
        dataset: 数据集
        device: 设备
        logger: 日志记录器
        output_dir: 输出目录

    Returns:
        敏感性分析结果
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("开始超参数敏感性分析...")

    # 感知损失权重敏感性
    lambda_perceptual_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    fid_perceptual = []

    for lp in lambda_perceptual_values:
        logger.info(f"测试 λ_perceptual={lp}")
        # 创建临时消融配置
        temp_config = copy.deepcopy(ABLATION_CONFIGS['full_model'])
        temp_config.lambda_perceptual = lp
        runner = AblationRunner(model_class, config, dataset, device, logger, output_dir / f"sensitivity_perceptual_{lp}")
        result = runner.run_single_ablation(temp_config, n_epochs=5, save_samples=False)
        fid_perceptual.append(result.get('fid', float('nan')))
        logger.info(f"  FID={result.get('fid', 'N/A')}")

    # 重构损失权重敏感性
    lambda_recon_values = [1, 5, 10, 20, 50, 100]
    fid_recon = []

    for lr_val in lambda_recon_values:
        logger.info(f"测试 λ_recon={lr_val}")
        temp_config = copy.deepcopy(ABLATION_CONFIGS['full_model'])
        temp_config.lambda_reconstruction = lr_val
        runner = AblationRunner(model_class, config, dataset, device, logger, output_dir / f"sensitivity_recon_{lr_val}")
        result = runner.run_single_ablation(temp_config, n_epochs=5, save_samples=False)
        fid_recon.append(result.get('fid', float('nan')))
        logger.info(f"  FID={result.get('fid', 'N/A')}")

    # 保存结果
    sensitivity_results = {
        'lambda_perceptual_values': lambda_perceptual_values,
        'fid_perceptual': fid_perceptual,
        'lambda_recon_values': lambda_recon_values,
        'fid_recon': fid_recon
    }

    with open(output_dir / "sensitivity_results.json", 'w') as f:
        json.dump(sensitivity_results, f, indent=2, default=str)

    logger.info("超参数敏感性分析完成!")
    return sensitivity_results
