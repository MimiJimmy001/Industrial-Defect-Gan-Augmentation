#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估器模块
计算FID、IS、LPIPS等指标
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from scipy import linalg
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple
# 可选导入
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    lpips = None

from model.models.focus_stylegan import FocusStyleGAN
from model.data.loader import MVTecADDataset
from model.utils.config import Config


def _load_inception_v3(device: torch.device) -> nn.Module:
    """加载 InceptionV3（优先本地权重），保留完整分类头。

    注意：Inception Score 需要 1000 维分类 logits，因此这里**不能**把
    fc 替换为 Identity。FID 的特征提取通过 hook 挂在 avgpool (pool3) 上，
    与 fc 无关。
    """
    from model.utils.weights import load_state_dict_from_local
    try:
        inception = models.inception_v3(pretrained=False, transform_input=False)
        state_dict = load_state_dict_from_local("inception_v3", map_location="cpu")
        inception.load_state_dict(state_dict)
    except (FileNotFoundError, RuntimeError):
        inception = models.inception_v3(pretrained=True, transform_input=False)
    inception = inception.to(device)
    inception.eval()
    for param in inception.parameters():
        param.requires_grad = False
    return inception


def _preprocess_for_inception(images: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> Inception 输入 ([0, 1], 299x299)"""
    images = (images + 1) / 2
    if images.shape[2] != 299 or images.shape[3] != 299:
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    return images.clamp(0, 1)


class InceptionScore:
    """Inception Score计算器（标准实现：对1000类softmax概率计算）"""

    def __init__(self, device: torch.device):
        """
        Args:
            device: 计算设备
        """
        self.device = device
        # 保留完整分类头：IS 的定义基于 p(y|x) 分类概率
        self.inception = _load_inception_v3(device)

    def compute_score(self, images: torch.Tensor, splits: int = 10) -> Tuple[float, float]:
        """
        计算Inception Score

        Args:
            images: 图像张量 [N, 3, H, W]，范围[-1, 1]
            splits: 分割数

        Returns:
            (IS均值, IS标准差)
        """
        n_images = images.shape[0]
        images = _preprocess_for_inception(images)

        # 获取1000类预测概率
        preds = []
        with torch.no_grad():
            for i in range(0, n_images, 32):
                batch = images[i:i + 32].to(self.device)
                pred = self.inception(batch)
                # inception_v3 在 eval 模式下直接返回 logits 张量
                preds.append(F.softmax(pred, dim=1).cpu())

        preds = torch.cat(preds, dim=0)

        # 计算IS: exp(E_x[KL(p(y|x) || p(y))])
        scores = []
        split_size = max(n_images // splits, 1)
        for k in range(splits):
            part = preds[k * split_size: (k + 1) * split_size]
            if part.shape[0] == 0:
                continue
            py = part.mean(0, keepdim=True)
            kl = (part * (part.log() - py.log())).sum(1).mean()
            scores.append(kl.exp())

        scores = torch.stack(scores)
        return scores.mean().item(), scores.std().item()


class FIDScore:
    """FID计算器（标准实现）

    - 特征：InceptionV3 pool3 层（avgpool，2048维）
    - 统计量：完整协方差矩阵（非对角近似）
    - 公式：||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*(Sigma_r @ Sigma_f)^{1/2})
    """

    def __init__(self, device: torch.device):
        """
        Args:
            device: 计算设备
        """
        self.device = device
        self.inception = _load_inception_v3(device)

        # 标准 FID 使用 pool3（avgpool 输出，2048维）
        self._features = []
        self.inception.avgpool.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        """收集 pool3 特征 [B, 2048, 1, 1] -> [B, 2048]"""
        self._features.append(output.view(output.shape[0], -1).cpu())

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        """
        提取 pool3 特征

        Args:
            images: 图像张量 [N, 3, H, W]，范围[-1, 1]
            batch_size: 批次大小

        Returns:
            特征矩阵 [N, 2048]
        """
        images = _preprocess_for_inception(images)
        self._features = []

        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size].to(self.device)
            _ = self.inception(batch)

        features = torch.cat(self._features, dim=0).numpy()
        self._features = []
        return features

    @staticmethod
    def compute_statistics_from_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """从特征矩阵计算均值与完整协方差"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    @staticmethod
    def frechet_distance(
        mu1: np.ndarray, sigma1: np.ndarray,
        mu2: np.ndarray, sigma2: np.ndarray,
        eps: float = 1e-6
    ) -> float:
        """标准 Frechet 距离（与 pytorch-fid 一致的处理方式）"""
        diff = mu1 - mu2

        # 协方差乘积平方根
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # 数值误差可能产生微量复数
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))

    def compute_score(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        计算FID

        Args:
            real_images: 真实图像 [N, 3, H, W]，范围[-1, 1]
            fake_images: 生成图像 [N, 3, H, W]，范围[-1, 1]

        Returns:
            FID值
        """
        real_features = self.extract_features(real_images)
        fake_features = self.extract_features(fake_images)

        mu_real, sigma_real = self.compute_statistics_from_features(real_features)
        mu_fake, sigma_fake = self.compute_statistics_from_features(fake_features)

        return self.frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


class Evaluator:
    """模型评估器"""

    def __init__(
        self,
        model: FocusStyleGAN,
        dataset: MVTecADDataset,
        config: Config,
        device: torch.device,
        logger: logging.Logger
    ):
        """
        初始化评估器

        Args:
            model: 模型
            dataset: 数据集
            config: 配置
            device: 设备
            logger: 日志记录器
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device
        self.logger = logger

        # 创建数据加载器
        self.loader = DataLoader(
            dataset,
            batch_size=config.evaluation.batch_size,
            shuffle=False,
            num_workers=4
        )

        # 评估指标
        self.metrics = {}

        # FID评分（内置标准实现，pool3特征 + 完整协方差，无需 pytorch_fid 依赖）
        self.metrics['fid'] = FIDScore(device)

        # Inception Score
        self.metrics['is'] = InceptionScore(device)

        # LPIPS
        if LPIPS_AVAILABLE and lpips is not None:
            from model.utils.weights import find_weight
            vgg16_path = find_weight("vgg16")
            if vgg16_path is not None:
                try:
                    self.metrics['lpips'] = lpips.LPIPS(net='vgg', model_path=vgg16_path).to(device)
                except TypeError:
                    self.metrics['lpips'] = lpips.LPIPS(net='vgg').to(device)
            else:
                self.metrics['lpips'] = lpips.LPIPS(net='vgg').to(device)
            # 设置为评估模式，不计算梯度
            self.metrics['lpips'].eval()
            for param in self.metrics['lpips'].parameters():
                param.requires_grad = False
        else:
            self.logger.warning("lpips未安装，LPIPS指标不可用")

    def generate_samples(self, n_samples: int, background_images: torch.Tensor = None) -> torch.Tensor:
        """
        生成样本

        Args:
            n_samples: 样本数量
            background_images: 可选的真实背景图像 [N, 3, H, W]，
                传入后背景分支以真实图像为条件（与实际增广管线一致）；
                否则退化为随机背景。

        Returns:
            生成的图像张量
        """
        self.model.eval()
        generated_images = []
        n_generated = 0

        with torch.no_grad():
            # 分批生成
            while n_generated < n_samples:
                batch_size = min(self.config.evaluation.batch_size, n_samples - n_generated)

                # 生成随机潜在向量
                z_defect = torch.randn(batch_size, self.config.model.generator.latent_dim).to(self.device)
                z_background = torch.randn(batch_size, self.config.model.generator.latent_dim).to(self.device)

                # 背景条件（循环使用真实图像）
                bg = None
                if background_images is not None and len(background_images) > 0:
                    idx = torch.arange(n_generated, n_generated + batch_size) % len(background_images)
                    bg = background_images[idx].to(self.device)

                # 生成图像
                images = self.model.generate(z_defect, z_background, background_images=bg)
                generated_images.append(images.cpu())
                n_generated += batch_size

        return torch.cat(generated_images, dim=0)

    def get_real_samples(self, n_samples: int) -> torch.Tensor:
        """
        获取真实样本

        Args:
            n_samples: 样本数量

        Returns:
            真实图像张量
        """
        real_images = []
        count = 0

        for batch in self.loader:
            images = batch['image']
            real_images.append(images)

            count += images.shape[0]
            if count >= n_samples:
                break

        real_images = torch.cat(real_images, dim=0)[:n_samples]
        return real_images

    def evaluate(self) -> Dict[str, float]:
        """
        执行评估

        Returns:
            评估指标字典
        """
        self.logger.info("开始模型评估...")

        n_samples = min(self.config.evaluation.n_samples, len(self.dataset))
        self.logger.info(f"评估样本数: {n_samples}")

        # 先取真实样本（既作为FID真实分布，也作为生成的背景条件）
        self.logger.info("加载真实样本...")
        real_images = self.get_real_samples(n_samples)

        # 生成样本（以真实图像为背景条件，与实际增广管线一致）
        self.logger.info("生成样本...")
        generated_images = self.generate_samples(n_samples, background_images=real_images)

        # 确保样本数量一致
        min_samples = min(generated_images.shape[0], real_images.shape[0])
        generated_images = generated_images[:min_samples]
        real_images = real_images[:min_samples]

        self.logger.info(f"实际评估样本数: {min_samples}")

        results = {}

        # 计算FID
        if 'fid' in self.config.evaluation.metrics and 'fid' in self.metrics:
            self.logger.info("计算FID...")
            fid_value = self.metrics['fid'].compute_score(real_images, generated_images)
            results['fid'] = fid_value
            self.logger.info(f"FID: {fid_value:.4f}")
        elif 'fid' in self.config.evaluation.metrics:
            self.logger.warning("FID指标配置但不可用")

        # 计算Inception Score
        if 'is' in self.config.evaluation.metrics:
            self.logger.info("计算Inception Score...")
            is_mean, is_std = self.metrics['is'].compute_score(generated_images)
            results['is_mean'] = is_mean
            results['is_std'] = is_std
            self.logger.info(f"IS: {is_mean:.4f} ± {is_std:.4f}")

        # 计算LPIPS
        if 'lpips' in self.config.evaluation.metrics and 'lpips' in self.metrics:
            self.logger.info("计算LPIPS...")
            lpips_values = []

            # 分批计算
            batch_size = self.config.evaluation.batch_size
            with torch.no_grad():
                for i in range(0, min_samples, batch_size):
                    real_batch = real_images[i:i + batch_size].to(self.device)
                    fake_batch = generated_images[i:i + batch_size].to(self.device)

                    lpips_batch = self.metrics['lpips'](real_batch, fake_batch)
                    lpips_values.extend(lpips_batch.cpu().detach().numpy())

            lpips_mean = np.mean(lpips_values)
            lpips_std = np.std(lpips_values)
            results['lpips_mean'] = lpips_mean
            results['lpips_std'] = lpips_std
            self.logger.info(f"LPIPS: {lpips_mean:.4f} ± {lpips_std:.4f}")
        elif 'lpips' in self.config.evaluation.metrics:
            self.logger.warning("LPIPS指标配置但不可用（需要安装lpips）")

        # 计算PSNR和SSIM
        if 'psnr' in self.config.evaluation.metrics or 'ssim' in self.config.evaluation.metrics:
            self.logger.info("计算PSNR和SSIM...")
            psnr_values = []
            ssim_values = []

            for i in range(min_samples):
                real_img = real_images[i].unsqueeze(0).to(self.device)
                fake_img = generated_images[i].unsqueeze(0).to(self.device)

                # PSNR（先转到[0,1]范围，峰值为1；mse=0时给上限值）
                real_01 = (real_img + 1) / 2
                fake_01 = (fake_img + 1) / 2
                mse = F.mse_loss(real_01, fake_01)
                if mse.item() < 1e-10:
                    psnr = torch.tensor(99.0, device=self.device)
                else:
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                psnr_values.append(psnr.item())

                # SSIM
                ssim_val = self._compute_ssim(real_img, fake_img)
                ssim_values.append(ssim_val)

            if 'psnr' in self.config.evaluation.metrics:
                results['psnr_mean'] = np.mean(psnr_values)
                results['psnr_std'] = np.std(psnr_values)
                self.logger.info(f"PSNR: {results['psnr_mean']:.4f} ± {results['psnr_std']:.4f}")

            if 'ssim' in self.config.evaluation.metrics:
                results['ssim_mean'] = np.mean(ssim_values)
                results['ssim_std'] = np.std(ssim_values)
                self.logger.info(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")

        # 计算PPS (Physical Plausibility Score)
        if 'pps' in self.config.evaluation.metrics:
            self.logger.info("计算PPS...")
            try:
                from model.evaluation.pps import PhysicalPlausibilityScore
                pps_calculator = PhysicalPlausibilityScore()
                pps_results = pps_calculator.compute(generated_images, real_images)
                results['pps'] = pps_results['pps']
                results['pps_s_geo'] = pps_results['s_geo']
                results['pps_s_illum'] = pps_results['s_illum']
                self.logger.info(f"PPS: {results['pps']:.4f} (S_geo={results['pps_s_geo']:.4f}, S_illum={results['pps_s_illum']:.4f})")
            except Exception as e:
                self.logger.warning(f"PPS计算失败: {e}")
                results['pps'] = float('nan')

        self.logger.info("评估完成!")
        return results

    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
        """
        计算SSIM

        Args:
            img1: 图像1
            img2: 图像2
            window_size: 窗口大小

        Returns:
            SSIM值
        """
        # 从[-1, 1]转换到[0, 1]
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2

        # 创建高斯窗口
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                                  for x in range(window_size)])
            return gauss / gauss.sum()

        def create_window(window_size, channel):
            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            return window

        channel = img1.shape[1]
        window = create_window(window_size, channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()

    def ablation_study(self) -> Dict[str, Dict[str, float]]:
        """
        消融实验

        Returns:
            消融实验结果
        """
        self.logger.info("开始消融实验...")

        ablation_results = {}

        # 基准模型
        self.logger.info("基准模型...")
        baseline_results = self.evaluate()
        ablation_results['baseline'] = baseline_results

        # 无AdaIN
        self.logger.info("无AdaIN...")
        original_adain = self.model.defect_branch.blocks[0].use_adain
        for block in self.model.defect_branch.blocks:
            block.use_adain = False
        for block in self.model.background_branch.decoder_blocks:
            block.use_adain = False

        no_adain_results = self.evaluate()
        ablation_results['no_adain'] = no_adain_results

        # 恢复AdaIN
        for block in self.model.defect_branch.blocks:
            block.use_adain = original_adain
        for block in self.model.background_branch.decoder_blocks:
            block.use_adain = original_adain

        # 无CBAM
        self.logger.info("无CBAM...")
        original_cbam = self.model.defect_branch.blocks[0].use_cbam
        for block in self.model.defect_branch.blocks:
            block.use_cbam = False
        for block in self.model.background_branch.decoder_blocks:
            block.use_cbam = False

        no_cbam_results = self.evaluate()
        ablation_results['no_cbam'] = no_cbam_results

        # 恢复CBAM
        for block in self.model.defect_branch.blocks:
            block.use_cbam = original_cbam
        for block in self.model.background_branch.decoder_blocks:
            block.use_cbam = original_cbam

        # 无感知损失
        self.logger.info("训练无感知损失的模型...")
        # 注意：这需要重新训练模型，这里简化处理
        ablation_results['no_perceptual'] = {
            'note': '需要重新训练模型'
        }

        self.logger.info("消融实验完成!")
        return ablation_results