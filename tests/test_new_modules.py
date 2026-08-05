#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
新模块测试：PPS、Visualizer、Ablation、PRO-AUC
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import logging


def test_pps():
    """测试物理合理性得分"""
    print("\n[PPS] 测试物理合理性得分...")
    try:
        from model.evaluation.pps import PhysicalPlausibilityScore

        pps_calc = PhysicalPlausibilityScore()

        # 创建模拟数据
        gen_images = torch.randn(8, 3, 256, 256)
        real_images = torch.randn(8, 3, 256, 256)

        results = pps_calc.compute(gen_images, real_images)
        print(f"  PPS: {results['pps']:.4f}")
        print(f"  S_geo: {results['s_geo']:.4f}")
        print(f"  S_illum: {results['s_illum']:.4f}")

        assert 0.0 <= results['pps'] <= 1.0, "PPS应在[0, 1]范围内"
        print("  [OK] PPS计算通过")
        return True
    except Exception as e:
        print(f"  [FAIL] PPS测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualizer():
    """测试可视化模块"""
    print("\n[Visualizer] 测试可视化模块...")
    try:
        from model.evaluation.visualizer import Visualizer

        output_dir = Path("test_outputs/visualizations")
        viz = Visualizer(output_dir)

        # 测试指标柱状图
        metrics = {
            'DCGAN': {'fid': 45.2, 'is': 2.1, 'lpips': 0.28},
            'WGAN-GP': {'fid': 38.6, 'is': 2.5, 'lpips': 0.25},
            'StyleGAN2': {'fid': 26.3, 'is': 2.7, 'lpips': 0.21},
            'Focus-StyleGAN': {'fid': 22.5, 'is': 2.9, 'lpips': 0.18},
        }
        viz.visualize_metrics_bar_chart(metrics)

        # 测试Pixel-AUC提升图
        baseline = {
            'bottle': 0.89, 'cable': 0.78, 'capsule': 0.82,
            'carpet': 0.85, 'hazelnut': 0.86, 'metal_nut': 0.84
        }
        augmented = {
            'bottle': 0.96, 'cable': 0.92, 'capsule': 0.94,
            'carpet': 0.95, 'hazelnut': 0.95, 'metal_nut': 0.93
        }
        viz.visualize_pixel_auc_improvement(baseline, augmented)

        # 测试超参数敏感性
        viz.visualize_hyperparameter_sensitivity(
            [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            [28.5, 25.2, 22.5, 23.8, 27.3, 32.1],
            [1, 5, 10, 20, 50, 100],
            [32.5, 27.8, 24.2, 22.5, 23.6, 26.8]
        )

        # 测试训练曲线
        history = {
            'g_loss': [2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.7, 0.6, 0.55, 0.5],
            'd_loss': [1.0, 0.9, 0.8, 0.75, 0.7, 0.68, 0.65, 0.62, 0.6, 0.58],
        }
        viz.visualize_training_curves(history)

        # 验证文件已创建
        created_files = list(output_dir.glob("*.png"))
        print(f"  创建了 {len(created_files)} 个可视化文件")
        for f in created_files:
            print(f"    - {f.name}")

        import shutil
        shutil.rmtree("test_outputs", ignore_errors=True)
        print("  [OK] 可视化测试通过")
        return True
    except Exception as e:
        print(f"  [FAIL] 可视化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cbam_in_discriminator():
    """测试判别器中的CBAM集成"""
    print("\n[CBAM-Disc] 测试判别器CBAM集成...")
    try:
        from model.models.focus_stylegan import (
            DiscriminatorBlock, MultiScaleDiscriminator, CBAM
        )

        # 测试带CBAM的判别器块
        disc_block = DiscriminatorBlock(
            n_layers=5, channel_multiplier=2,
            scale=0, use_cbam=True
        )

        x = torch.randn(2, 3, 256, 256)
        output, features = disc_block(x)
        print(f"  DiscriminatorBlock 输出形状: {output.shape}")
        print(f"  特征层数: {len(features)}")
        assert output.shape[0] == 2, "批次大小应为2"

        # 验证CBAM模块数量
        cbam_count = sum(1 for m in disc_block.modules() if isinstance(m, CBAM))
        print(f"  CBAM模块数: {cbam_count}")
        assert cbam_count >= 5, f"应有至少5个CBAM模块，实际: {cbam_count}"

        # 测试多尺度判别器
        ms_disc = MultiScaleDiscriminator(
            n_layers=5, channel_multiplier=2,
            use_cbam=True, n_scales=3
        )
        fused_output, all_features = ms_disc(x)
        print(f"  MultiScaleDiscriminator 融合输出形状: {fused_output.shape}")
        print(f"  特征层总数: {len(all_features)}")

        # 测试禁用CBAM
        ms_disc_no_cbam = MultiScaleDiscriminator(
            n_layers=5, channel_multiplier=1,
            use_cbam=False, n_scales=2
        )
        fused_no_cbam, _ = ms_disc_no_cbam(x)
        print(f"  禁用CBAM后输出形状: {fused_no_cbam.shape}")
        print("  [OK] CBAM判别器测试通过")
        return True
    except Exception as e:
        print(f"  [FAIL] CBAM判别器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ablation_configs():
    """测试消融实验配置"""
    print("\n[Ablation] 测试消融实验配置...")
    try:
        from model.evaluation.ablation import ABLATION_CONFIGS, AblationConfig

        assert len(ABLATION_CONFIGS) >= 8, f"至少8个消融配置，实际: {len(ABLATION_CONFIGS)}"
        print(f"  消融配置数: {len(ABLATION_CONFIGS)}")

        for name, cfg in ABLATION_CONFIGS.items():
            print(f"    - {name}: {cfg.description}")
            assert isinstance(cfg, AblationConfig), f"{name} 应为AblationConfig类型"

        # 验证关键配置
        assert not ABLATION_CONFIGS['no_perceptual'].use_perceptual_loss
        assert not ABLATION_CONFIGS['no_reconstruction'].use_reconstruction_loss
        assert not ABLATION_CONFIGS['no_cbam'].use_cbam
        assert not ABLATION_CONFIGS['single_branch'].use_dual_branch
        assert ABLATION_CONFIGS['single_scale_disc'].n_disc_scales == 1
        assert ABLATION_CONFIGS['full_model'].use_perceptual_loss
        assert ABLATION_CONFIGS['full_model'].use_dual_branch

        print("  [OK] 消融配置测试通过")
        return True
    except Exception as e:
        print(f"  [FAIL] 消融配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pro_auc():
    """测试PRO-AUC计算"""
    print("\n[PRO-AUC] 测试PRO-AUC计算...")
    try:
        from model.evaluation.anomaly_detection_evaluator import AnomalyDetectionEvaluator

        logger = logging.getLogger("test_pro_auc")
        logger.setLevel(logging.WARNING)

        # 创建模拟评估器（需要真实模型，这里测试计算方法的存在性）
        # 直接测试_compute_pro_score方法
        evaluator = AnomalyDetectionEvaluator.__new__(AnomalyDetectionEvaluator)

        # 模拟数据
        h, w = 64, 64
        anomaly_maps = []
        masks = []

        for _ in range(10):
            amap = np.random.rand(h, w).astype(np.float32)
            # 创建模拟的缺陷区域
            mask = np.zeros((h, w), dtype=np.float32)
            cx, cy = np.random.randint(10, h-10), np.random.randint(10, w-10)
            mask[cy-5:cy+5, cx-5:cx+5] = 1.0
            anomaly_maps.append(amap)
            masks.append(mask)

        pro_score = evaluator._compute_pro_score(anomaly_maps, masks)
        print(f"  PRO-AUC: {pro_score:.4f}")
        assert 0.0 <= pro_score <= 1.0, f"PRO-AUC应在[0, 1]范围内，实际: {pro_score}"
        print("  [OK] PRO-AUC计算通过")
        return True
    except Exception as e:
        print(f"  [FAIL] PRO-AUC测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("新模块测试")
    print("=" * 60)

    tests = [
        ("PPS计算", test_pps),
        ("可视化模块", test_visualizer),
        ("CBAM判别器集成", test_cbam_in_discriminator),
        ("消融实验配置", test_ablation_configs),
        ("PRO-AUC计算", test_pro_auc),
    ]

    results = []
    for name, func in tests:
        try:
            success = func()
            results.append((name, success))
        except Exception as e:
            print(f"  [FATAL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("测试总结:")
    for name, success in results:
        print(f"  {'[OK]' if success else '[FAIL]'} {name}")

    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\n通过率: {passed}/{total}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
