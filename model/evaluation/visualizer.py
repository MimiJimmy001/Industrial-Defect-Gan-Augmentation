"""
训练与评估结果可视化模块

提供论文/实验常用图表：指标对比柱状图、Pixel-AUC 提升图、
超参数敏感性曲线、训练损失曲线等。
所有图表保存为 PNG，使用 Agg 后端，可在无显示环境（服务器/Notebook）运行。
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")  # 无显示环境下必须
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """评估结果可视化器"""

    def __init__(self, output_dir: Union[str, Path] = "visualizations", dpi: int = 150):
        """
        Args:
            output_dir: 图表输出目录
            dpi: 输出分辨率
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def visualize_metrics_bar_chart(
        self, metrics: Dict[str, Dict[str, float]], name: str = "metrics_bar_chart"
    ) -> Path:
        """
        多方法多指标分组柱状图

        Args:
            metrics: {方法名: {指标名: 数值}}
            name: 输出文件名（不含扩展名）
        """
        methods = list(metrics.keys())
        metric_names = list(next(iter(metrics.values())).keys())

        x = np.arange(len(methods))
        n_metrics = len(metric_names)
        width = 0.8 / max(n_metrics, 1)

        fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.5), 4.5))
        for i, mname in enumerate(metric_names):
            values = [metrics[m].get(mname, 0.0) for m in methods]
            ax.bar(x + i * width - 0.4 + width / 2, values, width, label=mname)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylabel("Value")
        ax.set_title("Metrics Comparison")
        ax.legend()
        fig.tight_layout()
        return self._save(fig, name)

    def visualize_pixel_auc_improvement(
        self,
        baseline: Dict[str, float],
        augmented: Dict[str, float],
        name: str = "pixel_auc_improvement",
    ) -> Path:
        """
        增广前后 Pixel-AUC 对比柱状图

        Args:
            baseline: {类别: 增广前 AUC}
            augmented: {类别: 增广后 AUC}
        """
        categories = list(baseline.keys())
        x = np.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(7, len(categories) * 1.2), 4.5))
        ax.bar(x - width / 2, [baseline[c] for c in categories], width, label="Baseline")
        ax.bar(x + width / 2, [augmented.get(c, 0.0) for c in categories], width, label="Augmented")

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=15, ha="right")
        ax.set_ylabel("Pixel-AUC")
        ax.set_ylim(0, 1.05)
        ax.set_title("Pixel-AUC Improvement by Category")
        ax.legend()
        fig.tight_layout()
        return self._save(fig, name)

    def visualize_hyperparameter_sensitivity(
        self,
        param1_values: Sequence[float],
        param1_scores: Sequence[float],
        param2_values: Sequence[float],
        param2_scores: Sequence[float],
        param1_name: str = "lambda_perceptual",
        param2_name: str = "n_critic",
        score_name: str = "FID",
        name: str = "hyperparameter_sensitivity",
    ) -> Path:
        """
        双超参数敏感性曲线（左右两个子图，横轴对数刻度）
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(param1_values, param1_scores, "o-")
        ax1.set_xscale("log")
        ax1.set_xlabel(param1_name)
        ax1.set_ylabel(score_name)
        ax1.set_title(f"Sensitivity: {param1_name}")

        ax2.plot(param2_values, param2_scores, "s-", color="tab:orange")
        ax2.set_xscale("log")
        ax2.set_xlabel(param2_name)
        ax2.set_ylabel(score_name)
        ax2.set_title(f"Sensitivity: {param2_name}")

        fig.tight_layout()
        return self._save(fig, name)

    def visualize_training_curves(
        self,
        history: Dict[str, Sequence[float]],
        name: str = "training_curves",
        keys: Optional[List[str]] = None,
    ) -> Path:
        """
        训练损失/指标曲线

        Args:
            history: {曲线名: 逐 epoch 数值序列}
            keys: 要绘制的键，None 时全部绘制
        """
        keys = keys or list(history.keys())
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key in keys:
            values = history.get(key, [])
            if len(values) > 0:
                ax.plot(range(1, len(values) + 1), values, label=key)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Curves")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save(fig, name)

    def visualize_sample_grid(
        self, images: np.ndarray, name: str = "sample_grid", nrow: int = 4
    ) -> Path:
        """
        图像网格（NCHW 或 NHWC，取值 [0,1] 或 [-1,1]）
        """
        images = np.asarray(images)
        if images.ndim == 4 and images.shape[1] in (1, 3):
            images = images.transpose(0, 2, 3, 1)

        # 归一化到 [0,1]
        vmin, vmax = images.min(), images.max()
        if vmin < 0:
            images = (images + 1) / 2
        images = np.clip(images, 0, 1)

        n = len(images)
        ncol = min(nrow, n)
        nrow_actual = int(np.ceil(n / ncol))

        fig, axes = plt.subplots(nrow_actual, ncol, figsize=(ncol * 2, nrow_actual * 2))
        axes = np.atleast_2d(axes)
        for i in range(nrow_actual * ncol):
            ax = axes[i // ncol, i % ncol]
            ax.axis("off")
            if i < n:
                img = images[i]
                if img.shape[-1] == 1:
                    img = img[..., 0]
                    ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(img)

        fig.tight_layout()
        return self._save(fig, name)


def create_default_visualizations(
    history: Dict[str, Sequence[float]],
    output_dir: Union[str, Path] = "visualizations",
) -> List[Path]:
    """
    从训练历史生成默认图表集（训练曲线 + FID 曲线，如有）

    Args:
        history: trainer.train_history
        output_dir: 输出目录

    Returns:
        生成的文件路径列表
    """
    viz = Visualizer(output_dir)
    paths = []

    loss_keys = [k for k in ("g_loss", "d_loss", "perceptual_loss", "reconstruction_loss") if k in history]
    if loss_keys:
        paths.append(viz.visualize_training_curves(history, keys=loss_keys))

    if "fid" in history and len(history["fid"]) > 0:
        paths.append(viz.visualize_training_curves(
            {"fid": history["fid"]}, name="fid_curve"
        ))

    return paths
