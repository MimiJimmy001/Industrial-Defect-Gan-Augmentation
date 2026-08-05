import sys, json
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- 1. 运行消融实验获取样本 ----
from model.evaluation.ablation import AblationRunner, AblationConfig, ABLATION_CONFIGS
from model.models.focus_stylegan import FocusStyleGAN
from model.data.dataset import MVTecDataset
from model.utils.config import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_config("work/configs/default.yaml")
dataset = MVTecDataset("datasets/mvtec_anomaly_detection", category="transistor")
output_dir = Path("outputs/loss_progression")

# 三个配置
configs = {
      "仅WGAN-GP": ABLATION_CONFIGS["adversarial_only"],
      "+感知损失": AblationConfig(
          name="adv_perceptual",
          description="对抗+感知损失",
          use_perceptual_loss=True,
          use_reconstruction_loss=False,
          use_lpips_loss=False,
      ),
      "完整模型": ABLATION_CONFIGS["full_model"],
  }

  # 每个配置跑5个epoch，保存4张样本
sample_images = {}
for label, cfg in configs.items():
    runner = AblationRunner(FocusStyleGAN, config, dataset, device,
                            logger=None, output_dir=output_dir / cfg.name)
    result = runner.run_single_ablation(cfg, n_epochs=5, save_samples=True)
    # 样本保存在 output_dir/cfg.name/samples/
    sample_path = output_dir / cfg.name / "samples" / "generated_0000.pt"
    if sample_path.exists():
        sample_images[label] = torch.load(sample_path)

  # ---- 2. 绘制四列对比图 ----
input_img = ...  # 从dataset取一张正常样本

fig = plt.figure(figsize=(16, 6))
gs = GridSpec(1, 4, figure=fig)

labels = ["输入图像", "仅WGAN-GP", "+感知损失", "完整模型"]
images = [input_img] + [sample_images[k][0] for k in configs.keys()]
subtitles = [
      "正常样本（参考）",
      "对抗损失：分布对齐\nFID偏高，纹理粗糙",
      "+感知损失：语义一致\n纹理改善，细节不足",
      "+重构+LPIPS：像素精确\n纹理细腻，视觉自然",
  ]

for i, (img, sub) in enumerate(zip(images, subtitles)):
      ax = fig.add_subplot(gs[0, i])
      img_np = img.cpu().squeeze().permute(1, 2, 0).numpy()
      img_np = (img_np + 1) / 2  # [-1,1] → [0,1]
      ax.imshow(np.clip(img_np, 0, 1))
      ax.set_title(sub, fontsize=10)
      ax.axis("off")

plt.suptitle("损失函数递进效果：从分布对齐到像素级精确", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("loss_progression_comparison.png", dpi=200, bbox_inches="tight")
print("已保存: loss_progression_comparison.png")