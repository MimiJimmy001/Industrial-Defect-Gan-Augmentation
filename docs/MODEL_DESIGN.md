# Focus-StyleGAN 模型设计与训练说明

## 1. 文档边界

本文只描述仓库中已经实现的模型结构、损失函数和训练流程，不声称已经完成训练。配置值以 `core/integrated_config.yaml` 为准，代码入口以 `model/models/focus_stylegan.py` 和 `model/training/trainer.py` 为准。

## 2. 符号与张量约定

| 符号 | 含义 | 当前默认形状 |
|---|---|---|
| `B` | batch size | 32 |
| `H × W` | 图像尺寸 | 256 × 256 |
| `x` | 真实正常图像 | `[B, 3, H, W]` |
| `z_defect` | 缺陷分支潜变量 | `[B, 512]` |
| `z_background` | 背景分支潜变量 | `[B, 512]` |
| `x_defect` | 缺陷分支输出 | `[B, 3, H, W]` |
| `x_background` | 背景分支输出 | `[B, 3, H, W]` |
| `x_fake` | 融合后的伪异常图像 | `[B, 3, H, W]` |
| `D(x)` | 判别器输出 | 多尺度结果聚合后的标量 logits |

图像在训练器中使用 `[-1, 1]` 归一化范围；`evaluator.py` 会在计算 PSNR 时显式转换到 `[0, 1]`。

## 3. 模块职责

### 3.1 StyleMappingNetwork

将高斯潜变量映射为多层风格向量，默认设置：

- `latent_dim=512`
- `style_dim=512`
- `n_mlp=8`

风格向量通过 AdaIN 注入生成器块，使缺陷纹理和背景风格可以分别采样。

### 3.2 AdaIN

AdaIN 从风格向量预测通道均值和方差，并将内容特征的统计量对齐到目标风格。代码中的生成器块在多个尺度使用 AdaIN，不再重复使用独立的 InstanceNorm，以避免归一化层职责重叠。

### 3.3 CBAM

CBAM 包含通道注意力和空间注意力：

- 通道注意力判断哪些通道对缺陷或结构更重要。
- 空间注意力判断哪些位置更值得关注。
- 生成器利用 CBAM 强调缺陷纹理，判别器利用 CBAM 区分真实与生成局部细节。

### 3.4 DefectFocusedBranch

输入 `z_defect`，通过多层生成块输出缺陷图像。分支的目标是产生局部异常纹理，而不是重建整张正常图像。缺陷模式由 `core/defect_registry.py` 提供元数据和调制建议。

### 3.5 BackgroundPreservingBranch

输入真实图像和 `z_background`。分支保留主体结构、边缘和光照条件，但其输出仍是生成器特征，不等同于像素级复制。训练时需要与缺陷分支的输出一起参与重建约束。

### 3.6 FusionModule

融合关系可以写成：

```text
attention = sigmoid(Conv([Project(x_defect), Project(x_background)]))
x_fused = attention * x_defect_projected
          + (1 - attention) * x_background_projected
x_fused = FusionConv(x_fused)
```

`FusionConv` 包含两层卷积、InstanceNorm、LeakyReLU 和 Tanh。注意力权重不是人工设定的掩码，而是由两个分支的连接特征学习得到。

### 3.7 MultiScaleDiscriminator

判别器包含多个尺度的子判别器，每个子判别器可启用 CBAM。前向结果由各尺度输出和特征融合结果共同组成。判别器中的卷积块刻意不使用 BatchNorm 或 InstanceNorm，以满足 WGAN-GP 对单样本梯度的约束。

## 4. 前向数据流

```text
x_defect = DefectBranch(z_defect)
x_background = BackgroundBranch(x, z_background)
x_fake = FusionModule(x_defect, x_background)

d_real = Discriminator(x)
d_fake = Discriminator(stop_gradient(x_fake))
```

`forward()` 同时返回缺陷图像、背景图像、融合图像和各尺度判别结果，便于训练器、评估器和可视化模块复用同一套输出。

## 5. 训练目标

### 5.1 WGAN-GP 判别器损失

采用 WGAN 的对抗目标，并通过梯度惩罚约束判别器：

```text
L_D = -E[D(x_real)] + E[D(x_fake)]
      + λ_gp * L_GP

L_GP = E[(||∇_x D(epsilon * x_real + (1 - epsilon) * x_fake)||₂ - 1)²]
```

代码中 `λ_gp=10`。梯度惩罚在 FP32 上计算，避免 AMP 下的梯度范数误差。

### 5.2 生成器损失

```text
L_G = -E[D(x_fake)]
      + λ_reconstruction * ||x_defect - x_background||₁
      + λ_perceptual * L_perceptual
      + λ_lpips * L_LPIPS
```

当前配置：

- `λ_reconstruction=10`
- `λ_perceptual=0.1`
- `λ_lpips=1.0`

每项的职责不同：

| 损失项 | 目标 | 可能的副作用 |
|---|---|---|
| 对抗损失 | 生成判别器认为真实的样本 | 权重过高会产生不稳定纹理 |
| 重建损失 | 让缺陷分支和背景分支保持可组合性 | 过高会压制缺陷多样性 |
| 感知损失 | 保留高层纹理和结构语义 | 依赖 VGG 特征，计算成本较高 |
| LPIPS | 对齐人类感知的图像距离 | 与 FID 优化方向不一定一致 |

### 5.3 感知损失

`PerceptualLoss` 使用 VGG19 的 `relu1`、`relu3`、`relu5`、`relu9`、`relu13` 层，对输入和目标的特征 MSE 进行累加。输入使用 ImageNet 均值和标准差归一化。

### 5.4 LPIPS

LPIPS 使用 VGG 特征空间计算感知距离。仓库优先查找 `weights/` 下的本地权重，找不到时由 `lpips` 包按自身逻辑加载。

## 6. 训练顺序

1. 每个 batch 先从训练集取出正常图像。
2. 采样 `z_defect` 和 `z_background`。
3. 用 `no_grad` 生成一次用于判别器更新的假样本，降低显存占用。
4. 计算判别器 WGAN 损失，再用 FP32 计算梯度惩罚。
5. 使用 `GradScaler` 更新判别器。
6. 当 `batch_idx % n_critic == 0` 时，重新调用生成分支并更新生成器；默认 `n_critic=5`，即索引 0、5、10……执行生成器更新。
7. 每个 epoch 记录生成器损失、判别器损失、感知损失和重建损失。
8. 每隔 `fid_interval=10` 个 epoch 计算一次 FID，并保存最佳模型。

需要注意，当前更新条件是 `batch_idx % n_critic == 0` 的代码语义；后续如果要严格改成“先更新 5 次判别器再更新 1 次生成器”，应该调整循环结构并重新跑对照实验，不能在结果报告中混用两种训练节奏。

## 7. AMP 与显存

配置默认 `use_amp=true`，但只有 `device.type == 'cuda'` 时生效。AMP 不包含梯度惩罚，梯度惩罚前会执行 `.float()`。

显存不足时按以下顺序调整：

1. 将 `batch_size` 从 32 降到 16 或 8。
2. 将 `image_size` 从 256 降到 128，但必须重新记录评估口径。
3. 将 `n_scales` 从 3 降为 2 或 1，并记录对应消融。
4. 降低 `num_workers` 只影响数据加载进程，不直接降低 GPU 显存。

## 8. 失败模式与控制点

| 失败现象 | 可能原因 | 代码控制点 |
|---|---|---|
| 背景漂移 | 背景分支/重建约束不足 | `lambda_reconstruction`, `lambda_perceptual`, `FusionModule` |
| 缺陷不明显 | 注意力门控偏向背景 | `core/defect_registry.py`, 缺陷强度参数 |
| 模式崩溃 | 判别器过强或噪声使用单一 | `n_critic`, 生成器学习率, `z_defect` |
| 梯度爆炸 | WGAN-GP、学习率或 AMP 数值异常 | `lambda_gp`, lr, `GradScaler`, FP32 GP |
| FID 不稳定 | 样本太少或批次差异 | `fid_n_samples=1000`, 固定 seed、固定评估集 |
| 旧权重报错 | 结构变更导致 key 不匹配 | `model/utils/weights.py` 的加载提示和重新训练 |
| 生成样本单一 | 数据类别不平衡或风格空间收缩 | 缺陷类型注册表、增强策略、采样分布 |

## 9. 复现边界

- 仓库没有 MVTec AD 图片和 mask，训练前需自行准备数据目录。
- 仓库没有已训练 checkpoint，`data`、`checkpoints`、`weights` 和 `outputs` 默认不提交。
- 模型结构在 2026-08 有过变更，旧 checkpoint 不保证兼容。
- 完整训练需要 GPU；CPU 只能用于代码检查和极小规模烟雾测试，不适合作为正式指标来源。
- 训练结果应连同 `core/integrated_config.yaml`、Git commit、GPU 型号、batch size 和样本量一起记录。