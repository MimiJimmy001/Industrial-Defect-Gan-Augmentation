# Focus-StyleGAN 评估协议与结果模板

## 1. 目的与边界

这份协议定义未来如何评价 Focus-StyleGAN 和四类增广管线。当前仓库没有运行完整训练，也没有 MVTec AD 原始数据、checkpoint 或可复现实验日志，因此下面所有结果表都保持“待运行”。

本协议只定义实验方法、指标、固定参数和验收条件，不把尚未发生的实验包装成结果。

## 2. 固定实验协议

### 2.1 数据集阶段

| 阶段 | 类别 | 目的 |
|---|---|---|
| 烟雾测试 | `bottle` | 验证训练、生成、评估和文件路径能完整跑通 |
| 三类别试点 | `bottle`、`carpet`、`hazelnut` | 同时覆盖刚体、纹理和表面类别 |
| 扩展验证 | MVTec AD 其余类别 | 检查跨类别泛化和超参稳定性 |

训练只使用每个类别的 `train/good`。`test` 中的异常图和 mask 只用于最终评估；不能用于训练、超参搜索或 checkpoint 选择，否则会产生数据泄漏。

### 2.2 固定参数

| 参数 | 试点值 | 正式值 |
|---|---:|---:|
| 图像尺寸 | 256×256 | 256×256 |
| batch size | 32 | 32；显存不足时记录降级值 |
| epochs | 100 | 100 |
| seed | 42 | 42、43、44 三次重复 |
| `n_critic` | 5 | 5 |
| `lambda_gp` | 10 | 10 |
| `lambda_reconstruction` | 10 | 10 |
| `lambda_perceptual` | 0.1 | 0.1 |
| `lambda_lpips` | 1.0 | 1.0 |
| FID 训练间隔 | 每 10 epochs | 每 10 epochs |
| 训练中 FID 样本 | 1000 | 1000 |
| 正式生成质量评估 | `min(5000, 数据可用量)` | 与试点一致 |
| 异常检测评估 | 每类 50 个正常/异常样本 | 优先提高到 200，并记录实际值 |

试点阶段允许单 seed 运行，但结论只能标记为“pilot”。正式结果至少运行三个 seed，报告均值和标准差。模型选择使用验证集 FID；如果没有验证集，则使用训练过程中留出的正常图像，不允许查看 test 异常结果后再挑选 checkpoint。

### 2.3 评估配对

- FID：真实图像集合与生成图像集合比较。
- LPIPS/PSNR/SSIM：相同背景条件下的真实图像和生成图像一一比较，不能随机跨图配对。
- PPS：生成图像与真实异常分布比较；如果提供 mask，几何一致性计算会使用 mask。
- Pixel-AUC/Region-AUC/PRO-AUC：原始正常数据训练一个检测器，增量正常数据训练另一个检测器，两者在完全相同的异常测试集上评估。

## 3. 指标定义与解释

| 指标 | 适合回答的问题 | 方向 | 当前实现 |
|---|---|---:|---|
| FID | 生成分布和真实分布有多接近 | 越低越好 | InceptionV3 pool3 特征 + 完整协方差 |
| IS | 生成样本是否清晰且类别多样 | 越高越好 | InceptionV3 分类分布 |
| LPIPS | 两张图在感知特征空间有多远 | 越低越好 | VGG 特征距离 |
| PSNR | 生成图和参考图的像素误差 | 越高越好 | 在 `[0,1]` 范围计算 |
| SSIM | 结构、亮度和对比度是否接近 | 越高越好 | 局部窗口结构相似度 |
| PPS | 缺陷几何与光照是否合理 | 越高越好 | 自定义 `0.5*S_geo + 0.5*S_illum` |
| Pixel-AUC | 像素级异常排序能力 | 越高越好 | ResNet 特征 + 异常图 + `roc_auc_score` |
| Region-AUC | 区域级异常排序能力 | 越高越好 | 每张图的区域异常分数 |
| PRO-AUC | 每个真实异常区域被覆盖的能力 | 越高越好 | Per-Region Overlap 曲线 |

### 3.1 FID

```text
FID = ||mu_real - mu_fake||²
      + Tr(Sigma_real + Sigma_fake
           - 2 * (Sigma_real @ Sigma_fake)^(1/2))
```

FID 依赖特征提取器、图像预处理和样本量。换模型、换尺寸或换样本量以后不能直接比较，必须重新建立基线。

### 3.2 PSNR

评估器先把图像从 `[-1,1]` 转到 `[0,1]`，最大像素值为 1：

```text
PSNR = 20 * log10(1 / sqrt(MSE))
```

当 MSE 接近 0 时，代码返回上限 99.0，避免除零。

### 3.3 PPS

```text
PPS = 0.5 * S_geo + 0.5 * S_illum
```

- `S_geo`：用 Canny 轮廓、曲率、长宽比和边缘锐度等特征，比较生成缺陷和真实缺陷的统计分布。
- `S_illum`：用光照与阴影一致性衡量缺陷是否像真实材料上的异常。

PPS 是项目自定义指标，不是论文通用指标。正式报告中必须同时给出子分数，并说明它尚未经过人工专家标注验证。

### 3.4 异常检测指标

- `Pixel-AUC`：把所有像素异常分数和 mask 拉平后计算 ROC-AUC。
- `Region-AUC`：先聚合每张图的异常区域分数，再计算 ROC-AUC。
- `PRO-AUC`：对每个真实异常区域计算检测覆盖率，最后对 FPR 区间积分。

MVTec AD 中异常区域通常很小，Pixel-AUC 可能受到大量正常背景像素影响，因此必须同时报告 Region-AUC 和 PRO-AUC。

## 4. 实验矩阵

### 4.1 增广管线对比

| 编号 | 配置 | 依赖 | 状态 | FID | PPS | Pixel-AUC | Region-AUC | PRO-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A0 | 原始正常数据，无增广 | 无 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| A1 | 传统几何/颜色增广 | 无 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| A2 | GAN 伪异常 | checkpoint | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| A3 | 真实缺陷迁移 | 真实异常图 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| A4 | 检索式增广 | 缺陷库 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| A5 | 缺陷堆叠 | 缺陷库 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

所有配置必须使用同一原始训练集、同一异常测试集和同一检测器。至少比较 `bottle`、`carpet`、`hazelnut` 三个类别。

### 4.2 消融实验

| 编号 | 被移除或替换的模块 | 目的 | 状态 |
|---|---|---|---|
| B0 | 完整模型 | 主结果 | 待运行 |
| B1 | 无 CBAM | 判断注意力对缺陷定位的贡献 | 待运行 |
| B2 | 无 AdaIN | 判断风格控制对多样性的贡献 | 待运行 |
| B3 | 单分支生成器 | 判断双分支解耦是否有效 | 待运行 |
| B4 | 无重建损失 | 判断背景保持约束的作用 | 待运行 |
| B5 | 无感知损失 | 判断高层语义约束的作用 | 待运行 |
| B6 | 无 LPIPS | 判断感知距离权重是否必要 | 待运行 |
| B7 | 单尺度判别器 | 判断多尺度判别的贡献 | 待运行 |

`model/evaluation/ablation.py` 已经定义部分配置，但只有在同一训练轮数、同一数据划分和同一评估器下运行，消融结果才可比较。

## 5. 生成质量结果模板

复制以下表到真实实验报告中，并把“待运行”替换为带单位的结果。每个表头都必须附上实际配置。

| 配置 | 类别 | Seed | Checkpoint | 样本数 | FID ↓ | IS ↑ | LPIPS ↓ | PSNR ↑ | SSIM ↑ | PPS ↑ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 完整模型 | bottle | 42 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| 完整模型 | carpet | 42 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| 完整模型 | hazelnut | 42 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

## 6. 异常检测增益模板

改进百分比按当前代码实现计算：

```text
improvement_percent = (augmented_value - baseline_value) / baseline_value * 100
```

| 方法 | 类别 | Pixel-AUC 基线 | Pixel-AUC 增广 | Pixel 提升 | Region-AUC 提升 | PRO-AUC 提升 | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| A2 GAN | bottle | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| A3 真实迁移 | bottle | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| A4 检索式 | bottle | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| A5 缺陷堆叠 | bottle | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

至少报告三类别的均值、标准差和每个类别的独立结果，不能只选表现最好的类别。

## 7. 结果验收条件

### 运行有效性

- 所有数据、代码版本、配置、GPU、随机种子和 checkpoint 路径可追溯。
- 没有使用 test 异常图训练或选择模型。
- 所有生成图和 anomaly map 没有 NaN、全黑或尺寸不一致。
- 每个配置至少完成原始数据、增广数据和固定检测器的完整闭环。

### 结论有效性

- 单 seed 结果只称为 pilot，不写成稳定提升。
- 正式结论至少包含 3 个 seed 的均值与标准差。
- 至少 2/3 个试点类别出现同方向改进，才可以说“在试点范围内观察到增益”。
- FID、PSNR、SSIM 改善但 Pixel-AUC/PRO-AUC 下降时，不能声称对异常检测有效。
- PPS 改善必须同时展示 `S_geo` 和 `S_illum`，不能只报告综合分。

### 当前不可接受的说法

- “FID 达到某数值”而没有运行记录。
- “提升异常检测 20%”而没有基线、类别和检测器说明。
- 用训练集图像的 FID 当作泛化结果。
- 用不同样本量或不同预处理比较两个方法。
- 用单张展示图推断整体分布质量。

## 8. 运行入口

```bash
# 1. 环境
python setup_env.py

# 2. 训练
python -c "import logging, torch; from model.training.trainer import create_trainer_from_config; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); trainer=create_trainer_from_config('core/integrated_config.yaml', device, logging.getLogger('train')); trainer.train()"

# 3. 四类增广
python core/integrated_augmentor.py --mode single --input path/to/good.png --variations 5
python core/integrated_augmentor.py --mode real --input path/to/good.png --category bottle
python core/integrated_augmentor.py --mode retrieval --input path/to/good.png --category bottle
python core/integrated_augmentor.py --mode stacking --input path/to/good.png --category bottle
```

当前仓库没有统一的 `evaluate.py` CLI。生成质量评估通过 `model/evaluation/evaluator.py` 的 `Evaluator.evaluate()` 完成；异常检测增益通过 `AnomalyDetectionEvaluator.evaluate_augmentation_effect()` 完成。实现实验驱动脚本时，应只增加编排代码，不改变本协议中的指标口径。

## 9. 结果表当前状态

截至本文档编写时，所有 FID、IS、LPIPS、PSNR、SSIM、PPS、Pixel-AUC、Region-AUC、PRO-AUC 和 improvement 数值均为 **待运行**。仓库提供的是评估能力和实验协议，不是已经完成的实验结果。