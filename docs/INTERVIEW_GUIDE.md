# Focus-StyleGAN 面试指南

## 1. 面试前先明确边界

这个项目的强项是：完整的双分支生成架构、四类增广路径、WGAN-GP 训练器和多层评估代码。当前不能声称的强项是：完整训练结果、真实 FID/AUC 指标、跨类别泛化和生产部署。

面试时先讲机制，再讲代码，再讲验证状态。不要用“应该会提升”代替已经验证的事实。

## 2. 三种讲述版本

### 2.1 60 秒版本

“这是一个面向 MVTec AD 的工业缺陷图像增广项目。工业场景中缺陷样本很少，普通几何增广和简单复制粘贴容易破坏背景结构。我把生成过程拆成缺陷聚焦分支和背景保持分支，用 AdaIN 注入风格，用 CBAM 强化局部注意力，再通过门控融合生成伪异常图。训练侧使用多尺度判别器和 WGAN-GP；评估侧覆盖 FID、LPIPS、PSNR/SSIM、自定义 PPS，以及 Pixel-AUC、Region-AUC、PRO-AUC。当前代码已经完成，但数据集、权重和完整实验没有随仓库提交，所以文档里的指标表都标记为待运行。”

### 2.2 三分钟版本

1. **问题**：工业缺陷少、类别分布不均，生成或粘贴缺陷容易造成背景漂移、边界断裂和物理不合理。
2. **方法**：缺陷分支只负责异常表示，背景分支以真实图为条件保留结构，融合模块学习空间门控；AdaIN 控制风格，CBAM 强化局部特征。
3. **训练**：WGAN-GP 处理稳定性，多尺度判别器同时看局部纹理和整体结构；当前配置是 256×256、batch 32、100 epochs、`n_critic=5`、`lambda_gp=10`。
4. **增广**：除 GAN 以外，还实现了真实缺陷迁移、检索和缺陷堆叠，保证没有可控 checkpoint 时仍有替代路径。
5. **评估**：生成质量看 FID/IS，图像保真看 LPIPS/PSNR/SSIM，领域合理性看 PPS，下游效果看 Pixel-AUC/Region-AUC/PRO-AUC。
6. **边界**：当前只完成代码实现和语法级 CI，没有完成真实训练，官方指标表全部待运行。

### 2.3 十分钟版本

在 60 秒版本之上，按以下顺序展开：

- 用 Mermaid 数据流解释两个潜变量和两个分支的输入输出。
- 在白板上写出 `L_D`、`L_GP` 和 `L_G`，说明每个损失项的作用。
- 解释 `FusionModule` 的 attention 公式和为什么不能用固定 mask。
- 说明 `n_critic=5` 的训练节奏和 FP32 梯度惩罚。
- 说明 FID、LPIPS、PSNR、SSIM、PPS 和异常检测 AUC 的分工。
- 说明四类增广在不同数据条件下的适用性。
- 最后主动说明未完成训练、无权重、无真实指标和旧 checkpoint 不兼容等边界。

## 3. 高频问答

### Q1：这个项目解决什么业务问题？

工业质检中缺陷样本稀缺且类型不均衡。项目通过生成和组合缺陷样本，增加异常检测器可见的缺陷形态，同时尽量保持背景结构不漂移。

### Q2：为什么采用双分支，而不是用一个生成器？

单生成器在重建损失压力下容易改变整张图。双分支把缺陷生成和背景保持分开，既能增加异常多样性，又可以用重建和感知约束限制背景变化。

### Q3：为什么不用简单的复制粘贴？

复制粘贴保留了真实缺陷纹理，但容易忽略位置、尺度、边缘、光照和材料约束。真实缺陷迁移管线做了混合处理，但仍需要和 GAN 管线对照评估。复制粘贴是基线，不是默认最优解。

### Q4：AdaIN 在这个项目中起什么作用？

AdaIN 将内容特征的通道统计量替换为风格向量预测的统计量，让缺陷纹理和背景风格可以解耦。它提高风格多样性，但不能单独保证缺陷位置正确。

### Q5：CBAM 为什么同时放在生成器和判别器？

生成器中的 CBAM 帮助聚焦缺陷局部区域；判别器中的 CBAM 帮助区分真实和伪造的局部细节。它同时提供通道和空间注意力，但会增加计算量，因此应在消融中验证贡献。

### Q6：融合模块如何决定缺陷和背景的权重？

融合模块把两个分支特征投影到同一通道，再通过 `sigmoid(Conv(concat))` 预测空间注意力：

```text
x_fused = attention * x_defect + (1 - attention) * x_background
```

权重由网络学习，不是人工 mask。

### Q7：为什么使用 WGAN-GP？

WGAN-GP 使用 Wasserstein 目标和梯度惩罚，通常比普通 GAN 更稳定。对于缺陷局部差异明显、训练样本较少的情况，可以降低判别器梯度饱和和模式崩溃风险。

### Q8：为什么判别器不使用 BatchNorm？

WGAN-GP 的梯度惩罚需要约束每个样本附近的梯度，BatchNorm 会引入样本间依赖。代码中的判别器块避免这类范数层，以减少训练目标与实现之间的偏差。

### Q9：为什么梯度惩罚使用 FP32？

AMP 可以降低显存，但梯度范数对数值精度敏感。当前代码将梯度惩罚放在 FP32 计算，再把总判别器损失缩放回训练流程。

### Q10：为什么每个 batch 都更新判别器？

WGAN-GP 通常需要判别器更充分地估计 Wasserstein 距离。当前代码每个 batch 更新判别器，并在 `batch_idx % n_critic == 0` 时更新生成器，默认 `n_critic=5`。

### Q11：四类增广有什么区别？

- GAN 伪异常：依赖 checkpoint，异常多样性高，但生成质量不可控。
- 真实缺陷迁移：保留真实纹理，更接近数据分布，但位置和边界需要调参。
- 检索式增广：从已有缺陷库选择相似形态，适合缺陷内部差异小的场景。
- 缺陷堆叠：组合多个缺陷，适合模拟复杂表面，但空间关系可能不自然。

### Q12：为什么不能只看 FID？

FID 衡量真实和生成分布的全局距离，但不能保证缺陷位置、mask 或局部结构合理。必须同时看 LPIPS、PSNR、SSIM、PPS 和异常检测 AUC。

### Q13：PSNR 高是否说明增广效果一定好？

不一定。PSNR 依赖像素误差，如果生成图几乎复制了背景，PSNR 可能很高，但缺陷多样性很低。它必须与 FID、PPS 和下游 AUC 一起解释。

### Q14：IS 对工业异常检测有什么局限？

IS 基于 ImageNet 分类模型，主要衡量清晰度和类别多样性。工业缺陷是局部纹理和表面异常，IS 不能直接说明缺陷是否符合材料规律，因此只能作为辅助指标。

### Q15：PPS 为什么需要谨慎使用？

PPS 是项目自定义指标，由 `S_geo` 和 `S_illum` 加权组成。它目前没有经过人工专家标注和跨类别验证，因此只能说明“几何和光照分布是否接近”，不能当作最终业务结论。

### Q16：为什么同时使用 Pixel-AUC、Region-AUC 和 PRO-AUC？

Pixel-AUC 评估像素排序，Region-AUC 评估区域排序，PRO-AUC 评估每个真实异常区域的覆盖能力。工业缺陷区域通常很小，单看 Pixel-AUC 可能被背景像素稀释，三者一起看更全面。

### Q17：如何避免数据泄漏？

训练只用 `train/good`，异常图和 mask 只用于评估；模型选择使用验证集或留出正常图，不能根据 test 异常结果挑 checkpoint。增广数据集必须来自训练划分，不能把 test 图像加入生成样本。

### Q18：显存不足时怎么处理？

优先降低 batch size，其次降低图像尺寸，再考虑减少判别器尺度。每一个改变都要记录为新的配置，不能把降级配置和原配置的结果混在一起比较。

### Q19：训练不稳定时如何排查？

先看生成器和判别器损失曲线，再看梯度惩罚和样本网格。可以降低学习率、检查 AMP/scaler、调整 `lambda_gp`、限制判别器更新次数，并确认没有使用不兼容的范数层。

### Q20：旧 checkpoint 为什么可能加载失败？

模型在 2026-08 后调整了缺陷分支输出尺寸、判别器结构和融合模块。参数 key 或张量形状可能变化，加载器会给出缺失/多余权重警告，正式实验应重新训练。

## 4. 代码阅读路径

| 顺序 | 文件 | 要回答的问题 |
|---:|---|---|
| 1 | `core/integrated_config.yaml` | 图像尺寸、batch、损失权重和训练节奏是什么 |
| 2 | `model/models/focus_stylegan.py` | 双分支、AdaIN、CBAM、融合和多尺度判别器如何实现 |
| 3 | `model/training/trainer.py` | 损失如何计算、AMP 和梯度惩罚在哪里 |
| 4 | `model/evaluation/evaluator.py` | FID、IS、LPIPS、PSNR、SSIM、PPS 怎样计算 |
| 5 | `model/evaluation/anomaly_detection_evaluator.py` | Pixel-AUC、Region-AUC、PRO-AUC 和增广增益怎样评估 |
| 6 | `model/evaluation/ablation.py` | 哪些消融配置已经定义 |
| 7 | `core/integrated_augmentor.py` | 四类增广如何从 CLI 进入核心模块 |
| 8 | `web/app.py`、`web/web_api.py` | 演示链路怎样调用模型和返回结果 |

## 5. 白板讲解框架

1. 画正常图 `x`、`z_defect`、`z_background` 三个输入。
2. 分成缺陷分支和背景分支，标出各自输出。
3. 画注意力门控融合，并写权重公式。
4. 画多尺度判别器，标出 `d_real`、`d_fake`。
5. 写出 `L_D`、`L_GP`、`L_G` 的组成。
6. 最后画评估链路：FID/IS、LPIPS/PSNR/SSIM、PPS、Pixel/Region/PRO-AUC。

## 6. 简历表述

### 中文

- 设计双分支 Focus-StyleGAN，将缺陷区域生成与背景结构保持解耦，并通过 AdaIN、CBAM 和注意力门控融合输出伪异常图像。
- 实现 WGAN-GP 训练器、多尺度判别器、AMP、在线 FID、Optuna 接口和 checkpoint 管理，支持固定 seed 的复现训练。
- 实现 GAN 伪异常、真实缺陷迁移、检索式增广和缺陷堆叠四类管线，为不同缺陷数据条件提供可切换策略。
- 构建 FID、IS、LPIPS、PSNR、SSIM、PPS、Pixel-AUC、Region-AUC、PRO-AUC 和消融实验代码，分别评估生成质量和异常检测增益。

### English

- Designed a dual-branch Focus-StyleGAN that decouples defect synthesis from background preservation using AdaIN, CBAM, and attention-gated fusion.
- Implemented a WGAN-GP trainer with multi-scale discrimination, AMP, online FID, Optuna hooks, and reproducible-seed configuration.
- Built four augmentation pipelines: GAN-based pseudo anomalies, real-defect transfer, retrieval-based augmentation, and defect stacking.
- Implemented an evaluation suite covering FID, IS, LPIPS, PSNR, SSIM, PPS, Pixel-AUC, Region-AUC, PRO-AUC, and ablation configurations.

## 7. 面试中不要说的内容

- 不说“FID/AUC 已经达到某个值”，除非提供实际运行记录。
- 不说“模型已经完成生产部署”，当前只有 Flask 演示。
- 不说“PPS 是行业标准指标”，它仍是项目自定义指标。
- 不说“四类增广都已经被验证优于基线”，当前没有完整对比结果。
- 不把测试集异常图用于训练或模型选择。