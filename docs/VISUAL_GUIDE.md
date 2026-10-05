# 项目可视化与可解释性指南

本页集中展示 Focus-StyleGAN 的模型结构、增广决策、训练循环、评估逻辑和失败诊断。所有图均为 Mermaid 文本，不使用位图架构图。

## 图 1：项目能力思维导图

```mermaid
flowchart TD
    ROOT((Focus-StyleGAN))
    ROOT --> GEN[双分支生成器]
    ROOT --> TRAIN[训练体系]
    ROOT --> AUG[四类增广]
    ROOT --> EVAL[评估体系]
    ROOT --> DEMO[演示与工程化]

    GEN --> G1[缺陷聚焦分支]
    GEN --> G2[背景保持分支]
    GEN --> G3[AdaIN 风格控制]
    GEN --> G4[CBAM 注意力]
    GEN --> G5[注意力门控融合]

    TRAIN --> T1[WGAN-GP]
    TRAIN --> T2[多尺度判别器]
    TRAIN --> T3[AMP 混合精度]
    TRAIN --> T4[在线 FID]
    TRAIN --> T5[Optuna 接口]

    AUG --> A1[GAN 伪异常]
    AUG --> A2[真实缺陷迁移]
    AUG --> A3[检索式增广]
    AUG --> A4[缺陷堆叠]

    EVAL --> E1[FID / IS]
    EVAL --> E2[LPIPS / PSNR / SSIM]
    EVAL --> E3[PPS]
    EVAL --> E4[Pixel / Region / PRO-AUC]

    DEMO --> W1[CLI]
    DEMO --> W2[Flask Web]
    DEMO --> W3[REST API]
    DEMO --> W4[Syntax CI]
```

## 图 2：双分支生成与判别流

```mermaid
flowchart LR
    X[正常图像 x] --> BG[背景保持分支]
    ZB[z_background] --> BG
    ZD[z_defect] --> DF[缺陷聚焦分支]
    BG --> FU[注意力门控融合]
    DF --> FU
    FU --> FAKE[伪异常图像]
    X --> D[多尺度判别器 + CBAM]
    FAKE --> D
    D --> OUT[D_real / D_fake]
    OUT --> LOSS[WGAN-GP + 辅助损失]
```

图 2 的关键解释：缺陷分支和背景分支没有共享最终输出，而是先各自形成表示，再由融合模块学习局部异常与全局结构之间的权重。

## 图 3：增广策略决策树

```mermaid
flowchart TD
    START[需要增广] --> Q1{是否有同类别真实缺陷图}
    Q1 -->|有| Q2{缺陷纹理是否需要保留}
    Q1 -->|没有| Q3{是否有可用兼容 checkpoint}
    Q2 -->|是| REAL[真实缺陷迁移]
    Q2 -->|多样性不足| RETRIEVAL[检索式增广]
    Q3 -->|有| GAN[GAN 伪异常]
    Q3 -->|没有| FALLBACK[传统增广或重新训练]
    RETRIEVAL --> STACK{是否模拟复合缺陷}
    STACK -->|是| STACKING[缺陷堆叠]
    STACK -->|否| REAL
```

这张决策树说明四类增广不是竞争关系，而是依赖条件和业务目标不同的可切换方案。

## 图 4：训练循环与损失流

```mermaid
flowchart TD
    B[正常图 batch] --> Z[采样 z_defect / z_background]
    Z --> F[双分支生成与融合]
    F --> DR[判别真实图]
    F --> DF[判别生成图]
    DR --> DL[WGAN 判别器损失]
    DF --> DL
    DL --> GP[FP32 梯度惩罚]
    GP --> DSTEP[更新判别器]
    F --> STEP{batch_idx % n_critic == 0}
    STEP -->|是| GL[对抗 + L1 重建 + VGG19 感知 + LPIPS]
    GL --> GSTEP[更新生成器]
    STEP -->|否| NEXT[进入下一 batch]
    DSTEP --> NEXT
```

重要解释：

- 判别器每个 batch 更新。
- 生成器由 `batch_idx % n_critic == 0` 触发，默认在索引 0、5、10……更新。
- 梯度惩罚使用 FP32，对抗主体可以使用 AMP。

## 图 5：指标解释与问题归因

```mermaid
flowchart LR
    GEN[生成结果] --> Q[生成分布]
    GEN --> F[图像保真]
    GEN --> P[物理合理性]
    GEN --> AD[异常检测增益]

    Q --> FID[FID / IS]
    F --> LPIPS[LPIPS / PSNR / SSIM]
    P --> PPS[PPS 几何 + 光照]
    AD --> AUC[Pixel / Region / PRO-AUC]

    FID --> BAD1[分布接近但不代表缺陷位置合理]
    PPS --> BAD2[自定义指标，尚未完成专家验证]
    AUC --> BAD3[低 Pixel-AUC 不代表所有区域都失败]
    LPIPS --> BAD4[高 PSNR 可能来自复制背景]
```

## 视觉阅读顺序

1. 图 1 看项目全貌。
2. 图 2 看双分支生成器和判别器如何连接。
3. 图 3 看不同数据条件下选择哪种增广。
4. 图 4 看 WGAN-GP 和辅助损失的实际训练顺序。
5. 图 5 看不同指标各能解释什么、不能解释什么。