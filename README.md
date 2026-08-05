# 基于 Focus-StyleGAN 的工业缺陷图像增广系统

毕业设计项目：面向 MVTec AD 工业质检场景的双分支 GAN 缺陷图像增广系统。

## 功能概览

- **Focus-StyleGAN 模型**：缺陷聚焦分支 + 背景保持分支 + 注意力门控融合，多尺度 CBAM 判别器，WGAN-GP 训练
- **四类增广管线**：
  1. GAN 伪异常生成（缺陷类型注册表驱动潜在向量调制）
  2. 真实缺陷迁移（`core/real_defect_blender.py`）
  3. 检索式增广（`core/retrieval_augmentor.py`）
  4. 缺陷堆叠增广（`core/retrieval_augmentor.py`）
- **评估体系**：FID / IS / LPIPS / PSNR / SSIM / PPS（自研物理合理性指标）/ Pixel-AUC / PRO-AUC / 消融实验框架
- **Web 演示**：Flask 交互界面 + RESTful API

## 环境安装

推荐使用自带的环境安装工具——可选 GPU（CUDA 12.1 / 11.8）或 CPU 版 PyTorch，
逐项检查已装依赖，只安装缺失或版本不符的：

```bash
python setup_env.py
```

也可以手动安装：

```bash
# GPU（CUDA 12.1，其他版本见 pytorch.org）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 或 CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

预训练权重（VGG16/19、InceptionV3、ResNet）可放入项目根目录 `weights/` 下离线加载，否则首次运行自动下载。

## 目录结构

```
core/                   # 增广管线与公共模块
  integrated_augmentor.py   # 统一增广入口（CLI）
  real_defect_blender.py    # 真实缺陷迁移
  retrieval_augmentor.py    # 检索式 / 堆叠式增广
  defect_registry.py        # 缺陷类型 → pattern → 潜在空间调制注册表
  features.py               # 图像特征提取（公共模块）
  defect_classification.py  # 缺陷类型识别（公共模块）
  blending_utils.py         # 缺陷融合底层操作（公共模块）
  super_resolution.py       # Real-ESRGAN 超分（可选）
  integrated_config.yaml    # 主配置文件
model/
  models/focus_stylegan.py  # 核心模型
  training/trainer.py       # 训练器 + Optuna 超参搜索
  data/loader.py            # MVTec AD 数据集
  augmentation/augmentor.py # GAN 增广器
  evaluation/               # FID/IS/LPIPS/PPS/PRO-AUC/消融
  utils/                    # 配置 / 日志 / 种子 / 权重管理
web/                    # Flask 演示（app.py）与 REST API（web_api.py）
scripts/                # 训练曲线等论文图表生成脚本
tests/                  # 功能测试
```

> 以下为运行时目录，不包含在仓库中（见 .gitignore）：
> `datasets/mvtec_anomaly_detection/`（MVTec AD 数据集）、`checkpoints/`（模型检查点）、`weights/`（预训练权重）

## 快速开始

### 训练

```python
import torch, logging
from model.training.trainer import create_trainer_from_config

logger = logging.getLogger("train")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = create_trainer_from_config("core/integrated_config.yaml", device, logger)
trainer.train()
```

默认 GPU 配置：batch_size=32、混合精度（AMP）、cuDNN benchmark 自动开启；
训练过程中每隔 `training.fid_interval` 个 epoch 自动计算真实 FID 并保存最佳模型。

### 增广（CLI）

```bash
# GAN 伪异常
python core/integrated_augmentor.py --mode single --input path/to/good.png --variations 5

# 真实缺陷迁移（推荐，不依赖 GAN）
python core/integrated_augmentor.py --mode real --input path/to/good.png --category bottle

# 检索式 / 堆叠式
python core/integrated_augmentor.py --mode retrieval --input path/to/good.png --category bottle
python core/integrated_augmentor.py --mode stacking  --input path/to/good.png --category bottle
```

### Web 演示

```bash
pip install -r web/requirements-web.txt   # Web 依赖（Flask 等）

python web/app.py        # 交互界面
python web/web_api.py    # RESTful API
```

### 测试

```bash
python tests/test_basic.py
```

## 配置说明

主要配置见 `core/integrated_config.yaml`：

- `data.dataset_path`：MVTec AD 数据集根目录（默认 `./datasets/mvtec_anomaly_detection`）
- `training.seed`：全局随机种子（默认 42，保证可复现）
- `training.use_amp`：CUDA 下启用混合精度训练
- `training.fid_interval` / `fid_n_samples`：训练中 FID 评估频率与样本数
- `model.*`：生成器/判别器结构参数

## 注意事项

- 2026-08 代码优化后模型结构有变更（缺陷分支输出 256×256、判别器去除 InstanceNorm、
  融合模块静态化），**旧 checkpoint 与新代码不完全兼容**，加载时缺失/多余权重键会有警告提示，
  正式实验请重新训练。
- 异常检测评估（Pixel-AUC / PRO-AUC）需要数据集包含 ground_truth mask。
