#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
随机种子工具 — 统一设置全局种子保证实验可复现
"""

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    设置全局随机种子

    Args:
        seed: 随机种子
        deterministic: 是否启用 CUDA 确定性算法（会降低性能，默认关闭）
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
