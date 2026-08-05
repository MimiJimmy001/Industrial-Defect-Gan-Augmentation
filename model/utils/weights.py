#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预训练权重文件管理工具
按优先级查找权重: 项目本地 weights/ → PyTorch 默认缓存 → 在线下载
"""

import os
from pathlib import Path
from typing import Optional

import torch

# 项目内权重目录
PROJECT_WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "weights"

# 模型权重文件名映射
WEIGHT_FILES = {
    "vgg16": "vgg16-397923af.pth",
    "vgg19": "vgg19-dcbb9e9d.pth",
    "inception_v3": "inception_v3_google-1a9a5a14.pth",
    "resnet18": "resnet18-f37072fd.pth",
    "resnet50": "resnet50-0676ba61.pth",
    "wide_resnet50": "wide_resnet50_2-95faca4d.pth",
}


def get_torch_cache_dir() -> Path:
    """获取 PyTorch 默认缓存目录"""
    hub_dir = os.environ.get("TORCH_HOME", "")
    if not hub_dir:
        hub_dir = torch.hub.get_dir() if hasattr(torch.hub, "get_dir") else ""
    if not hub_dir:
        hub_dir = os.path.expanduser("~/.cache/torch/hub")
    return Path(hub_dir) / "checkpoints"


def find_weight(model_name: str) -> Optional[str]:
    """
    查找预训练权重文件

    优先级:
    1. 项目本地 weights/ 目录
    2. PyTorch 默认缓存目录
    3. 返回 None（触发在线下载）

    Args:
        model_name: 模型名称 (vgg16, vgg19, inception_v3, resnet18, resnet50 等)

    Returns:
        权重文件路径，如果找不到则返回 None
    """
    if model_name not in WEIGHT_FILES:
        return None

    filename = WEIGHT_FILES[model_name]

    # 1. 项目本地
    local_path = PROJECT_WEIGHTS_DIR / filename
    if local_path.exists() and local_path.stat().st_size > 100 * 1024:
        return str(local_path)

    # 2. PyTorch 缓存
    cache_path = get_torch_cache_dir() / filename
    if cache_path.exists() and cache_path.stat().st_size > 100 * 1024:
        return str(cache_path)

    # 3. 未找到
    return None


def load_state_dict_from_local(model_name: str, map_location: str = "cpu") -> dict:
    """
    从本地加载权重 state_dict

    Args:
        model_name: 模型名称
        map_location: 设备

    Returns:
        state_dict

    Raises:
        FileNotFoundError: 本地未找到权重文件
    """
    path = find_weight(model_name)
    if path is None:
        raise FileNotFoundError(
            f"未找到 {model_name} 的权重文件。请将 {WEIGHT_FILES[model_name]} 放入 "
            f" {PROJECT_WEIGHTS_DIR}/ 目录，或允许 PyTorch 在线下载。"
        )
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def ensure_weights_dir():
    """确保项目本地权重目录存在"""
    PROJECT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def check_weights_status() -> dict:
    """
    检查所有权重文件的状态

    Returns:
        {模型名: 状态字符串}
    """
    ensure_weights_dir()
    status = {}
    for name, filename in WEIGHT_FILES.items():
        local = PROJECT_WEIGHTS_DIR / filename
        cache = get_torch_cache_dir() / filename
        if local.exists() and local.stat().st_size > 100 * 1024:
            status[name] = f"本地 ({local})"
        elif cache.exists() and cache.stat().st_size > 100 * 1024:
            status[name] = f"缓存 ({cache})"
        else:
            status[name] = "未找到（将在线下载）"
    return status
