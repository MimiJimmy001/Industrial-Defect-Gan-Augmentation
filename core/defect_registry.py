#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缺陷类型注册表 — 从 MVTec AD 数据集提取真实缺陷类型，
并映射到潜在空间调制参数，用于指导 GAN 增广。
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

# MVTec AD 数据集根目录
_DATASET_ROOT = Path(__file__).parent.parent / "datasets" / "mvtec_anomaly_detection"

# 缺陷类型 → 潜在空间调制参数映射
# 每种缺陷类型的调制策略针对其在真实工业场景中的视觉特征设计
DEFECT_MODULATION: Dict[str, dict] = {
    # === 裂纹 / 断裂类 ===
    "broken_large": {
        "pattern": "sharp_local",
        "z_amplify": 1.4, "z_sharpen": 0.7, "z_spread": 0.15,
        "mask_temperature": 0.6, "description": "大面积断裂"
    },
    "broken_small": {
        "pattern": "sharp_local",
        "z_amplify": 1.2, "z_sharpen": 0.6, "z_spread": 0.12,
        "mask_temperature": 0.7, "description": "小面积断裂"
    },
    "broken_teeth": {
        "pattern": "sharp_local",
        "z_amplify": 1.3, "z_sharpen": 0.65, "z_spread": 0.14,
        "mask_temperature": 0.65, "description": "齿部断裂"
    },
    "crack": {
        "pattern": "directional",
        "z_amplify": 1.1, "z_sharpen": 0.5, "z_spread": 0.10,
        "mask_temperature": 0.8, "description": "裂纹"
    },
    "split_teeth": {
        "pattern": "sharp_local",
        "z_amplify": 1.2, "z_sharpen": 0.6, "z_spread": 0.13,
        "mask_temperature": 0.7, "description": "齿部开裂"
    },

    # === 划痕 / 切割类 ===
    "scratch": {
        "pattern": "directional",
        "z_amplify": 1.0, "z_sharpen": 0.4, "z_spread": 0.08,
        "mask_temperature": 0.85, "description": "划痕"
    },
    "scratch_head": {
        "pattern": "directional",
        "z_amplify": 1.0, "z_sharpen": 0.45, "z_spread": 0.07,
        "mask_temperature": 0.85, "description": "头部划痕"
    },
    "scratch_neck": {
        "pattern": "directional",
        "z_amplify": 0.95, "z_sharpen": 0.4, "z_spread": 0.08,
        "mask_temperature": 0.9, "description": "颈部划痕"
    },
    "cut": {
        "pattern": "directional",
        "z_amplify": 1.1, "z_sharpen": 0.5, "z_spread": 0.08,
        "mask_temperature": 0.8, "description": "切割"
    },
    "cut_inner_insulation": {
        "pattern": "directional",
        "z_amplify": 1.0, "z_sharpen": 0.5, "z_spread": 0.07,
        "mask_temperature": 0.85, "description": "内绝缘层切割"
    },
    "cut_outer_insulation": {
        "pattern": "directional",
        "z_amplify": 1.05, "z_sharpen": 0.5, "z_spread": 0.07,
        "mask_temperature": 0.85, "description": "外绝缘层切割"
    },
    "thread": {
        "pattern": "directional",
        "z_amplify": 0.8, "z_sharpen": 0.35, "z_spread": 0.06,
        "mask_temperature": 0.9, "description": "线状缺陷"
    },
    "thread_side": {
        "pattern": "directional",
        "z_amplify": 0.8, "z_sharpen": 0.35, "z_spread": 0.06,
        "mask_temperature": 0.9, "description": "侧面螺纹缺陷"
    },
    "thread_top": {
        "pattern": "directional",
        "z_amplify": 0.8, "z_sharpen": 0.35, "z_spread": 0.06,
        "mask_temperature": 0.9, "description": "顶部螺纹缺陷"
    },
    "gray_stroke": {
        "pattern": "directional",
        "z_amplify": 0.7, "z_sharpen": 0.3, "z_spread": 0.05,
        "mask_temperature": 0.95, "description": "灰色笔触"
    },

    # === 孔洞 / 穿刺类 ===
    "hole": {
        "pattern": "localized",
        "z_amplify": 1.3, "z_sharpen": 0.5, "z_spread": 0.05,
        "mask_temperature": 0.7, "description": "孔洞"
    },
    "poke": {
        "pattern": "localized",
        "z_amplify": 1.0, "z_sharpen": 0.45, "z_spread": 0.05,
        "mask_temperature": 0.8, "description": "穿刺"
    },
    "poke_insulation": {
        "pattern": "localized",
        "z_amplify": 1.0, "z_sharpen": 0.45, "z_spread": 0.04,
        "mask_temperature": 0.8, "description": "绝缘层穿刺"
    },

    # === 污染 / 颜色类 ===
    "contamination": {
        "pattern": "diffuse",
        "z_amplify": 0.6, "z_sharpen": 0.15, "z_spread": 0.35,
        "mask_temperature": 1.2, "description": "污染"
    },
    "color": {
        "pattern": "diffuse",
        "z_amplify": 0.5, "z_sharpen": 0.1, "z_spread": 0.4,
        "mask_temperature": 1.3, "description": "颜色异常"
    },
    "oil": {
        "pattern": "diffuse",
        "z_amplify": 0.4, "z_sharpen": 0.1, "z_spread": 0.45,
        "mask_temperature": 1.4, "description": "油污"
    },
    "glue": {
        "pattern": "diffuse",
        "z_amplify": 0.55, "z_sharpen": 0.15, "z_spread": 0.3,
        "mask_temperature": 1.2, "description": "胶水残留"
    },
    "glue_strip": {
        "pattern": "diffuse",
        "z_amplify": 0.5, "z_sharpen": 0.2, "z_spread": 0.25,
        "mask_temperature": 1.25, "description": "胶条"
    },
    "metal_contamination": {
        "pattern": "diffuse",
        "z_amplify": 0.7, "z_sharpen": 0.2, "z_spread": 0.3,
        "mask_temperature": 1.1, "description": "金属污染"
    },
    "rough": {
        "pattern": "diffuse",
        "z_amplify": 0.5, "z_sharpen": 0.15, "z_spread": 0.35,
        "mask_temperature": 1.3, "description": "粗糙表面"
    },

    # === 变形 / 弯曲类 ===
    "bent": {
        "pattern": "structural",
        "z_amplify": 1.0, "z_sharpen": 0.2, "z_spread": 0.5,
        "mask_temperature": 1.0, "description": "弯曲变形"
    },
    "bent_wire": {
        "pattern": "structural",
        "z_amplify": 1.0, "z_sharpen": 0.2, "z_spread": 0.5,
        "mask_temperature": 1.0, "description": "线缆弯曲"
    },
    "bent_lead": {
        "pattern": "structural",
        "z_amplify": 0.9, "z_sharpen": 0.2, "z_spread": 0.45,
        "mask_temperature": 1.0, "description": "引脚弯曲"
    },
    "fold": {
        "pattern": "structural",
        "z_amplify": 0.8, "z_sharpen": 0.2, "z_spread": 0.4,
        "mask_temperature": 1.1, "description": "折叠"
    },
    "squeeze": {
        "pattern": "structural",
        "z_amplify": 0.8, "z_sharpen": 0.2, "z_spread": 0.45,
        "mask_temperature": 1.05, "description": "挤压变形"
    },
    "squeezed_teeth": {
        "pattern": "structural",
        "z_amplify": 0.85, "z_sharpen": 0.2, "z_spread": 0.4,
        "mask_temperature": 1.05, "description": "齿部挤压"
    },
    "manipulated_front": {
        "pattern": "structural",
        "z_amplify": 0.9, "z_sharpen": 0.2, "z_spread": 0.45,
        "mask_temperature": 1.0, "description": "前端操作痕迹"
    },

    # === 印刷 / 表面类 ===
    "faulty_imprint": {
        "pattern": "surface",
        "z_amplify": 0.5, "z_sharpen": 0.2, "z_spread": 0.25,
        "mask_temperature": 1.2, "description": "印刷缺陷"
    },
    "print": {
        "pattern": "surface",
        "z_amplify": 0.4, "z_sharpen": 0.15, "z_spread": 0.3,
        "mask_temperature": 1.25, "description": "打印异常"
    },

    # === 缺失 / 错位类 ===
    "missing_cable": {
        "pattern": "missing",
        "z_amplify": 1.5, "z_sharpen": 0.5, "z_spread": 0.6,
        "mask_temperature": 0.6, "description": "线缆缺失"
    },
    "missing_wire": {
        "pattern": "missing",
        "z_amplify": 1.4, "z_sharpen": 0.5, "z_spread": 0.55,
        "mask_temperature": 0.65, "description": "导线缺失"
    },
    "misplaced": {
        "pattern": "structural",
        "z_amplify": 0.8, "z_sharpen": 0.2, "z_spread": 0.5,
        "mask_temperature": 1.0, "description": "位置偏移"
    },
    "flip": {
        "pattern": "structural",
        "z_amplify": 0.7, "z_sharpen": 0.2, "z_spread": 0.5,
        "mask_temperature": 1.0, "description": "翻转"
    },

    # === 混合 / 其他 ===
    "combined": {
        "pattern": "mixed",
        "z_amplify": 0.9, "z_sharpen": 0.35, "z_spread": 0.3,
        "mask_temperature": 1.0, "description": "复合缺陷"
    },
    "cable_swap": {
        "pattern": "structural",
        "z_amplify": 1.0, "z_sharpen": 0.2, "z_spread": 0.55,
        "mask_temperature": 0.95, "description": "线缆交换"
    },
    "damaged_case": {
        "pattern": "sharp_local",
        "z_amplify": 1.1, "z_sharpen": 0.5, "z_spread": 0.2,
        "mask_temperature": 0.75, "description": "外壳损坏"
    },
    "fabric_border": {
        "pattern": "structural",
        "z_amplify": 0.7, "z_sharpen": 0.2, "z_spread": 0.4,
        "mask_temperature": 1.1, "description": "织物边缘异常"
    },
    "fabric_interior": {
        "pattern": "diffuse",
        "z_amplify": 0.6, "z_sharpen": 0.15, "z_spread": 0.35,
        "mask_temperature": 1.2, "description": "织物内部异常"
    },
    "pill_type": {
        "pattern": "structural",
        "z_amplify": 0.7, "z_sharpen": 0.2, "z_spread": 0.4,
        "mask_temperature": 1.05, "description": "药片类型错误"
    },
    "defective": {
        "pattern": "mixed",
        "z_amplify": 0.8, "z_sharpen": 0.3, "z_spread": 0.3,
        "mask_temperature": 1.0, "description": "通用缺陷"
    },
    "liquid": {
        "pattern": "diffuse",
        "z_amplify": 0.5, "z_sharpen": 0.1, "z_spread": 0.4,
        "mask_temperature": 1.3, "description": "液体污染"
    },
}


def get_categories() -> List[str]:
    """返回所有可用的产品类别"""
    if not _DATASET_ROOT.exists():
        return []
    categories = []
    for d in sorted(_DATASET_ROOT.iterdir()):
        if d.is_dir() and (d / "test").exists():
            categories.append(d.name)
    return categories


def get_defect_types(category: str) -> List[Dict[str, str]]:
    """
    返回指定类别的真实缺陷类型列表

    Args:
        category: 产品类别名称，如 "bottle"

    Returns:
        [{"name": "broken_large", "description": "大面积断裂", "pattern": "sharp_local"}, ...]
    """
    test_dir = _DATASET_ROOT / category / "test"
    if not test_dir.exists():
        return []

    defect_types = []
    for d in sorted(test_dir.iterdir()):
        if d.is_dir() and d.name != "good":
            modulation = DEFECT_MODULATION.get(d.name, {})
            defect_types.append({
                "name": d.name,
                "description": modulation.get("description", d.name.replace("_", " ")),
                "pattern": modulation.get("pattern", "mixed"),
                "sample_count": len(list(d.glob("*.png"))),
            })
    return defect_types


def get_modulation_params(defect_type: str) -> dict:
    """
    获取指定缺陷类型的潜在空间调制参数

    Returns:
        {"pattern", "z_amplify", "z_sharpen", "z_spread", "mask_temperature", "description"}
    """
    default = {
        "pattern": "mixed",
        "z_amplify": 0.8,
        "z_sharpen": 0.3,
        "z_spread": 0.3,
        "mask_temperature": 1.0,
        "description": defect_type.replace("_", " "),
    }
    return DEFECT_MODULATION.get(defect_type, default)


def get_defect_sample(category: str, defect_type: str) -> Optional[Path]:
    """随机获取一张真实缺陷样本图像"""
    defect_dir = _DATASET_ROOT / category / "test" / defect_type
    if not defect_dir.exists():
        return None
    images = list(defect_dir.glob("*.png"))
    if not images:
        return None
    return random.choice(images)


def select_defect_types_for_augmentation(
    category: str,
    n: int = 5,
) -> List[Dict[str, any]]:
    """
    为一个类别选择 n 个缺陷类型用于增广。
    优先选择不同 pattern 的类型以确保多样性。

    Returns:
        [{"defect_type": "crack", "description": "裂纹", "modulation": {...}}, ...]
    """
    defect_types = get_defect_types(category)
    if not defect_types:
        return []

    # 按 pattern 分组
    by_pattern: Dict[str, list] = {}
    for dt in defect_types:
        pattern = dt["pattern"]
        by_pattern.setdefault(pattern, []).append(dt)

    # 轮询选取不同 pattern
    selected = []
    all_available = list(defect_types)  # 用于不足时补足
    pattern_keys = list(by_pattern.keys())
    idx = 0
    while len(selected) < n and by_pattern:
        pattern = pattern_keys[idx % len(pattern_keys)]
        if by_pattern[pattern]:
            dt = by_pattern[pattern].pop(0)
            modulation = get_modulation_params(dt["name"])
            selected.append({
                "defect_type": dt["name"],
                "description": dt["description"],
                "pattern": dt["pattern"],
                "modulation": modulation,
            })
        else:
            pattern_keys.remove(pattern)
            if not pattern_keys:
                break
            continue
        idx += 1

    # 不足 n 个时，从已有中轮询重复（不同随机种子会产生不同实例）
    while len(selected) < n and all_available:
        dt = all_available[len(selected) % len(all_available)]
        modulation = get_modulation_params(dt["name"])
        selected.append({
            "defect_type": f"{dt['name']}_v{len(selected) // len(all_available) + 1}",
            "description": dt["description"],
            "pattern": dt["pattern"],
            "modulation": modulation,
        })

    return selected
