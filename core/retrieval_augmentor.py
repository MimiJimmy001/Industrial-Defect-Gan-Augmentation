#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检索式增广器 — 从数据集中按相似度检索同类型样本，或堆叠同类不同属性缺陷。

两种模式:
  1. retrieval  — 提取图像特征向量，在数据集中搜索最相近的 top-K 张图像直接输出
  2. stacking   — 从同类型中选取不同属性缺陷，几何变换后堆叠融合到正常图像上

特征索引可预构建并缓存到磁盘，避免每次检索都遍历整个数据集。
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 数据集根目录
_DATASET_ROOT = Path(__file__).parent.parent / "datasets" / "mvtec_anomaly_detection"


# ============================================================
#  特征提取（统一实现见 core/features.py，此处做兼容再导出）
# ============================================================

try:
    from core.features import (
        FEATURE_KEYS,
        extract_feature_vector,
        features_vector_to_dict,
        features_dict_to_vector,
        build_features_from_bgr,
        build_features_from_rgb,
    )
except ImportError:
    _root = Path(__file__).parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from core.features import (
        FEATURE_KEYS,
        extract_feature_vector,
        features_vector_to_dict,
        features_dict_to_vector,
        build_features_from_bgr,
        build_features_from_rgb,
    )


# ============================================================
#  数据集特征索引
# ============================================================

class DatasetFeatureIndex:
    """
    数据集特征索引，支持预构建和磁盘缓存。

    为 MVTec AD 数据集中每张图像预计算特征向量，
    按 category/defect_type 组织，支持快速相似度检索。
    """

    def __init__(self, dataset_root: Path = None, cache_path: Path = None):
        self.dataset_root = dataset_root or _DATASET_ROOT
        self.cache_path = cache_path or (Path(__file__).parent / "feature_index_cache.json")

        # 结构: {category: {defect_type: [(image_path, feature_vec_6), ...]}}
        self._index: Dict[str, Dict[str, List[Tuple[str, np.ndarray]]]] = {}
        self._built = False

    # ---- 序列化辅助 ----
    @staticmethod
    def _serialize(index: dict) -> dict:
        out = {}
        for cat, types in index.items():
            out[cat] = {}
            for dt, entries in types.items():
                out[cat][dt] = [[p, vec.tolist()] for p, vec in entries]
        return out

    @staticmethod
    def _deserialize(data: dict) -> dict:
        index = {}
        for cat, types in data.items():
            index[cat] = {}
            for dt, entries in types.items():
                index[cat][dt] = [(p, np.array(v, dtype=np.float32)) for p, v in entries]
        return index

    def build(self, force_rebuild: bool = False) -> int:
        """遍历数据集构建特征索引，返回索引图像总数"""
        if self._built and not force_rebuild:
            return sum(len(v) for t in self._index.values() for v in t.values())

        # 尝试加载缓存
        if not force_rebuild and self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._index = self._deserialize(json.load(f))
                self._built = True
                total = sum(len(v) for t in self._index.values() for v in t.values())
                print(f"从缓存加载特征索引: {total} 张图像 ({self.cache_path})")
                return total
            except Exception as e:
                print(f"缓存加载失败 ({e})，重新构建")

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"数据集不存在: {self.dataset_root}")

        total = 0
        for cat_dir in sorted(self.dataset_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat = cat_dir.name
            test_dir = cat_dir / "test"
            if not test_dir.exists():
                continue

            self._index.setdefault(cat, {})

            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                dt = defect_dir.name
                if dt == "good":
                    continue  # 跳过正常图像，只索引缺陷样本
                entries = []
                for img_path in sorted(defect_dir.glob("*.png")):
                    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    vec = extract_feature_vector(img)
                    entries.append((str(img_path), vec))
                if entries:
                    self._index[cat][dt] = entries
                    total += len(entries)

        # 保存缓存
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._serialize(self._index), f, ensure_ascii=False)
            print(f"特征索引导出: {self.cache_path} ({total} 张)")
        except Exception as e:
            print(f"缓存保存失败: {e}")

        self._built = True
        print(f"特征索引构建完成: {total} 张图像, {len(self._index)} 个类别")
        return total

    def get_categories(self) -> List[str]:
        return sorted(self._index.keys())

    def get_defect_types(self, category: str) -> List[str]:
        return sorted(self._index.get(category, {}).keys())

    def search_similar(
        self,
        query_vec: np.ndarray,
        category: str,
        defect_type: str = None,
        top_k: int = 5,
        exclude_paths: List[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        在指定类别/缺陷类型中搜索最相似的 top_k 张图像。

        Args:
            query_vec: 查询特征向量 (6,)
            category: 产品类别
            defect_type: 缺陷类型，None 则搜索该类别所有缺陷类型
            top_k: 返回最相似的 K 个结果
            exclude_paths: 排除的路径列表

        Returns:
            [(image_path, similarity_score), ...] 按相似度降序
        """
        if not self._built:
            raise RuntimeError("索引未构建，请先调用 build()")

        exclude = set(exclude_paths or [])

        # 收集候选
        candidates = []
        cat_index = self._index.get(category, {})

        types_to_search = [defect_type] if defect_type else list(cat_index.keys())
        for dt in types_to_search:
            for img_path, vec in cat_index.get(dt, []):
                if img_path in exclude:
                    continue
                sim = self._cosine_similarity(query_vec, vec)
                candidates.append((img_path, sim, dt))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [(p, s) for p, s, _ in candidates[:top_k]]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-10:
            return 0.0
        return float(np.dot(a, b) / denom)


# ============================================================
#  检索式增广
# ============================================================

class RetrievalAugmentor:
    """检索式增广：从数据集中选最相近的同类图像作为增广输出"""

    def __init__(self, dataset_root: Path = None, cache_path: Path = None):
        self.dataset_root = dataset_root or _DATASET_ROOT
        self.index = DatasetFeatureIndex(dataset_root, cache_path)

    def build_index(self, force: bool = False) -> int:
        return self.index.build(force)

    def augment_by_retrieval(
        self,
        image_path: str,
        output_dir: str,
        category: str,
        n_variations: int = 5,
        top_k: int = None,
    ) -> List[str]:
        """
        检索式增广：提取输入图像特征，从同类型中找最相似的图像直接输出。

        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            category: 产品类别 (如 'bottle')
            n_variations: 输出变体数
            top_k: 检索候选数，默认等于 n_variations

        Returns:
            输出文件路径列表
        """
        if top_k is None:
            top_k = n_variations

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        query_vec = extract_feature_vector(img)

        stem = Path(image_path).stem
        query_abs = str(Path(image_path).resolve())
        results = self.index.search_similar(
            query_vec, category=category, top_k=max(top_k, n_variations * 3),
            exclude_paths=[query_abs],
        )

        if not results:
            print(f"警告: 未在类别 '{category}' 中找到相似图像")
            return []

        # 从检索结果中按相似度 + 随机采样选出 n_variations 张
        # 前 60% 按相似度选，后 40% 从剩余中随机选（增加多样性）
        n_top = max(1, int(n_variations * 0.6))
        n_rand = n_variations - n_top

        selected = results[:n_top]
        if n_rand > 0 and len(results) > n_top:
            rest = results[n_top:]
            selected += random.sample(rest, min(n_rand, len(rest)))

        output_paths = []
        for i, (src_path, sim_score) in enumerate(selected):
            src_img = cv2.imdecode(np.fromfile(src_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if src_img is None:
                continue

            out_path = output_dir / f"{stem}_retrieved_{i:02d}.png"
            cv2.imencode(".png", src_img)[1].tofile(str(out_path))
            output_paths.append(str(out_path))
            print(f"  [{i+1}/{len(selected)}] 相似度={sim_score:.4f}  {Path(src_path).name}")

        return output_paths


# ============================================================
#  缺陷堆叠增广
# ============================================================

class DefectStackingAugmentor:
    """
    缺陷堆叠增广：从同类型中选取不同属性缺陷，几何变换后叠加到正常图像上。

    不同于 RealDefectBlender 的随机选取，本类专注于:
      - 同类型内不同属性缺陷的组合堆叠
      - 按缺陷 pattern 互补性选择堆叠组合
      - 控制堆叠密度和位置避免缺陷重叠覆盖
    """

    def __init__(self, category: str = None, dataset_root: Path = None,
                 rng_seed: int = None):
        self.dataset_root = dataset_root or _DATASET_ROOT
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        # 扫描可用类别
        self._categories: Dict[str, List[str]] = {}
        self._init_categories()

        self.category = category or random.choice(list(self._categories.keys()))
        if self.category not in self._categories:
            raise ValueError(f"类别 '{self.category}' 不可用，可选: {list(self._categories.keys())}")

    def _init_categories(self):
        if not self.dataset_root.exists():
            return
        for cat_dir in sorted(self.dataset_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            gt_dir = cat_dir / "ground_truth"
            test_dir = cat_dir / "test"
            if not gt_dir.exists() or not test_dir.exists():
                continue
            defect_types = []
            for d in sorted(test_dir.iterdir()):
                if d.is_dir() and d.name != "good" and (gt_dir / d.name).exists():
                    defect_types.append(d.name)
            if defect_types:
                self._categories[cat_dir.name] = defect_types

    def get_categories(self) -> List[str]:
        return sorted(self._categories.keys())

    def get_defect_types(self, category: str = None) -> List[str]:
        return self._categories.get(category or self.category, [])

    # ---- 缺陷 pattern 分组 ----
    # 主数据源为 defect_registry.DEFECT_MODULATION（单一维护点），
    # 本地映射仅作为 registry 不可用时的回退
    _PATTERN_MAP_FALLBACK = {
        # directional
        "scratch": "directional", "scratch_head": "directional", "scratch_neck": "directional",
        "cut": "directional", "cut_inner_insulation": "directional", "cut_outer_insulation": "directional",
        "crack": "directional", "thread": "directional", "thread_side": "directional",
        "thread_top": "directional", "gray_stroke": "directional",
        # sharp_local
        "broken_large": "sharp_local", "broken_small": "sharp_local", "broken_teeth": "sharp_local",
        "split_teeth": "sharp_local", "damaged_case": "sharp_local",
        # localized
        "hole": "localized", "poke": "localized", "poke_insulation": "localized",
        # diffuse
        "contamination": "diffuse", "color": "diffuse", "oil": "diffuse",
        "glue": "diffuse", "glue_strip": "diffuse", "metal_contamination": "diffuse",
        "rough": "diffuse", "liquid": "diffuse", "fabric_interior": "diffuse",
        # structural
        "bent": "structural", "bent_wire": "structural", "bent_lead": "structural",
        "fold": "structural", "squeeze": "structural", "squeezed_teeth": "structural",
        "manipulated_front": "structural", "misplaced": "structural", "flip": "structural",
        "cable_swap": "structural", "fabric_border": "structural", "pill_type": "structural",
        # surface
        "faulty_imprint": "surface", "print": "surface",
        # missing
        "missing_cable": "missing", "missing_wire": "missing",
        # mixed
        "combined": "mixed", "defective": "mixed",
    }

    @classmethod
    def _get_pattern(cls, defect_type: str) -> str:
        """优先从 defect_registry 获取 pattern（单一维护点），失败时回退本地映射"""
        try:
            from core.defect_registry import DEFECT_MODULATION
            if defect_type in DEFECT_MODULATION:
                return DEFECT_MODULATION[defect_type].get("pattern", "mixed")
        except ImportError:
            pass
        return cls._PATTERN_MAP_FALLBACK.get(defect_type, "mixed")

    def select_complementary_defects(
        self, n: int = 3, prefer_different_pattern: bool = True,
    ) -> List[str]:
        """
        选择互补的缺陷类型用于堆叠。

        优先选择不同 pattern 的缺陷类型，使堆叠结果覆盖更多视觉特征。
        """
        available = self.get_defect_types()
        if not available:
            return []

        if not prefer_different_pattern or len(available) <= n:
            return random.sample(available, min(n, len(available)))

        # 按 pattern 分组
        by_pattern: Dict[str, List[str]] = {}
        for dt in available:
            p = self._get_pattern(dt)
            by_pattern.setdefault(p, []).append(dt)

        # 轮询选取不同 pattern
        selected = []
        patterns = list(by_pattern.keys())
        idx = 0
        while len(selected) < n and patterns:
            p = patterns[idx % len(patterns)]
            if by_pattern[p]:
                selected.append(by_pattern[p].pop())
            else:
                patterns.remove(p)
                continue
            idx += 1

        # 不足则随机补足
        remaining = [dt for dt in available if dt not in selected]
        while len(selected) < n and remaining:
            selected.append(remaining.pop(random.randint(0, len(remaining) - 1)))

        return selected[:n]

    # ---- 缺陷样本加载 ----
    def _load_defect_sample(self, defect_type: str
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载一个缺陷样本及其 mask。

        Returns:
            (defect_img_bgr, mask_gray) — BGR uint8
        """
        cat = self.category
        test_dir = self.dataset_root / cat / "test" / defect_type
        gt_dir = self.dataset_root / cat / "ground_truth" / defect_type

        images = sorted(test_dir.glob("*.png"))
        if not images:
            raise ValueError(f"类别 {cat}/{defect_type} 无图像")
        img_path = random.choice(images)
        defect_img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)

        mask_name = img_path.stem.replace(".", "") + "_mask.png"
        mask_path = gt_dir / mask_name
        if mask_path.exists():
            mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            diff = cv2.absdiff(defect_img, defect_img)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

        return defect_img, mask

    # ---- 几何变换（委托 core.blending_utils 统一实现） ----
    @staticmethod
    def _geometric_transform(defect_roi: np.ndarray, mask_roi: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """随机缩放 + 旋转 + 翻转"""
        from core.blending_utils import random_geometric_transform
        return random_geometric_transform(defect_roi, mask_roi)

    # ---- ROI 提取（委托 core.blending_utils） ----
    @staticmethod
    def _extract_roi(img: np.ndarray, mask: np.ndarray, padding: int = 4
                     ) -> Tuple[np.ndarray, np.ndarray]:
        from core.blending_utils import extract_roi
        return extract_roi(img, mask, padding)

    # ---- 色彩匹配（委托 core.blending_utils） ----
    @staticmethod
    def _color_match(source: np.ndarray, target_bg: np.ndarray, mask: np.ndarray,
                     strength: float = 0.45) -> np.ndarray:
        from core.blending_utils import color_match
        return color_match(source, target_bg, mask, strength)

    # ---- Alpha 融合（委托 core.blending_utils） ----
    @staticmethod
    def _alpha_blend(defect: np.ndarray, mask: np.ndarray, background: np.ndarray,
                     position: Tuple[int, int]) -> np.ndarray:
        from core.blending_utils import alpha_blend
        return alpha_blend(defect, mask, background, position, feather=0.5)

    # ---- 主方法：堆叠增广 ----
    def stack_defects(
        self,
        good_image_path: str,
        output_path: str,
        n_defects: int = None,
        defect_types: List[str] = None,
        intensity: float = 1.0,
    ) -> str:
        """
        将同类型内不同属性缺陷堆叠到正常图像上。

        Args:
            good_image_path: 正常(good)图像路径
            output_path: 输出路径
            n_defects: 堆叠缺陷数量 (1~4)
            defect_types: 指定缺陷类型列表，None 则自动选择互补类型
            intensity: 缺陷强度 0.5~1.5

        Returns:
            输出文件路径
        """
        if n_defects is None:
            n_defects = random.randint(1, 3)

        if defect_types is None:
            defect_types = self.select_complementary_defects(n_defects)
        else:
            defect_types = defect_types[:n_defects]

        img = cv2.imdecode(np.fromfile(good_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {good_image_path}")
        h, w = img.shape[:2]

        # 记录已占用的位置矩形，避免缺陷重叠
        occupied_rects: List[Tuple[int, int, int, int]] = []

        for dt in defect_types:
            try:
                defect_img, mask = self._load_defect_sample(dt)
            except (ValueError, FileNotFoundError) as e:
                print(f"  加载缺陷类型 '{dt}' 失败: {e}")
                continue

            # 提取 ROI
            defect_roi, mask_roi = self._extract_roi(defect_img, mask)
            if (mask_roi > 30).sum() < 15:
                continue

            # 几何变换
            defect_roi, mask_roi = self._geometric_transform(defect_roi, mask_roi)
            if (mask_roi > 30).sum() < 10:
                continue

            # 约束尺寸
            dh, dw = defect_roi.shape[:2]
            max_dim = int(min(h, w) * 0.6)
            min_dim = int(min(h, w) * 0.04)
            if max(dh, dw) > max_dim:
                s = max_dim / max(dh, dw)
                defect_roi = cv2.resize(defect_roi, (max(6, int(dw * s)), max(6, int(dh * s))),
                                        interpolation=cv2.INTER_LANCZOS4)
                mask_roi = cv2.resize(mask_roi, (max(6, int(dw * s)), max(6, int(dh * s))),
                                      interpolation=cv2.INTER_NEAREST)
            elif max(dh, dw) < min_dim:
                s = min_dim / max(dh, dw)
                defect_roi = cv2.resize(defect_roi, (int(dw * s), int(dh * s)),
                                        interpolation=cv2.INTER_LANCZOS4)
                mask_roi = cv2.resize(mask_roi, (int(dw * s), int(dh * s)),
                                      interpolation=cv2.INTER_NEAREST)

            dh, dw = defect_roi.shape[:2]

            # 选位置（避免与已有缺陷重叠）
            margin = 3
            for _ in range(20):
                max_x = max(margin, w - dw - margin)
                max_y = max(margin, h - dh - margin)
                if max_x <= margin or max_y <= margin:
                    x, y = margin, margin
                else:
                    x = random.randint(margin, max_x)
                    y = random.randint(margin, max_y)

                new_rect = (x, y, x + dw, y + dh)
                if not self._has_overlap(new_rect, occupied_rects, iou_threshold=0.25):
                    break
            else:
                # 20次尝试后仍重叠则跳过
                continue

            occupied_rects.append(new_rect)

            # 色彩匹配
            bg_roi = img[y:y + dh, x:x + dw]
            defect_roi = self._color_match(defect_roi, bg_roi, mask_roi, strength=random.uniform(0.3, 0.55))

            # Alpha 融合
            img = self._alpha_blend(defect_roi, mask_roi, img, (x, y))

        # 后处理
        img = self._post_process(img, intensity)

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", img)[1].tofile(output_path)
        return output_path

    @staticmethod
    def _has_overlap(rect: Tuple[int, int, int, int],
                     rects: List[Tuple[int, int, int, int]],
                     iou_threshold: float = 0.2) -> bool:
        from core.blending_utils import has_overlap
        return has_overlap(rect, rects, iou_threshold)

    @staticmethod
    def _post_process(image: np.ndarray, intensity: float) -> np.ndarray:
        """轻量后处理：锐化 + 微噪声"""
        kernel = np.array([[-0.3, -0.3, -0.3],
                           [-0.3,  3.4, -0.3],
                           [-0.3, -0.3, -0.3]], dtype=np.float32)
        sharp = cv2.filter2D(image, -1, kernel)
        strength = 0.08 * intensity
        result = cv2.addWeighted(image, 1 - strength, sharp, strength, 0)

        noise = np.random.randn(*result.shape).astype(np.float32) * (0.6 * intensity)
        return np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # ---- 批量生成 ----
    def generate_batch(
        self,
        good_image_path: str,
        output_dir: str,
        n_variations: int = 5,
        n_defects: int = None,
        intensity: float = 1.0,
    ) -> List[str]:
        """为一张正常图像批量生成堆叠缺陷变体"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(good_image_path).stem
        paths = []
        for i in range(n_variations):
            out_path = output_dir / f"{stem}_stacked_{i:02d}.png"
            self.stack_defects(
                good_image_path, str(out_path),
                n_defects=n_defects, defect_types=None,
                intensity=intensity * random.uniform(0.8, 1.2),
            )
            paths.append(str(out_path))
        return paths


# ============================================================
#  便捷函数
# ============================================================

def build_index_once(dataset_root: str = None, cache_path: str = None,
                     force: bool = False) -> DatasetFeatureIndex:
    """构建特征索引（单例模式，可重复调用）"""
    idx = DatasetFeatureIndex(
        Path(dataset_root) if dataset_root else None,
        Path(cache_path) if cache_path else None,
    )
    idx.build(force_rebuild=force)
    return idx


_INDEX_INSTANCE: Optional[DatasetFeatureIndex] = None


def get_index(dataset_root: str = None, cache_path: str = None
              ) -> DatasetFeatureIndex:
    """获取全局单例索引"""
    global _INDEX_INSTANCE
    if _INDEX_INSTANCE is None:
        _INDEX_INSTANCE = build_index_once(dataset_root, cache_path)
    return _INDEX_INSTANCE


# ============================================================
#  命令行接口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="检索式/堆叠式增广器")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["retrieval", "stacking", "build_index"],
                        help="运行模式")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output", type=str, required=True, help="输出路径/目录")
    parser.add_argument("--category", type=str, required=True, help="产品类别 (如 bottle)")
    parser.add_argument("--n_variations", type=int, default=5, help="生成变体数")
    parser.add_argument("--n_defects", type=int, default=None, help="堆叠缺陷数")
    parser.add_argument("--intensity", type=float, default=1.0, help="缺陷强度")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.mode == "build_index":
        idx = build_index_once(force=True)
        print(f"索引构建完成，共 {sum(len(v) for t in idx._index.values() for v in t.values())} 张图像")
        return

    if args.mode == "retrieval":
        aug = RetrievalAugmentor()
        aug.build_index()
        paths = aug.augment_by_retrieval(
            args.input, args.output, args.category, n_variations=args.n_variations,
        )
        print(f"检索增广完成: {len(paths)} 张 -> {args.output}")

    elif args.mode == "stacking":
        aug = DefectStackingAugmentor(category=args.category)
        paths = aug.generate_batch(
            args.input, args.output, n_variations=args.n_variations,
            n_defects=args.n_defects, intensity=args.intensity,
        )
        print(f"堆叠增广完成: {len(paths)} 张 -> {args.output}")


if __name__ == "__main__":
    main()
