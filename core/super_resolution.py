#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU 超分辨率模块 — 基于 Real-ESRGAN
用于云 GPU 环境（AutoDL / Colab / 等）的工业缺陷图像超分增强。

安装依赖（云环境首次运行）:
    pip install realesrgan

模型自动下载到 ~/.realesrgan/ 目录，首次运行稍慢。

用法:
    from super_resolution import SuperResolution
    sr = SuperResolution(device='cuda')          # GPU
    sr = SuperResolution(device='cpu')           # CPU 回退
    sr = SuperResolution(device='auto')          # 自动选择
    upscaled = sr.upscale(image_np, factor=4)    # numpy [H,W,3] uint8 RGB
    sr.upscale_file('input.png', 'output.png', factor=4)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# 抑制 Real-ESRGAN 的调试日志
logging.getLogger("realesrgan").setLevel(logging.WARNING)


class SuperResolution:
    """GPU 加速超分辨率（Real-ESRGAN），CPU 回退可选"""

    def __init__(
        self,
        device: str = "auto",
        model_name: str = "RealESRGAN_x4plus",
        tile_size: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
    ):
        """
        Args:
            device: 'auto' | 'cuda' | 'cpu'
            model_name: Real-ESRGAN 模型名
                - 'RealESRGAN_x4plus'        (4x, 通用, 质量最高)
                - 'RealESRGAN_x2plus'        (2x, 通用)
                - 'RealESRNet_x4plus'        (4x, 更平滑)
                - 'realesr-general-x4v3'     (4x, 通用场景 v3)
            tile_size: 分块推理大小 (0=不分块, 显存不足时设 256~512)
            tile_pad: 分块重叠像素
            pre_pad: 预处理填充
        """
        self.model_name = model_name
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad

        # 检测设备
        if device == "auto":
            try:
                import torch
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self._upsampler = None
        print(f"SuperResolution: device={self.device}, model={model_name}")

    def _get_upsampler(self):
        """延迟加载 Real-ESRGAN 上采样器"""
        if self._upsampler is not None:
            return self._upsampler

        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except ImportError as e:
            raise ImportError(
                "需要安装 Real-ESRGAN:\n"
                "  pip install realesrgan\n"
                f"原始错误: {e}"
            )

        # 根据模型名确定网络架构和缩放倍率
        model_config = self._get_model_config()
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=model_config["scale"],
        )

        # Real-ESRGAN 会自动下载模型权重
        self._upsampler = RealESRGANer(
            scale=model_config["scale"],
            model_path=None,  # 自动下载
            dni_weight=None,
            model=model,
            tile=self.tile_size,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=True if self.device != "cpu" else False,
            device=self.device,
        )
        print(f"Real-ESRGAN 模型加载完成: {self.model_name} ({model_config['scale']}x)")
        return self._upsampler

    def _get_model_config(self) -> dict:
        """返回模型的缩放倍率等配置"""
        configs = {
            "RealESRGAN_x4plus": {"scale": 4},
            "RealESRGAN_x2plus": {"scale": 2},
            "RealESRNet_x4plus": {"scale": 4},
            "realesr-general-x4v3": {"scale": 4},
        }
        if self.model_name in configs:
            return configs[self.model_name]
        # 尝试从模型名推断
        for k, v in configs.items():
            if k.lower() in self.model_name.lower():
                return v
        return {"scale": 4}  # 默认 4x

    def upscale(self, image: np.ndarray, factor: Optional[int] = None) -> np.ndarray:
        """
        对单张图像进行超分辨率重建

        Args:
            image: RGB uint8 [H, W, 3]
            factor: 目标倍率 (必须与模型兼容，None 则使用模型默认倍率)

        Returns:
            RGB uint8 超分图像
        """
        upsampler = self._get_upsampler()
        model_scale = self._get_model_config()["scale"]

        if factor is not None and factor != model_scale:
            # 组合放大：先 ESRGAN 再 Lanczos
            h, w = image.shape[:2]
            intermediate = upsampler.enhance(image, outscale=model_scale)[0]
            target_w = int(w * factor)
            target_h = int(h * factor)
            if (intermediate.shape[1], intermediate.shape[0]) != (target_w, target_h):
                intermediate = cv2.resize(
                    intermediate, (target_w, target_h),
                    interpolation=cv2.INTER_LANCZOS4
                )
            return intermediate
        else:
            output, _ = upsampler.enhance(image, outscale=model_scale)
            return output

    def upscale_file(
        self,
        input_path: str,
        output_path: str,
        factor: Optional[int] = None,
    ) -> str:
        """
        读取图像文件 → 超分 → 保存

        Returns:
            输出路径
        """
        img = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {input_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.upscale(img, factor=factor)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        _, buf = cv2.imencode(Path(output_path).suffix, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        with open(output_path, 'wb') as f:
            f.write(buf.tobytes())

        h, w = result.shape[:2]
        print(f"超分完成: {input_path} → {output_path} ({w}x{h})")
        return output_path


# ============================================================
# 便捷函数：集成到 real_defect_blender 管线
# ============================================================

def create_sr_upscaler(device: str = "auto", model: str = "RealESRGAN_x4plus"):
    """创建超分放大函数，可直接传给 RealDefectBlender"""
    try:
        sr = SuperResolution(device=device, model_name=model)
        def upscale_fn(image: np.ndarray, factor: float) -> np.ndarray:
            return sr.upscale(image, factor=int(factor))
        return upscale_fn, sr
    except ImportError:
        return None, None


# ============================================================
# 命令行
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU 超分辨率（Real-ESRGAN）")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output", type=str, required=True, help="输出图像路径")
    parser.add_argument("--model", type=str, default="RealESRGAN_x4plus",
                        help="模型名称")
    parser.add_argument("--factor", type=int, default=None,
                        help="放大倍率 (默认使用模型默认倍率)")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备: auto/cuda/cpu")
    parser.add_argument("--tile", type=int, default=0,
                        help="分块大小 (0=不分块, 显存不足设 400)")

    args = parser.parse_args()

    sr = SuperResolution(
        device=args.device,
        model_name=args.model,
        tile_size=args.tile,
    )
    sr.upscale_file(args.input, args.output, factor=args.factor)


if __name__ == "__main__":
    main()
