#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成Slide 10 图2(CBAM结构) 和 图3(融合模块公式)"""

import sys, os
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT = Path('output_figures')
OUTPUT.mkdir(exist_ok=True)

C = {
    'blue': '#2E86AB', 'red': '#D64045', 'orange': '#F18F01',
    'green': '#2ECC71', 'purple': '#8E44AD', 'dark': '#2C3E50',
    'gray': '#95A5A6', 'light_blue': '#85C1E9', 'bg': '#F8F9FA',
    'ca': '#3498DB', 'sa': '#E67E22', 'arrow': '#7F8C8D'
}

# ============================================================
# 图2：CBAM模块结构图
# ============================================================
def generate_cbam_diagram():
    fig, (ax_cha, ax_spa) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle('CBAM (Convolutional Block Attention Module)',
                 fontsize=14, fontweight='bold', y=0.98)

    # ---- 通道注意力 ----
    ax_cha.set_xlim(0, 12)
    ax_cha.set_ylim(0, 4)
    ax_cha.axis('off')
    ax_cha.set_title('Channel Attention: "What" features to focus on',
                     fontsize=12, fontweight='bold', color=C['ca'], pad=8)

    # 输入特征图
    def draw_box(ax, x, y, w, h, text, color, fontsize=9, text_color='white'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color)

    # Input
    draw_box(ax_cha, 0.2, 1.2, 1.6, 1.4, 'Input\nFeature\nH×W×C', C['dark'])
    # GAP
    draw_box(ax_cha, 2.4, 1.2, 1.4, 1.4, 'Global\nAvgPool', C['ca'])
    # MaxPool
    draw_box(ax_cha, 2.4, 2.8, 1.4, 0.8, 'MaxPool', '#2980B9')
    # Shared MLP
    draw_box(ax_cha, 4.4, 1.7, 1.6, 1.1, 'Shared\nMLP', C['ca'])
    # Add
    draw_box(ax_cha, 6.6, 1.7, 0.8, 1.1, '⊕', C['orange'], fontsize=14, text_color='white')
    # Sigmoid
    draw_box(ax_cha, 8.0, 1.7, 1.2, 1.1, 'Sigmoid', C['dark'])
    # Channel Attention
    draw_box(ax_cha, 9.8, 1.2, 1.8, 1.4, 'Channel\nAttention\n1×1×C', C['green'], text_color='white')

    # Arrows
    for (x1, x2, y) in [(1.8, 2.4, 1.9), (3.8, 4.4, 2.1), (6.0, 6.6, 2.25), (7.4, 8.0, 2.25), (9.2, 9.8, 1.9)]:
        ax_cha.annotate('', xy=(x2, y), xytext=(x1, y),
                        arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.5))

    # ---- 空间注意力 ----
    ax_spa.set_xlim(0, 12)
    ax_spa.set_ylim(0, 4)
    ax_spa.axis('off')
    ax_spa.set_title('Spatial Attention: "Where" to focus',
                     fontsize=12, fontweight='bold', color=C['sa'], pad=8)

    # Input refined features
    draw_box(ax_spa, 0.2, 1.2, 1.6, 1.4, 'Refined\nFeature\nH×W×C', C['dark'])
    # [AvgPool, MaxPool]
    draw_box(ax_spa, 2.4, 0.8, 1.6, 1.1, 'AvgPool\n(Channel)', C['sa'])
    draw_box(ax_spa, 2.4, 2.1, 1.6, 1.1, 'MaxPool\n(Channel)', '#D35400')
    # Concat
    draw_box(ax_spa, 4.6, 1.3, 1.4, 1.4, 'Concat\nH×W×2', C['sa'])
    # Conv7x7
    draw_box(ax_spa, 6.6, 1.3, 1.4, 1.4, 'Conv\n7×7', C['sa'])
    # Sigmoid
    draw_box(ax_spa, 8.6, 1.3, 1.2, 1.4, 'Sigmoid', C['dark'])
    # Spatial Attention
    draw_box(ax_spa, 10.3, 1.3, 1.5, 1.4, 'Spatial\nAttention\nH×W×1', C['green'], text_color='white')

    # Arrows for spatial
    for (x1, x2, y) in [(1.8, 2.4, 1.35), (1.8, 2.4, 2.65), (4.0, 4.6, 2.0), (6.0, 6.6, 2.0), (8.0, 8.6, 2.0), (9.8, 10.3, 2.0)]:
        ax_spa.annotate('', xy=(x2, y), xytext=(x1, y),
                        arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.5))

    plt.tight_layout()
    fig.savefig(OUTPUT / 'fig10_cbam_diagram.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('已保存: fig10_cbam_diagram.png')

# ============================================================
# 图3：融合模块公式示意图
# ============================================================
def generate_fusion_formula():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Attention-Guided Fusion Module  |  Paper Section 3.2.4',
                 fontsize=14, fontweight='bold', pad=12)

    # 输入框
    def draw_block(ax, x, y, w, h, text, color, fontsize=11, tc='white'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                             facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=tc)

    # I_defect
    draw_block(ax, 0.3, 2.5, 2.2, 1.2, 'I_defect\n(Defect Image)', C['blue'])
    # I_background
    draw_block(ax, 0.3, 0.8, 2.2, 1.2, 'I_background\n(Background Image)', C['green'])
    # Conv + Concat
    draw_block(ax, 3.3, 1.7, 2.0, 1.5, 'Conv → Concat\n→ Attention\nNetwork', C['orange'])
    # Sigmoid -> alpha
    draw_block(ax, 6.1, 2.0, 1.8, 1.2, 'α = Sigmoid(...)\nα ∈ [0,1]^{H×W}', C['dark'], fontsize=10)
    # Weighted sum
    draw_block(ax, 8.7, 0.6, 2.5, 2.8,
               'Output =\nα ⊙ I_defect\n+\n(1−α) ⊙ I_background',
               C['purple'], fontsize=12)
    # Refinement
    draw_block(ax, 12.0, 1.7, 1.6, 1.5, 'Refinement\nConv Block\n→ Final', C['red'])

    # Arrows
    arrows_data = [
        (2.5, 3.1, 3.3, 2.45), (2.5, 1.4, 3.3, 2.45),
        (5.3, 2.45, 6.1, 2.6), (7.9, 2.6, 8.7, 1.8),
        (11.2, 2.45, 12.0, 2.45)
    ]
    for x1, y1, x2, y2 in arrows_data:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=2.0))

    # 公式标注
    ax.text(3.3, 4.8, 'Formula (3-1):', fontsize=10, fontweight='bold', color=C['dark'])
    ax.text(3.3, 4.3, 'α = σ( Conv( [I_defect; I_background] ) )',
            fontsize=11, fontfamily='monospace', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#EBF5FB', alpha=0.8))

    ax.text(8.0, 0.1, 'Formula (3-2):', fontsize=10, fontweight='bold', color=C['dark'])
    ax.text(8.0, -0.4, 'I_output = α ⊙ I_defect + (1−α) ⊙ I_background',
            fontsize=11, fontfamily='monospace', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FEF9E7', alpha=0.8))

    plt.tight_layout()
    fig.savefig(OUTPUT / 'fig10_fusion_formula.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('已保存: fig10_fusion_formula.png')

# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    generate_cbam_diagram()
    generate_fusion_formula()
    print('\nSlide 10 图2图3 完成!')
