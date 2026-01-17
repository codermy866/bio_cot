#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件生成信息:
- 生成时间: 2025-01-XX 15:35:00
- 更新时间: 2025-12-25 09:14:24 (使用nvidia-smi获取CUDA时间)
- 生成需求: 用户要求统一的可视化配置，包括统一的英文字体和8种配色方案（#c7522a, #e5c185, #f0daa5, #fbf2c4, #b8cdab, #74a892, #008585, #004343）
- 更新需求: 用户要求使用Arial字体，适合SCI论文
- 生成原因: MICCAI论文需要统一的可视化风格，确保所有图表使用相同的字体和配色方案
- 更新原因: SCI论文通常要求使用Arial字体，优先使用Arial，如果不可用则使用Liberation Sans（Arial的开源替代）
- 相关任务: MICCAI论文准备和可视化标准化

文件功能: MICCAI论文可视化配置，提供统一的字体和配色方案，设置matplotlib样式；已更新：优先使用Arial字体（适合SCI论文）
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ==================== 配色方案 ====================
# MICCAI论文统一配色
MICCAI_COLORS = [
    '#c7522a',  # 深红棕色
    '#e5c185',  # 浅金色
    '#f0daa5',  # 米黄色
    '#fbf2c4',  # 浅米色
    '#b8cdab',  # 浅绿色
    '#74a892',  # 青绿色
    '#008585',  # 深青色
    '#004343',  # 深墨绿色
]

# RGB格式（用于某些库）
MICCAI_COLORS_RGB = [
    (199, 82, 42),    # #c7522a
    (229, 193, 133),  # #e5c185
    (240, 218, 165),  # #f0daa5
    (251, 242, 196),  # #fbf2c4
    (184, 205, 171),  # #b8cdab
    (116, 168, 146),  # #74a892
    (0, 133, 133),    # #008585
    (0, 67, 67),      # #004343
]

# 归一化RGB（0-1范围）
MICCAI_COLORS_RGB_NORM = [(r/255, g/255, b/255) for r, g, b in MICCAI_COLORS_RGB]

# ==================== 字体配置 ====================
# 设置matplotlib使用Arial字体（SCI论文要求）
def setup_matplotlib_fonts():
    """配置matplotlib使用Arial字体（优先），适合SCI论文"""
    # 优先使用Arial及其变体（SCI论文常用字体）
    font_candidates = [
        'Arial',
        'Arial Unicode MS',
        'Arial Black',
        'Liberation Sans',  # Arial的开源替代，外观相似
        'Helvetica',  # Arial的替代
        'DejaVu Sans',
        'sans-serif',
    ]
    
    # 获取所有可用字体（包括完整路径信息）
    available_fonts = {}
    for font in fm.fontManager.ttflist:
        font_name = font.name
        if font_name not in available_fonts:
            available_fonts[font_name] = font
    
    # 查找Arial相关字体（不区分大小写）
    arial_fonts = [f for f in available_fonts.keys() if 'arial' in f.lower()]
    
    # 优先使用Arial
    for font_name in font_candidates:
        # 精确匹配
        if font_name in available_fonts:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['font.sans-serif'] = [font_name] + [f for f in font_candidates if f != font_name]
            print(f"✓ Using font: {font_name} (for SCI paper)")
            return font_name
        # 模糊匹配（不区分大小写）
        for available_font in available_fonts.keys():
            if font_name.lower() in available_font.lower() or available_font.lower() in font_name.lower():
                plt.rcParams['font.family'] = available_font
                plt.rcParams['font.sans-serif'] = [available_font] + [f for f in font_candidates if f != font_name]
                print(f"✓ Using font: {available_font} (similar to {font_name}, for SCI paper)")
                return available_font
    
    # 如果没有找到Arial，使用Liberation Sans（Arial的开源替代，外观非常相似）
    if 'Liberation Sans' in available_fonts:
        plt.rcParams['font.family'] = 'Liberation Sans'
        plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'Arial', 'DejaVu Sans']
        print("✓ Using font: Liberation Sans (Arial alternative, for SCI paper)")
        return 'Liberation Sans'
    
    # 最后备选
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans']
    print("⚠ Using default sans-serif (Arial preferred but not found)")
    return 'sans-serif'

# ==================== 样式配置 ====================
def setup_miccai_style():
    """设置MICCAI论文的matplotlib样式"""
    # 设置字体
    setup_matplotlib_fonts()
    
    # 设置matplotlib参数
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 300,  # 高分辨率
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    
    print("✓ MICCAI样式配置完成")

def get_color_palette(n_colors=None):
    """
    获取配色方案
    
    Args:
        n_colors: 需要的颜色数量，如果为None则返回所有颜色
    
    Returns:
        颜色列表
    """
    if n_colors is None:
        return MICCAI_COLORS
    elif n_colors <= len(MICCAI_COLORS):
        return MICCAI_COLORS[:n_colors]
    else:
        # 如果需要更多颜色，使用插值
        return MICCAI_COLORS + [MICCAI_COLORS[i % len(MICCAI_COLORS)] 
                                for i in range(len(MICCAI_COLORS), n_colors)]

def get_color_cycle():
    """获取matplotlib颜色循环"""
    return plt.cycler('color', MICCAI_COLORS)

# ==================== 示例使用 ====================
if __name__ == '__main__':
    # 测试配置
    setup_miccai_style()
    
    # 创建示例图
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.linspace(0, 10, 100)
    colors = get_color_palette(4)
    
    for i, color in enumerate(colors[:4]):
        y = np.sin(x + i * np.pi / 4)
        ax.plot(x, y, color=color, label=f'Series {i+1}', linewidth=2)
    
    ax.set_xlabel('X Label', fontsize=12)
    ax.set_ylabel('Y Label', fontsize=12)
    ax.set_title('MICCAI Color Palette Test', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cursor_file/miccai_color_test.png', dpi=300, bbox_inches='tight')
    print("✓ 测试图已保存: cursor_file/miccai_color_test.png")
    plt.close()

