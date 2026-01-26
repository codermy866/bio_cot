#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nature/Science Style Color Palette
Nature/Science风格的高级配色方案 - 低饱和度
"""

# 统一配色方案：D69584 到 C7CCD6 色阶
# D69584: 红棕色 (起点)
# C7CCD6: 蓝灰色 (终点)
# 创建从红棕到蓝灰的渐变色阶

NATURE_COLORS = {
    # 主色调（基于D69584到C7CCD6色阶）
    'positive': '#D69584',      # 红棕色（起点色）
    'negative': '#C7CCD6',       # 蓝灰色（终点色）
    
    # 5个中心的配色（从D69584到C7CCD6的均匀分布）
    'center_0': '#D69584',      # 红棕色
    'center_1': '#D2A392',      # 浅红棕灰
    'center_2': '#CEB1A0',      # 中红棕灰
    'center_3': '#CABFAE',      # 浅蓝灰棕
    'center_4': '#C7CCD6',      # 蓝灰色
    
    # 辅助色（色阶中间过渡色）
    'accent_1': '#D49A8A',      # 红棕偏红
    'accent_2': '#D0A896',      # 红棕偏灰
    'accent_3': '#CCB6A4',      # 灰棕偏蓝
    'accent_4': '#C9C4D1',      # 蓝灰偏紫
    'accent_5': '#C7CCD6',      # 蓝灰色
    
    # 背景和网格（保持干净）
    'background': '#FAFAFA',    # 浅灰背景
    'grid': '#E0E0E0',          # 浅灰网格
    
    # 热图配色（使用D69584到C7CCD6的色阶）
    'heatmap_cmap': 'custom_d69584_c7ccd6',  # 自定义色阶
    'heatmap_center': 0,         # 中心值
    
    # 散点图配色
    'scatter_alpha': 0.7,        # 提高透明度，使颜色更明显
    'scatter_edge': '#FFFFFF',    # 白色边缘
    
    # 线图配色
    'line_width': 2.5,
    'line_alpha': 0.9,            # 提高透明度，使线条更明显
}

# 创建从D69584到C7CCD6的自定义colormap
def create_custom_colormap():
    """创建从D69584到C7CCD6的自定义colormap"""
    from matplotlib.colors import LinearSegmentedColormap
    # 从红棕(D69584)到蓝灰(C7CCD6)的渐变
    colors_list = ['#D69584', '#D2A392', '#CEB1A0', '#CABFAE', '#C7CCD6']
    n_bins = 100
    custom_cmap = LinearSegmentedColormap.from_list('custom_d69584_c7ccd6', colors_list, N=n_bins)
    # 尝试注册colormap（兼容不同版本的matplotlib）
    try:
        import matplotlib
        if hasattr(matplotlib, 'colormaps'):
            matplotlib.colormaps.register(custom_cmap, name='custom_d69584_c7ccd6')
        elif hasattr(matplotlib.cm, 'register_cmap'):
            matplotlib.cm.register_cmap(name='custom_d69584_c7ccd6', cmap=custom_cmap)
    except:
        pass  # 如果注册失败，直接返回colormap对象
    return custom_cmap

# 全局字体设置
FONT_FAMILY = 'Calibri'

# Feature名称映射（英文，有具体含义）
FEATURE_NAMES = {
    'feature_0': 'OCT Texture',
    'feature_1': 'OCT Intensity',
    'feature_2': 'OCT Morphology',
    'feature_3': 'Colposcopy Texture',
    'feature_4': 'Colposcopy Color',
    'feature_5': 'Colposcopy Vascular',
    'feature_6': 'Clinical Age',
    'feature_7': 'Clinical HPV',
    'feature_8': 'Clinical TCT',
    'feature_9': 'Fused Feature',
}

# 中心名称（英文）
CENTER_NAMES_EN = {
    0: 'Enshi-1 (M20105)',
    1: 'Small Center (M20203)',
    2: 'Xiangyang (M22102)',
    3: 'Enshi/Wuda (M22105)',
    4: 'Shiyan/Jingzhou (External)'
}

# 辅助函数：获取feature显示名称
def get_feature_display_name(feature_key):
    """获取feature的显示名称"""
    if feature_key in FEATURE_NAMES:
        return FEATURE_NAMES[feature_key]
    return feature_key.replace('_', ' ').title()

