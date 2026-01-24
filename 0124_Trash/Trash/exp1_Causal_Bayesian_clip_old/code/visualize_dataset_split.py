#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分可视化
展示内外部数据集的分布情况
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List
import matplotlib.patches as mpatches

# 设置字体和样式（优先使用Arial，fallback到DejaVu Sans）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# 字体函数：统一处理字体设置
def get_font_props():
    """获取字体属性，优先Arial，fallback到DejaVu Sans"""
    try:
        from matplotlib import font_manager
        # 尝试查找Arial
        arial_fonts = [f.name for f in font_manager.fontManager.ttflist if 'arial' in f.name.lower()]
        if arial_fonts:
            return {'family': 'Arial'}, 'Arial'
    except:
        pass
    # Fallback到DejaVu Sans（matplotlib默认字体）
    return {'family': 'DejaVu Sans'}, 'DejaVu Sans'

def get_font_family():
    """获取字体族名称"""
    _, family = get_font_props()
    return family


class DatasetSplitVisualizer:
    """数据集划分可视化器"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / 'visualizations'
        self.output_dir.mkdir(exist_ok=True)
        
        # ========== 配色方案配置（蓝粉调）==========
        # 蓝粉调配色：#104e8b、#376b9e、#5f89b1、#afc3d8、#c5e9e3、#d7e1eb、#f2dada、#e5b565、#d89090、#b22222
        self.color_palette = {
            'blue_dark': '#104e8b',      # 深蓝
            'blue_medium': '#376b9e',    # 中蓝
            'blue_light': '#5f89b1',    # 浅蓝
            'blue_pale': '#afc3d8',     # 淡蓝
            'cyan': '#c5e9e3',          # 青色
            'gray_blue': '#d7e1eb',     # 灰蓝
            'pink_pale': '#f2dada',     # 淡粉
            'gold': '#e5b565',          # 金色
            'pink': '#d89090',          # 粉色
            'red': '#b22222'            # 红色
        }
        
        # 中心颜色映射（使用蓝粉调）
        self.center_colors = {
            'Enshi': self.color_palette['blue_dark'],      # 恩施 - 深蓝
            'Xiangyang': self.color_palette['blue_medium'], # 襄阳 - 中蓝
            'Shiyan': self.color_palette['blue_light'],     # 十堰 - 浅蓝
            'Jingzhou': self.color_palette['pink'],        # 荆州 - 粉色
            'Wuda': self.color_palette['red']               # 武大 - 红色
        }
        
        # 数据集类型颜色（使用蓝粉调）
        self.split_colors = {
            'Train': self.color_palette['blue_medium'],
            'Validation': self.color_palette['cyan'],
            'External Test': self.color_palette['red']
        }
        
        # 阳性/阴性颜色
        self.label_colors = {
            'Positive': self.color_palette['red'],
            'Negative': self.color_palette['blue_light']
        }
        
        # 中心名称映射（中文->英文）
        self.center_name_map = {
            '恩施': 'Enshi',
            '襄阳': 'Xiangyang',
            '十堰': 'Shiyan',
            '荆州': 'Jingzhou',
            '武大': 'Wuda'
        }
    
    def load_data(self) -> Dict:
        """加载所有数据"""
        print("📂 加载数据...")
        
        # 加载标签文件
        train_df = pd.read_csv(self.data_dir / 'train_labels.csv')
        val_df = pd.read_csv(self.data_dir / 'val_labels.csv')
        external_df = pd.read_csv(self.data_dir / 'external_test_labels.csv')
        
        # 加载统计信息
        with open(self.data_dir / 'split_statistics.json', 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        # 识别中心（返回英文名称）
        def identify_center(oct_id):
            oct_str = str(oct_id)
            if 'M22105' in oct_str:
                return 'Enshi'
            elif 'M22102' in oct_str:
                return 'Xiangyang'
            elif 'M22104' in oct_str or 'M22101' in oct_str:
                return 'Shiyan'
            elif 'M0008' in oct_str:
                return 'Jingzhou'
            elif 'M20203' in oct_str or 'M20105' in oct_str:
                return 'Wuda'
            return 'Unknown'
        
        train_df['center'] = train_df['OCT'].apply(identify_center)
        val_df['center'] = val_df['OCT'].apply(identify_center)
        external_df['center'] = external_df['OCT'].apply(identify_center)
        
        return {
            'train': train_df,
            'val': val_df,
            'external': external_df,
            'stats': stats
        }
    
    def plot_sample_distribution_by_center(self, data: Dict):
        """绘制各中心样本分布"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        font_props, font_family = get_font_props()
        fig.suptitle('Dataset Distribution by Medical Center', fontsize=16, fontweight='bold', **font_props)
        
        # 1. 各中心总样本数
        ax1 = axes[0, 0]
        centers = ['Enshi', 'Xiangyang', 'Shiyan', 'Jingzhou', 'Wuda']
        train_counts = [
            len(data['train'][data['train']['center'] == c]) for c in centers
        ]
        val_counts = [
            len(data['val'][data['val']['center'] == c]) for c in centers
        ]
        external_counts = [
            len(data['external'][data['external']['center'] == c]) for c in centers
        ]
        
        x = np.arange(len(centers))
        width = 0.25
        
        ax1.bar(x - width, train_counts, width, label='Train', color=self.split_colors['Train'], alpha=0.8)
        ax1.bar(x, val_counts, width, label='Validation', color=self.split_colors['Validation'], alpha=0.8)
        ax1.bar(x + width, external_counts, width, label='External Test', color=self.split_colors['External Test'], alpha=0.8)
        
        ax1.set_xlabel('Medical Center', fontsize=12, **font_props)
        ax1.set_ylabel('Number of Samples', fontsize=12, **font_props)
        ax1.set_title('Sample Distribution by Center', fontsize=14, fontweight='bold', **font_props)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontfamily(font_family)
        ax1.set_xticks(x)
        ax1.set_xticklabels(centers, rotation=0)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, (t, v, e) in enumerate(zip(train_counts, val_counts, external_counts)):
            if t > 0:
                ax1.text(i - width, t + 5, str(t), ha='center', va='bottom', fontsize=9, **font_props)
            if v > 0:
                ax1.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=9, **font_props)
            if e > 0:
                ax1.text(i + width, e + 5, str(e), ha='center', va='bottom', fontsize=9, **font_props)
        ax1.legend(prop={'family': font_family})
        
        # 2. 各中心阳性率
        ax2 = axes[0, 1]
        center_positive_rates = {}
        for center in centers:
            train_center = data['train'][data['train']['center'] == center]
            val_center = data['val'][data['val']['center'] == center]
            external_center = data['external'][data['external']['center'] == center]
            
            all_center = pd.concat([train_center, val_center, external_center], ignore_index=True)
            if len(all_center) > 0:
                center_positive_rates[center] = all_center['label'].mean() * 100
        
        # 使用蓝粉调配色
        colors = [self.center_colors.get(c, self.color_palette['gray_blue']) for c in centers]
        
        colors = [self.center_colors.get(c, '#95A5A6') for c in centers]
        bars = ax2.bar(centers, [center_positive_rates.get(c, 0) for c in centers], color=colors, alpha=0.8)
        ax2.set_xlabel('Medical Center', fontsize=12, **font_props)
        ax2.set_ylabel('Positive Rate (%)', fontsize=12, **font_props)
        ax2.set_title('Positive Rate by Center', fontsize=14, fontweight='bold', **font_props)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontfamily(font_family)
        ax2.set_ylim(0, 110)
        ax2.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, center in zip(bars, centers):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, **font_props)
        
        # 3. 数据集划分饼图
        ax3 = axes[1, 0]
        sizes = [
            len(data['train']),
            len(data['val']),
            len(data['external'])
        ]
        labels = ['Train\n(Internal)', 'Validation\n(Internal)', 'External Test']
        colors_pie = [
            self.split_colors['Train'],
            self.split_colors['Validation'],
            self.split_colors['External Test']
        ]
        # 设置标签字体
        label_props = {'fontsize': 11, 'fontweight': 'bold', 'fontfamily': 'Arial'}
        explode = (0.05, 0.05, 0.1)
        
        wedges, texts, autotexts = ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                          autopct='%1.1f%%', shadow=True, startangle=90,
                                          textprops=label_props)
        for text in texts + autotexts:
            text.set_fontfamily(font_family)
        
        ax3.set_title('Dataset Split Distribution', fontsize=14, fontweight='bold', **font_props)
        for text in ax3.texts:
            text.set_fontfamily(font_family)
        
        # 4. 阳性/阴性样本分布
        ax4 = axes[1, 1]
        splits = ['Train', 'Validation', 'External Test']
        positive_counts = [
            data['train']['label'].sum(),
            data['val']['label'].sum(),
            data['external']['label'].sum()
        ]
        negative_counts = [
            (data['train']['label'] == 0).sum(),
            (data['val']['label'] == 0).sum(),
            (data['external']['label'] == 0).sum()
        ]
        
        x = np.arange(len(splits))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, positive_counts, width, label='Positive', 
                       color=self.label_colors['Positive'], alpha=0.8)
        bars2 = ax4.bar(x + width/2, negative_counts, width, label='Negative', 
                       color=self.label_colors['Negative'], alpha=0.8)
        
        ax4.set_xlabel('Dataset Split', fontsize=12, **font_props)
        ax4.set_ylabel('Number of Samples', fontsize=12, **font_props)
        ax4.set_title('Positive vs Negative Distribution', fontsize=14, fontweight='bold', **font_props)
        for label in ax4.get_xticklabels() + ax4.get_yticklabels():
            label.set_fontfamily(font_family)
        ax4.legend(prop={'family': font_family})
        ax4.set_xticks(x)
        ax4.set_xticklabels(splits)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9, **font_props)
        
        plt.tight_layout()
        # 保存PNG和PDF两种格式
        output_file_png = self.output_dir / 'dataset_distribution_by_center.png'
        output_file_pdf = self.output_dir / 'dataset_distribution_by_center.pdf'
        plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight', format='pdf')
        print(f"✅ 已保存: {output_file_png}")
        print(f"✅ 已保存: {output_file_pdf}")
        plt.close()
    
    def plot_center_comparison(self, data: Dict):
        """绘制中心对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        font_props, font_family = get_font_props()
        fig.suptitle('Medical Center Comparison', fontsize=16, fontweight='bold', **font_props)
        
        centers = ['Enshi', 'Xiangyang', 'Shiyan', 'Jingzhou', 'Wuda']
        
        # 1. 样本总数对比
        ax1 = axes[0]
        total_counts = []
        for center in centers:
            train_count = len(data['train'][data['train']['center'] == center])
            val_count = len(data['val'][data['val']['center'] == center])
            external_count = len(data['external'][data['external']['center'] == center])
            total_counts.append(train_count + val_count + external_count)
        
        colors = [self.center_colors.get(c, self.color_palette['gray_blue']) for c in centers]
        bars = ax1.bar(centers, total_counts, color=colors, alpha=0.8)
        ax1.set_xlabel('Medical Center', fontsize=12, **font_props)
        ax1.set_ylabel('Total Samples', fontsize=12, **font_props)
        ax1.set_title('Total Sample Count by Center', fontsize=14, fontweight='bold', **font_props)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontfamily(font_family)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, total_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold', **font_props)
        
        # 2. 内部 vs 外部
        ax2 = axes[1]
        internal_centers = ['Enshi', 'Xiangyang', 'Shiyan']
        external_centers = ['Jingzhou', 'Wuda']
        
        internal_counts = [total_counts[centers.index(c)] for c in internal_centers]
        external_counts = [total_counts[centers.index(c)] for c in external_centers]
        
        x_internal = np.arange(len(internal_centers))
        x_external = np.arange(len(external_centers)) + len(internal_centers) + 0.5
        
        ax2.bar(x_internal, internal_counts, color=self.color_palette['blue_medium'], alpha=0.8, label='Internal (Train/Val)')
        ax2.bar(x_external, external_counts, color=self.color_palette['red'], alpha=0.8, label='External (Test)')
        
        ax2.set_xlabel('Medical Center', fontsize=12, **font_props)
        ax2.set_ylabel('Total Samples', fontsize=12, **font_props)
        ax2.set_title('Internal vs External Centers', fontsize=14, fontweight='bold', **font_props)
        ax2.legend(prop={'family': font_family})
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontfamily(font_family)
        ax2.set_xticks(list(x_internal) + list(x_external))
        ax2.set_xticklabels(internal_centers + external_centers)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. 阳性率对比
        ax3 = axes[2]
        positive_rates = []
        for center in centers:
            train_center = data['train'][data['train']['center'] == center]
            val_center = data['val'][data['val']['center'] == center]
            external_center = data['external'][data['external']['center'] == center]
            
            all_center = pd.concat([train_center, val_center, external_center], ignore_index=True)
            if len(all_center) > 0:
                positive_rates.append(all_center['label'].mean() * 100)
            else:
                positive_rates.append(0)
        
        colors = [self.center_colors.get(c, self.color_palette['gray_blue']) for c in centers]
        bars = ax3.bar(centers, positive_rates, color=colors, alpha=0.8)
        ax3.set_xlabel('Medical Center', fontsize=12, **font_props)
        ax3.set_ylabel('Positive Rate (%)', fontsize=12, **font_props)
        ax3.set_title('Positive Rate by Center', fontsize=14, fontweight='bold', **font_props)
        ax3.legend(prop={'family': font_family})
        for label in ax3.get_xticklabels() + ax3.get_yticklabels():
            label.set_fontfamily(font_family)
        ax3.set_ylim(0, 110)
        ax3.grid(axis='y', alpha=0.3)
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Baseline')
        
        for bar, rate in zip(bars, positive_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', **font_props)
        
        plt.tight_layout()
        # 保存PNG和PDF两种格式
        output_file_png = self.output_dir / 'center_comparison.png'
        output_file_pdf = self.output_dir / 'center_comparison.pdf'
        plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight', format='pdf')
        print(f"✅ 已保存: {output_file_png}")
        print(f"✅ 已保存: {output_file_pdf}")
        plt.close()
    
    def plot_detailed_statistics(self, data: Dict):
        """绘制详细统计信息"""
        fig = plt.figure(figsize=(22, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4, left=0.08, right=0.95, top=0.95, bottom=0.1)
        font_props, font_family = get_font_props()
        
        fig.suptitle('Detailed Dataset Statistics', fontsize=16, fontweight='bold', y=0.98, **font_props)
        
        centers = ['Enshi', 'Xiangyang', 'Shiyan', 'Jingzhou', 'Wuda']
        
        # 1. 各数据集样本数（堆叠柱状图）
        ax1 = fig.add_subplot(gs[0, 0])
        train_counts = [len(data['train'][data['train']['center'] == c]) for c in centers]
        val_counts = [len(data['val'][data['val']['center'] == c]) for c in centers]
        external_counts = [len(data['external'][data['external']['center'] == c]) for c in centers]
        
        x = np.arange(len(centers))
        ax1.bar(x, train_counts, label='Train', color=self.split_colors['Train'], alpha=0.8)
        ax1.bar(x, val_counts, bottom=train_counts, label='Validation', color=self.split_colors['Validation'], alpha=0.8)
        ax1.bar(x, external_counts, bottom=np.array(train_counts) + np.array(val_counts), 
               label='External Test', color=self.split_colors['External Test'], alpha=0.8)
        
        ax1.set_xlabel('Medical Center', fontsize=10, **font_props)
        ax1.set_ylabel('Number of Samples', fontsize=10, **font_props)
        ax1.set_title('Stacked Sample Distribution', fontsize=11, fontweight='bold', **font_props)
        ax1.set_xticks(x)
        ax1.set_xticklabels(centers, rotation=15, ha='right', fontsize=9)
        ax1.legend(fontsize=8, prop={'family': font_family}, loc='upper left')
        for label in ax1.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(9)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 各数据集阳性率
        ax2 = fig.add_subplot(gs[0, 1])
        train_rates = [data['train'][data['train']['center'] == c]['label'].mean() * 100 
                      if len(data['train'][data['train']['center'] == c]) > 0 else 0 for c in centers]
        val_rates = [data['val'][data['val']['center'] == c]['label'].mean() * 100 
                    if len(data['val'][data['val']['center'] == c]) > 0 else 0 for c in centers]
        external_rates = [data['external'][data['external']['center'] == c]['label'].mean() * 100 
                         if len(data['external'][data['external']['center'] == c]) > 0 else 0 for c in centers]
        
        x = np.arange(len(centers))
        width = 0.25
        ax2.bar(x - width, train_rates, width, label='Train', color=self.split_colors['Train'], alpha=0.8)
        ax2.bar(x, val_rates, width, label='Validation', color=self.split_colors['Validation'], alpha=0.8)
        ax2.bar(x + width, external_rates, width, label='External Test', color=self.split_colors['External Test'], alpha=0.8)
        
        ax2.set_xlabel('Medical Center', fontsize=10, **font_props)
        ax2.set_ylabel('Positive Rate (%)', fontsize=10, **font_props)
        ax2.set_title('Positive Rate by Dataset Split', fontsize=11, fontweight='bold', **font_props)
        ax2.set_xticks(x)
        ax2.set_xticklabels(centers, rotation=15, ha='right', fontsize=9)
        ax2.legend(fontsize=8, prop={'family': font_family}, loc='upper left')
        for label in ax2.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(9)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 110)
        
        # 3. 数据集划分饼图（带详细标签）
        ax3 = fig.add_subplot(gs[0, 2])
        sizes = [len(data['train']), len(data['val']), len(data['external'])]
        labels = [f'Train\n{len(data["train"])} samples', 
                 f'Validation\n{len(data["val"])} samples',
                 f'External Test\n{len(data["external"])} samples']
        colors_pie = [self.split_colors['Train'], self.split_colors['Validation'], self.split_colors['External Test']]
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie,
                                          autopct='%1.1f%%', shadow=True, startangle=90,
                                          textprops={'fontsize': 9, 'fontfamily': font_family},
                                          pctdistance=0.85, labeldistance=1.1)
        ax3.set_title('Dataset Split Overview', fontsize=11, fontweight='bold', **font_props)
        for text in texts:
            text.set_fontfamily(font_family)
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontfamily(font_family)
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        
        # 4. 各中心阳性/阴性分布
        ax4 = fig.add_subplot(gs[1, :])
        centers_internal = ['Enshi', 'Xiangyang', 'Shiyan']
        centers_external = ['Jingzhou', 'Wuda']
        
        # 内部中心
        internal_positive = []
        internal_negative = []
        for center in centers_internal:
            all_center = pd.concat([
                data['train'][data['train']['center'] == center],
                data['val'][data['val']['center'] == center]
            ], ignore_index=True)
            if len(all_center) > 0:
                internal_positive.append(all_center['label'].sum())
                internal_negative.append((all_center['label'] == 0).sum())
            else:
                internal_positive.append(0)
                internal_negative.append(0)
        
        # 外部中心
        external_positive = []
        external_negative = []
        for center in centers_external:
            all_center = data['external'][data['external']['center'] == center]
            if len(all_center) > 0:
                external_positive.append(all_center['label'].sum())
                external_negative.append((all_center['label'] == 0).sum())
            else:
                external_positive.append(0)
                external_negative.append(0)
        
        x_internal = np.arange(len(centers_internal))
        x_external = np.arange(len(centers_external)) + len(centers_internal) + 1
        
        width = 0.35
        ax4.bar(x_internal - width/2, internal_positive, width, label='Positive (Internal)', 
               color=self.label_colors['Positive'], alpha=0.8)
        ax4.bar(x_internal + width/2, internal_negative, width, label='Negative (Internal)', 
               color=self.label_colors['Negative'], alpha=0.8)
        ax4.bar(x_external - width/2, external_positive, width, label='Positive (External)', 
               color=self.color_palette['red'], alpha=0.8)
        ax4.bar(x_external + width/2, external_negative, width, label='Negative (External)', 
               color=self.color_palette['blue_pale'], alpha=0.8)
        
        ax4.set_xlabel('Medical Center', fontsize=11, **font_props)
        ax4.set_ylabel('Number of Samples', fontsize=11, **font_props)
        ax4.set_title('Positive vs Negative Distribution by Center', fontsize=12, fontweight='bold', **font_props)
        ax4.set_xticks(list(x_internal) + list(x_external))
        ax4.set_xticklabels(centers_internal + centers_external, fontsize=10)
        ax4.legend(fontsize=9, prop={'family': font_family}, ncol=2, loc='upper right')
        for label in ax4.get_xticklabels() + ax4.get_yticklabels():
            label.set_fontfamily(font_family)
        ax4.grid(axis='y', alpha=0.3)
        
        # 添加分隔线
        ax4.axvline(x=len(centers_internal) - 0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
        ax4.text(len(centers_internal) - 0.5, ax4.get_ylim()[1] * 0.95, 'Internal | External', 
                ha='center', va='top', fontsize=10, fontweight='bold', **font_props,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. 统计表格
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        table_data = []
        for center in centers:
            train_center = data['train'][data['train']['center'] == center]
            val_center = data['val'][data['val']['center'] == center]
            external_center = data['external'][data['external']['center'] == center]
            
            all_center = pd.concat([train_center, val_center, external_center], ignore_index=True)
            
            if len(all_center) > 0:
                total = len(all_center)
                positive = int(all_center['label'].sum())
                negative = total - positive
                positive_rate = all_center['label'].mean() * 100
                
                split_type = 'Internal' if center in ['Enshi', 'Xiangyang', 'Shiyan'] else 'External'
                
                table_data.append([
                    center,
                    split_type,
                    total,
                    positive,
                    negative,
                    f'{positive_rate:.2f}%',
                    len(train_center),
                    len(val_center),
                    len(external_center)
                ])
        
        columns = ['Center', 'Type', 'Total', 'Positive', 'Negative', 'Positive Rate', 
                  'Train', 'Val', 'External Test']
        table = ax5.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center',
                         colWidths=[0.10, 0.10, 0.08, 0.08, 0.08, 0.12, 0.08, 0.08, 0.12])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2.5)
        
        # 设置表头样式
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#34495E')
            table[(0, i)].set_text_props(weight='bold', color='white', family=font_family, size=9)
        
        # 设置内部/外部行的颜色
        for i, row in enumerate(table_data, 1):
            if row[1] == 'Internal':
                for j in range(len(columns)):
                    table[(i, j)].set_facecolor('#EBF5FB')
                    table[(i, j)].set_text_props(family=font_family, size=8)
            else:
                for j in range(len(columns)):
                    table[(i, j)].set_facecolor('#FADBD8')
                    table[(i, j)].set_text_props(family=font_family, size=8)
        
        ax5.set_title('Detailed Statistics Table', fontsize=12, fontweight='bold', pad=15, **font_props)
        
        # 保存PNG和PDF两种格式
        output_file_png = self.output_dir / 'detailed_statistics.png'
        output_file_pdf = self.output_dir / 'detailed_statistics.pdf'
        plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight', format='pdf')
        print(f"✅ 已保存: {output_file_png}")
        print(f"✅ 已保存: {output_file_pdf}")
        plt.close()
    
    def create_summary_report(self, data: Dict):
        """创建文本摘要报告"""
        report_file = self.output_dir / 'visualization_summary.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DATASET SPLIT VISUALIZATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("INTERNAL DEVELOPMENT SET (Train/Validation)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Samples: {len(data['train']) + len(data['val'])}\n")
            f.write(f"  - Train: {len(data['train'])} samples\n")
            f.write(f"  - Validation: {len(data['val'])} samples\n")
            f.write(f"Centers: Enshi, Xiangyang, Shiyan\n\n")
            
            f.write("EXTERNAL TEST SET\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Samples: {len(data['external'])}\n")
            f.write(f"Centers: Jingzhou, Wuda\n")
            f.write(f"⚠️  STRICTLY FORBIDDEN FROM TRAINING PROCESS\n\n")
            
            f.write("CENTER DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            centers = ['Enshi', 'Xiangyang', 'Shiyan', 'Jingzhou', 'Wuda']
            for center in centers:
                train_center = data['train'][data['train']['center'] == center]
                val_center = data['val'][data['val']['center'] == center]
                external_center = data['external'][data['external']['center'] == center]
                
                all_center = pd.concat([train_center, val_center, external_center], ignore_index=True)
                if len(all_center) > 0:
                    total = len(all_center)
                    positive = int(all_center['label'].sum())
                    positive_rate = all_center['label'].mean() * 100
                    split_type = 'Internal' if center in ['Enshi', 'Xiangyang', 'Shiyan'] else 'External'
                    
                    f.write(f"{center} ({split_type}): {total} samples, {positive} positive ({positive_rate:.2f}%)\n")
                    f.write(f"  - Train: {len(train_center)}, Val: {len(val_center)}, External: {len(external_center)}\n")
        
        print(f"✅ 已保存: {report_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据集划分可视化')
    parser.add_argument('--data_dir', type=str,
                       default='/data2/hmy/5Center_datas/5centers_multi_internal_external_final',
                       help='数据目录')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 数据集划分可视化")
    print("=" * 80)
    print(f"数据目录: {args.data_dir}")
    print("=" * 80)
    
    visualizer = DatasetSplitVisualizer(Path(args.data_dir))
    
    # 加载数据
    data = visualizer.load_data()
    
    # 生成可视化
    print("\n🎨 生成可视化图表...")
    visualizer.plot_sample_distribution_by_center(data)
    visualizer.plot_center_comparison(data)
    visualizer.plot_detailed_statistics(data)
    visualizer.create_summary_report(data)
    
    print("\n" + "=" * 80)
    print("✅ 可视化完成！")
    print("=" * 80)
    print(f"\n📁 输出目录: {visualizer.output_dir}")
    print("  - dataset_distribution_by_center.png / .pdf")
    print("  - center_comparison.png / .pdf")
    print("  - detailed_statistics.png / .pdf")
    print("  - visualization_summary.txt")
    print("\n💡 PDF格式适合论文使用，矢量图，高清无损！")


if __name__ == '__main__':
    main()

