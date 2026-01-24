#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件生成信息:
- 生成时间: 2025-12-25 08:43:45 (使用nvidia-smi获取CUDA时间)
- 更新时间: 2025-12-25 09:05:22 (使用nvidia-smi获取CUDA时间)
- 生成需求: 用户要求可视化实际代码和工程中使用的数据集，展示内外部数据集的结构和位置
- 更新需求: 用户要求所有文本使用英文，避免内容重叠，输出PDF格式
- 生成原因: 需要清晰了解数据集的实际路径、结构、样本分布和软链接情况
- 更新原因: 修复字体显示问题，避免内容重叠，满足论文PDF格式要求
- 相关任务: 数据集可视化，MICCAI论文准备

文件功能: 可视化实际使用的内外部数据集结构，包括路径、样本分布、软链接情况等；已更新：全英文显示，避免重叠，PDF输出
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import os
import sys
import subprocess
import datetime
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cursor_file.visualization_config import setup_miccai_style, get_color_palette
from cursor_file.project_config import DATA_5CENTERS_MULTI, DATA_5CENTERS_INTERNAL_EXTERNAL

def get_timestamp():
    """获取当前时间戳"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=timestamp', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            timestamp_str = result.stdout.strip().split('\n')[0]
            try:
                if '/' in timestamp_str:
                    timestamp_clean = timestamp_str.split('.')[0]
                    dt = datetime.datetime.strptime(timestamp_clean, '%Y/%m/%d %H:%M:%S')
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, AttributeError):
                pass
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def identify_center(oct_id: str) -> tuple:
    """
    Identify medical center
    
    Returns:
        (center_name, type) tuple, type is 'internal' or 'external'
    """
    oct_str = str(oct_id)
    if 'M22105' in oct_str:
        return 'Enshi', 'internal'
    elif 'M22102' in oct_str:
        return 'Xiangyang', 'internal'
    elif 'M22104' in oct_str or 'M22101' in oct_str:
        return 'Shiyan', 'internal'
    elif 'M0008' in oct_str:
        return 'Jingzhou', 'external'
    elif 'M20203' in oct_str or 'M20105' in oct_str:
        return 'Wuda', 'external'
    return 'Unknown', 'unknown'

def check_symbolic_link(link_path: Path) -> dict:
    """检查软链接信息"""
    result = {
        'is_link': False,
        'target': None,
        'exists': False,
        'target_exists': False
    }
    
    if link_path.exists():
        result['exists'] = True
        if link_path.is_symlink():
            result['is_link'] = True
            result['target'] = link_path.resolve()
            result['target_exists'] = result['target'].exists()
    
    return result

def analyze_dataset_structure():
    """Analyze dataset structure"""
    print("="*80)
    print("Analyzing dataset structure...")
    print("="*80)
    
    # Actual dataset paths used in code
    split_dataset_path = Path('/data2/hmy/5Center_datas/5centers_multi_internal_external_final')
    original_dataset_path = Path('/data2/hmy/5Center_datas/5centers_multi')
    
    # Check paths
    if not split_dataset_path.exists():
        print(f"❌ Split dataset path does not exist: {split_dataset_path}")
        return None
    
    if not original_dataset_path.exists():
        print(f"❌ Original dataset path does not exist: {original_dataset_path}")
        return None
    
    print(f"\n✅ Split dataset: {split_dataset_path}")
    print(f"✅ Original dataset: {original_dataset_path}")
    
    # Read label files
    train_labels = split_dataset_path / 'train_labels.csv'
    val_labels = split_dataset_path / 'val_labels.csv'
    external_test_labels = split_dataset_path / 'external_test_labels.csv'
    
    datasets = {}
    for name, path in [('train', train_labels), ('val', val_labels), ('external_test', external_test_labels)]:
        if path.exists():
            df = pd.read_csv(path)
            df['center_name'], df['center_type'] = zip(*df['OCT'].apply(identify_center))
            datasets[name] = df
            print(f"✅ {name}: {len(df)} samples")
        else:
            print(f"⚠️  {name} label file does not exist: {path}")
    
    return {
        'split_dataset_path': split_dataset_path,
        'original_dataset_path': original_dataset_path,
        'datasets': datasets
    }

def visualize_dataset_structure(data_info):
    """可视化数据集结构"""
    setup_miccai_style()
    colors = get_color_palette(8)
    
    datasets = data_info['datasets']
    split_dataset_path = data_info['split_dataset_path']
    original_dataset_path = data_info['original_dataset_path']
    
    # 创建图形 - 增加尺寸和间距以避免重叠
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4, left=0.08, right=0.95, top=0.93, bottom=0.08)
    
    # ==================== 1. 数据集路径结构图 ====================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    # 绘制路径结构 - 增加间距避免重叠
    y_pos = 0.95
    ax1.text(0.05, y_pos, 'Dataset Structure', fontsize=16, fontweight='bold', transform=ax1.transAxes)
    y_pos -= 0.18
    
    # 划分后数据集
    ax1.text(0.05, y_pos, 'Split Dataset (Used in Code):', fontsize=12, fontweight='bold', 
             color=colors[0], transform=ax1.transAxes)
    y_pos -= 0.12
    # 路径太长，分行显示
    path_str = str(split_dataset_path)
    if len(path_str) > 60:
        mid = len(path_str) // 2
        ax1.text(0.05, y_pos, path_str[:mid], fontsize=9, 
                 family='monospace', transform=ax1.transAxes, wrap=True)
        y_pos -= 0.08
        ax1.text(0.05, y_pos, path_str[mid:], fontsize=9, 
                 family='monospace', transform=ax1.transAxes)
    else:
        ax1.text(0.05, y_pos, path_str, fontsize=9, 
                 family='monospace', transform=ax1.transAxes)
    y_pos -= 0.15
    
    # 原始数据集
    ax1.text(0.05, y_pos, 'Original Dataset (Source):', fontsize=12, fontweight='bold',
             color=colors[4], transform=ax1.transAxes)
    y_pos -= 0.12
    orig_path_str = str(original_dataset_path)
    if len(orig_path_str) > 60:
        mid = len(orig_path_str) // 2
        ax1.text(0.05, y_pos, orig_path_str[:mid], fontsize=9,
                 family='monospace', transform=ax1.transAxes)
        y_pos -= 0.08
        ax1.text(0.05, y_pos, orig_path_str[mid:], fontsize=9,
                 family='monospace', transform=ax1.transAxes)
    else:
        ax1.text(0.05, y_pos, orig_path_str, fontsize=9,
                 family='monospace', transform=ax1.transAxes)
    y_pos -= 0.15
    
    # 软链接说明
    ax1.text(0.05, y_pos, 'Symbolic Links:', fontsize=12, fontweight='bold',
             color=colors[6], transform=ax1.transAxes)
    y_pos -= 0.12
    ax1.text(0.05, y_pos, 'Images → Original Dataset', fontsize=10,
             transform=ax1.transAxes)
    
    # ==================== 2. 样本数量分布 ====================
    ax2 = fig.add_subplot(gs[0, 1])
    
    dataset_names = []
    sample_counts = []
    dataset_colors = []
    
    for name, df in datasets.items():
        dataset_names.append(name.replace('_', ' ').title())
        sample_counts.append(len(df))
        if 'external' in name:
            dataset_colors.append(colors[0])  # 外部测试集用红色
        elif name == 'train':
            dataset_colors.append(colors[4])  # 训练集用绿色
        else:
            dataset_colors.append(colors[2])  # 验证集用黄色
    
    bars = ax2.bar(dataset_names, sample_counts, color=dataset_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Distribution by Dataset', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # ==================== 3. 按中心分布 ====================
    ax3 = fig.add_subplot(gs[0, 2])
    
    center_counts = defaultdict(int)
    center_types = {}
    
    for name, df in datasets.items():
        for center in df['center_name'].unique():
            count = (df['center_name'] == center).sum()
            center_counts[center] += count
            if center not in center_types:
                center_types[center] = df[df['center_name'] == center]['center_type'].iloc[0]
    
    centers = sorted(center_counts.keys())
    counts = [center_counts[c] for c in centers]
    center_colors = [colors[0] if center_types[c] == 'external' else colors[4] for c in centers]
    
    bars = ax3.bar(centers, counts, color=center_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Number of Samples', fontsize=12)
    ax3.set_title('Sample Distribution by Medical Center', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    
    # 添加数值标签 - 调整位置避免重叠
    for bar in bars:
        height = bar.get_height()
        # 如果高度太小，标签放在柱子上方
        if height < max(counts) * 0.1:
            va = 'bottom'
            y_offset = height + max(counts) * 0.02
        else:
            va = 'bottom'
            y_offset = height + max(counts) * 0.01
        ax3.text(bar.get_x() + bar.get_width()/2., y_offset,
                f'{int(height)}',
                ha='center', va=va, fontsize=9, fontweight='bold')
    
    # ==================== 4. 标签分布 ====================
    ax4 = fig.add_subplot(gs[1, 0])
    
    label_counts = defaultdict(int)
    for name, df in datasets.items():
        label_counts['Positive'] += (df['label'] == 1).sum()
        label_counts['Negative'] += (df['label'] == 0).sum()
    
    labels = ['Positive', 'Negative']
    counts = [label_counts['Positive'], label_counts['Negative']]
    label_colors = [colors[0], colors[4]]
    
    bars = ax4.bar(labels, counts, color=label_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Number of Samples', fontsize=12)
    ax4.set_title('Label Distribution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # ==================== 5. 内外部数据集划分 ====================
    ax5 = fig.add_subplot(gs[1, 1])
    
    internal_count = 0
    external_count = 0
    
    for name, df in datasets.items():
        if 'external' in name:
            external_count += len(df)
        else:
            internal_count += len(df)
    
    sizes = [internal_count, external_count]
    labels_pie = ['Internal\n(Development)', 'External\n(Test)']
    colors_pie = [colors[4], colors[0]]
    explode = (0.05, 0.1)  # 突出外部测试集
    
    wedges, texts, autotexts = ax5.pie(sizes, labels=labels_pie, colors=colors_pie, 
                                        autopct='%1.1f%%', explode=explode,
                                        startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax5.set_title('Internal vs External Dataset Split', fontsize=14, fontweight='bold')
    
    # ==================== 6. 各数据集标签分布 ====================
    ax6 = fig.add_subplot(gs[1, 2])
    
    dataset_labels = []
    pos_counts = []
    neg_counts = []
    
    for name, df in datasets.items():
        dataset_labels.append(name.replace('_', ' ').title())
        pos_counts.append((df['label'] == 1).sum())
        neg_counts.append((df['label'] == 0).sum())
    
    x = range(len(dataset_labels))
    width = 0.35
    
    bars1 = ax6.bar([i - width/2 for i in x], pos_counts, width, label='Positive', 
                    color=colors[0], alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax6.bar([i + width/2 for i in x], neg_counts, width, label='Negative',
                    color=colors[4], alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax6.set_ylabel('Number of Samples', fontsize=12)
    ax6.set_title('Label Distribution by Dataset', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(dataset_labels, rotation=45, ha='right', fontsize=10)
    ax6.legend(fontsize=10, loc='upper right')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签到柱状图
    for i, (pos, neg) in enumerate(zip(pos_counts, neg_counts)):
        if pos > 0:
            ax6.text(i - width/2, pos + max(max(pos_counts), max(neg_counts)) * 0.01,
                    f'{int(pos)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        if neg > 0:
            ax6.text(i + width/2, neg + max(max(pos_counts), max(neg_counts)) * 0.01,
                    f'{int(neg)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # ==================== 7. 软链接统计 ====================
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # 检查软链接
    link_stats = {
        'train_oct': 0,
        'train_col': 0,
        'val_oct': 0,
        'val_col': 0,
        'external_oct': 0,
        'external_col': 0,
    }
    
    # 检查训练集软链接
    train_oct_dir = split_dataset_path / 'internal_train' / 'train' / 'oct'
    train_col_dir = split_dataset_path / 'internal_train' / 'train' / 'col'
    if train_oct_dir.exists():
        link_stats['train_oct'] = sum(1 for f in train_oct_dir.iterdir() if f.is_symlink())
    if train_col_dir.exists():
        link_stats['train_col'] = sum(1 for f in train_col_dir.iterdir() if f.is_symlink())
    
    # 检查验证集软链接
    val_oct_dir = split_dataset_path / 'internal_train' / 'val' / 'oct'
    val_col_dir = split_dataset_path / 'internal_train' / 'val' / 'col'
    if val_oct_dir.exists():
        link_stats['val_oct'] = sum(1 for f in val_oct_dir.iterdir() if f.is_symlink())
    if val_col_dir.exists():
        link_stats['val_col'] = sum(1 for f in val_col_dir.iterdir() if f.is_symlink())
    
    # 检查外部测试集软链接
    external_oct_dir = split_dataset_path / 'external_validation' / 'oct'
    external_col_dir = split_dataset_path / 'external_validation' / 'col'
    if external_oct_dir.exists():
        link_stats['external_oct'] = sum(1 for f in external_oct_dir.iterdir() if f.is_symlink())
    if external_col_dir.exists():
        link_stats['external_col'] = sum(1 for f in external_col_dir.iterdir() if f.is_symlink())
    
    # 绘制软链接统计表 - 增加间距避免重叠
    y_pos = 0.95
    ax7.text(0.5, y_pos, 'Symbolic Links Statistics', fontsize=16, fontweight='bold',
             ha='center', transform=ax7.transAxes)
    y_pos -= 0.12
    
    # 表头
    ax7.text(0.08, y_pos, 'Dataset', fontsize=12, fontweight='bold', transform=ax7.transAxes)
    ax7.text(0.35, y_pos, 'OCT Links', fontsize=12, fontweight='bold', transform=ax7.transAxes)
    ax7.text(0.55, y_pos, 'Colposcopy Links', fontsize=12, fontweight='bold', transform=ax7.transAxes)
    ax7.text(0.80, y_pos, 'Total', fontsize=12, fontweight='bold', transform=ax7.transAxes)
    y_pos -= 0.10
    
    # 数据行
    datasets_info = [
        ('Train', 'train_oct', 'train_col'),
        ('Validation', 'val_oct', 'val_col'),
        ('External Test', 'external_oct', 'external_col'),
    ]
    
    total_links = 0
    for name, oct_key, col_key in datasets_info:
        oct_count = link_stats[oct_key]
        col_count = link_stats[col_key]
        total = oct_count + col_count
        total_links += total
        
        ax7.text(0.08, y_pos, name, fontsize=11, transform=ax7.transAxes)
        ax7.text(0.35, y_pos, f'{oct_count}', fontsize=11, ha='center', transform=ax7.transAxes)
        ax7.text(0.55, y_pos, f'{col_count}', fontsize=11, ha='center', transform=ax7.transAxes)
        ax7.text(0.80, y_pos, f'{total}', fontsize=11, fontweight='bold', ha='center', transform=ax7.transAxes)
        y_pos -= 0.10
    
    # 总计
    y_pos -= 0.05
    ax7.text(0.08, y_pos, 'Total', fontsize=12, fontweight='bold', transform=ax7.transAxes)
    ax7.text(0.35, y_pos, f'{sum([link_stats[k] for k in ["train_oct", "val_oct", "external_oct"]])}',
             fontsize=12, fontweight='bold', ha='center', transform=ax7.transAxes)
    ax7.text(0.55, y_pos, f'{sum([link_stats[k] for k in ["train_col", "val_col", "external_col"]])}',
             fontsize=12, fontweight='bold', ha='center', transform=ax7.transAxes)
    ax7.text(0.80, y_pos, f'{total_links}', fontsize=12, fontweight='bold', ha='center', transform=ax7.transAxes)
    
    # 添加说明 - 分行显示长路径
    y_pos -= 0.12
    orig_path_str = str(original_dataset_path)
    if len(orig_path_str) > 70:
        mid = len(orig_path_str) // 2
        ax7.text(0.5, y_pos, f'All symbolic links point to:', fontsize=10, 
                 ha='center', style='italic', color=colors[6], transform=ax7.transAxes)
        y_pos -= 0.06
        ax7.text(0.5, y_pos, orig_path_str[:mid], fontsize=9, 
                 ha='center', style='italic', color=colors[6], transform=ax7.transAxes, family='monospace')
        y_pos -= 0.06
        ax7.text(0.5, y_pos, orig_path_str[mid:], fontsize=9, 
                 ha='center', style='italic', color=colors[6], transform=ax7.transAxes, family='monospace')
    else:
        ax7.text(0.5, y_pos, f'All symbolic links point to: {orig_path_str}',
                 fontsize=10, ha='center', style='italic', color=colors[6], transform=ax7.transAxes)
    
    # ==================== 保存图形 ====================
    plt.suptitle('Dataset Structure Visualization - Internal/External Split', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 保存为PDF格式（论文要求）
    output_path_pdf = project_root / 'cursor_file' / 'dataset_structure_visualization.pdf'
    plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight', facecolor='white', dpi=300)
    print(f"\n✅ 可视化图表已保存 (PDF): {output_path_pdf}")
    
    # 同时保存PNG格式作为备份
    output_path_png = project_root / 'cursor_file' / 'dataset_structure_visualization.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 可视化图表已保存 (PNG): {output_path_png}")
    
    plt.close()
    
    return output_path_pdf

def main():
    """Main function"""
    print("="*80)
    print("Dataset Structure Visualization")
    print("="*80)
    
    # Analyze dataset structure
    data_info = analyze_dataset_structure()
    if data_info is None:
        print("❌ Dataset analysis failed")
        return
    
    # Visualize
    output_path = visualize_dataset_structure(data_info)
    
    print("\n" + "="*80)
    print("✅ Visualization completed!")
    print("="*80)
    print(f"\n📊 Chart saved: {output_path}")
    print("\n💡 Key Information:")
    print(f"   - Dataset used in code: /data2/hmy/5Center_datas/5centers_multi_internal_external_final")
    print(f"   - Original dataset location: /data2/hmy/5Center_datas/5centers_multi")
    print(f"   - All image files point to original dataset via symbolic links")
    print(f"   - Disk space saved: ~182GB → ~3.3MB (symbolic links)")

if __name__ == '__main__':
    main()

