#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件生成信息:
- 生成时间: 2025-12-25 10:46:50 (使用nvidia-smi获取CUDA时间)
- 生成需求: 用户要求为Leave-Centers-Out数据集创建软链接，并生成可视化图表
- 生成原因: 确保训练过程中使用正确的数据集划分，并通过可视化让读者一目了然地了解数据集分布
- 相关任务: 数据集软链接创建，可视化生成，SCI论文准备

文件功能: 为Leave-Centers-Out数据集创建软链接并生成可视化图表
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
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cursor_file.visualization_config import setup_miccai_style, get_color_palette

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
    """Identify medical center"""
    oct_str = str(oct_id)
    if 'M22105' in oct_str:
        return 'Enshi', 'internal'
    elif 'M22102' in oct_str:
        return 'Xiangyang', 'internal'
    elif 'M22104' in oct_str or 'M22101' in oct_str:
        return 'Shiyan', 'external'
    elif 'M0008' in oct_str:
        return 'Jingzhou', 'external'
    elif 'M20203' in oct_str or 'M20105' in oct_str:
        return 'Wuda', 'internal'
    return 'Unknown', 'unknown'

def find_data_in_original_dataset(oct_id: str, patient_id: str, original_data_root: Path, 
                                   search_in_train: bool = True, search_in_test: bool = True):
    """Find image files in original dataset"""
    result = {'oct': None, 'col': None, 'source': None}
    
    search_paths = []
    if search_in_train:
        search_paths.append(('train', original_data_root / 'train'))
    if search_in_test:
        search_paths.append(('test', original_data_root / 'test'))
    
    for source_name, base_path in search_paths:
        if result['oct'] is None:
            oct_path = base_path / 'oct' / oct_id
            if oct_path.exists() and oct_path.is_dir():
                image_files = list(oct_path.glob('*.png')) + list(oct_path.glob('*.jpg')) + list(oct_path.glob('*.jpeg'))
                if image_files:
                    result['oct'] = oct_path
                    if result['source'] is None:
                        result['source'] = source_name
        
        if result['col'] is None:
            col_path = base_path / 'col' / patient_id
            if col_path.exists() and col_path.is_dir():
                image_files = list(col_path.glob('*.png')) + list(col_path.glob('*.jpg')) + list(col_path.glob('*.jpeg'))
                if image_files:
                    result['col'] = col_path
                    if result['source'] is None:
                        result['source'] = source_name
        
        if result['oct'] is not None and result['col'] is not None:
            break
    
    return result

def create_symbolic_links(split_labels_file: Path, original_data_root: Path,
                          target_base_dir: Path, split_type: str = 'train'):
    """Create symbolic links for split dataset"""
    print(f"\n{'='*80}")
    print(f"Creating symbolic links for {split_type} dataset")
    print(f"{'='*80}")
    
    if not split_labels_file.exists():
        print(f"❌ Label file not found: {split_labels_file}")
        return False
    
    df = pd.read_csv(split_labels_file)
    print(f"✅ Loaded label file: {len(df)} samples")
    
    # Determine target directories
    if split_type == 'external_test':
        target_oct_dir = target_base_dir / 'external_validation' / 'oct'
        target_col_dir = target_base_dir / 'external_validation' / 'col'
    else:
        target_oct_dir = target_base_dir / 'internal_train' / split_type / 'oct'
        target_col_dir = target_base_dir / 'internal_train' / split_type / 'col'
    
    target_oct_dir.mkdir(parents=True, exist_ok=True)
    target_col_dir.mkdir(parents=True, exist_ok=True)
    
    found_oct_count = 0
    found_col_count = 0
    missing_oct_count = 0
    missing_col_count = 0
    
    for idx, row in df.iterrows():
        oct_id = str(row['OCT'])
        patient_id = str(row['ID'])
        
        data_info = find_data_in_original_dataset(
            oct_id, patient_id, original_data_root,
            search_in_train=True, search_in_test=True
        )
        
        # Create OCT symbolic link
        if data_info['oct'] is not None:
            target_oct_link = target_oct_dir / oct_id
            if not target_oct_link.exists():
                try:
                    os.symlink(data_info['oct'], target_oct_link)
                    found_oct_count += 1
                except OSError as e:
                    missing_oct_count += 1
            else:
                found_oct_count += 1
        else:
            missing_oct_count += 1
        
        # Create Colposcopy symbolic link
        if data_info['col'] is not None:
            target_col_link = target_col_dir / patient_id
            if not target_col_link.exists():
                try:
                    os.symlink(data_info['col'], target_col_link)
                    found_col_count += 1
                except OSError as e:
                    missing_col_count += 1
            else:
                found_col_count += 1
        else:
            missing_col_count += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{len(df)} (OCT: {found_oct_count}, Col: {found_col_count})")
    
    print(f"\n✅ {split_type} dataset processing completed:")
    print(f"   OCT images: {found_oct_count}/{len(df)} ({found_oct_count/len(df)*100:.1f}%)")
    print(f"   Colposcopy images: {found_col_count}/{len(df)} ({found_col_count/len(df)*100:.1f}%)")
    
    return found_oct_count > 0 or found_col_count > 0

def visualize_leave_centers_out_dataset(split_dataset_path: Path):
    """Visualize Leave-Centers-Out dataset distribution"""
    print("\n" + "="*80)
    print("Generating Visualization for Leave-Centers-Out Dataset")
    print("="*80)
    
    setup_miccai_style()
    colors = get_color_palette(8)
    
    # Load data
    train_df = pd.read_csv(split_dataset_path / 'train_labels.csv')
    val_df = pd.read_csv(split_dataset_path / 'val_labels.csv')
    external_df = pd.read_csv(split_dataset_path / 'external_test_labels.csv')
    
    # Identify centers
    for name, df in [('train', train_df), ('val', val_df), ('external', external_df)]:
        df['center_name'], df['center_type'] = zip(*df['OCT'].apply(identify_center))
    
    # Create figure
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4, left=0.08, right=0.95, top=0.93, bottom=0.08)
    
    # ==================== 1. Leave-Centers-Out Strategy Overview ====================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    y_pos = 0.95
    ax1.text(0.05, y_pos, 'Leave-Centers-Out Strategy', fontsize=16, fontweight='bold', transform=ax1.transAxes)
    y_pos -= 0.15
    
    ax1.text(0.05, y_pos, 'External Test Set:', fontsize=12, fontweight='bold', color=colors[0], transform=ax1.transAxes)
    y_pos -= 0.12
    ax1.text(0.08, y_pos, 'Shiyan + Jingzhou', fontsize=11, transform=ax1.transAxes)
    y_pos -= 0.10
    ax1.text(0.08, y_pos, f'{len(external_df)} samples', fontsize=10, style='italic', transform=ax1.transAxes)
    y_pos -= 0.12
    ax1.text(0.08, y_pos, 'Never participated in training', fontsize=9, color=colors[0], transform=ax1.transAxes)
    y_pos -= 0.15
    
    ax1.text(0.05, y_pos, 'Internal Dev Set:', fontsize=12, fontweight='bold', color=colors[4], transform=ax1.transAxes)
    y_pos -= 0.12
    ax1.text(0.08, y_pos, 'Enshi + Xiangyang + Wuda', fontsize=11, transform=ax1.transAxes)
    y_pos -= 0.10
    ax1.text(0.08, y_pos, f'{len(train_df) + len(val_df)} samples (800+)', fontsize=10, style='italic', transform=ax1.transAxes)
    y_pos -= 0.12
    ax1.text(0.08, y_pos, 'Used for training and validation', fontsize=9, color=colors[4], transform=ax1.transAxes)
    
    # ==================== 2. Sample Distribution by Dataset ====================
    ax2 = fig.add_subplot(gs[0, 1])
    
    dataset_names = ['Train', 'Validation', 'External Test']
    sample_counts = [len(train_df), len(val_df), len(external_df)]
    dataset_colors = [colors[4], colors[2], colors[0]]
    
    bars = ax2.bar(dataset_names, sample_counts, color=dataset_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Distribution by Dataset', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # ==================== 3. Center Distribution ====================
    ax3 = fig.add_subplot(gs[0, 2])
    
    center_counts = defaultdict(int)
    center_types = {}
    
    for name, df in [('train', train_df), ('val', val_df), ('external', external_df)]:
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
    
    for bar in bars:
        height = bar.get_height()
        y_offset = height + max(counts) * 0.01
        ax3.text(bar.get_x() + bar.get_width()/2., y_offset,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ==================== 4. Class Distribution ====================
    ax4 = fig.add_subplot(gs[1, 0])
    
    label_counts = defaultdict(int)
    for name, df in [('train', train_df), ('val', val_df), ('external', external_df)]:
        label_counts['Positive'] += (df['label'] == 1).sum()
        label_counts['Negative'] += (df['label'] == 0).sum()
    
    labels = ['Positive', 'Negative']
    counts = [label_counts['Positive'], label_counts['Negative']]
    label_colors = [colors[0], colors[4]]
    
    bars = ax4.bar(labels, counts, color=label_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Number of Samples', fontsize=12)
    ax4.set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # ==================== 5. Internal vs External Split ====================
    ax5 = fig.add_subplot(gs[1, 1])
    
    internal_count = len(train_df) + len(val_df)
    external_count = len(external_df)
    
    sizes = [internal_count, external_count]
    labels_pie = ['Internal\n(Development)', 'External\n(Test)']
    colors_pie = [colors[4], colors[0]]
    explode = (0.05, 0.1)
    
    wedges, texts, autotexts = ax5.pie(sizes, labels=labels_pie, colors=colors_pie, 
                                        autopct='%1.1f%%', explode=explode,
                                        startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax5.set_title('Internal vs External Dataset Split', fontsize=14, fontweight='bold')
    
    # ==================== 6. Class Distribution by Dataset ====================
    ax6 = fig.add_subplot(gs[1, 2])
    
    dataset_labels = ['Train', 'Validation', 'External Test']
    pos_counts = [(train_df['label'] == 1).sum(), (val_df['label'] == 1).sum(), (external_df['label'] == 1).sum()]
    neg_counts = [(train_df['label'] == 0).sum(), (val_df['label'] == 0).sum(), (external_df['label'] == 0).sum()]
    
    x = range(len(dataset_labels))
    width = 0.35
    
    bars1 = ax6.bar([i - width/2 for i in x], pos_counts, width, label='Positive', 
                    color=colors[0], alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax6.bar([i + width/2 for i in x], neg_counts, width, label='Negative',
                    color=colors[4], alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax6.set_ylabel('Number of Samples', fontsize=12)
    ax6.set_title('Class Distribution by Dataset', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(dataset_labels, fontsize=10)
    ax6.legend(fontsize=10, loc='upper right')
    ax6.grid(True, alpha=0.3, axis='y')
    
    for i, (pos, neg) in enumerate(zip(pos_counts, neg_counts)):
        if pos > 0:
            ax6.text(i - width/2, pos + max(max(pos_counts), max(neg_counts)) * 0.01,
                    f'{int(pos)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        if neg > 0:
            ax6.text(i + width/2, neg + max(max(pos_counts), max(neg_counts)) * 0.01,
                    f'{int(neg)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # ==================== 7. Detailed Statistics Table ====================
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    y_pos = 0.95
    ax7.text(0.5, y_pos, 'Leave-Centers-Out Dataset Statistics', fontsize=16, fontweight='bold',
             ha='center', transform=ax7.transAxes)
    y_pos -= 0.12
    
    # Table header
    ax7.text(0.08, y_pos, 'Dataset', fontsize=12, fontweight='bold', transform=ax7.transAxes)
    ax7.text(0.25, y_pos, 'Samples', fontsize=12, fontweight='bold', transform=ax7.transAxes)
    ax7.text(0.40, y_pos, 'Positive', fontsize=12, fontweight='bold', transform=ax7.transAxes)
    ax7.text(0.55, y_pos, 'Negative', fontsize=12, fontweight='bold', transform=ax7.transAxes)
    ax7.text(0.70, y_pos, 'Positive Rate', fontsize=12, fontweight='bold', transform=ax7.transAxes)
    ax7.text(0.85, y_pos, 'Centers', fontsize=12, fontweight='bold', transform=ax7.transAxes)
    y_pos -= 0.10
    
    # Data rows
    datasets_info = [
        ('Train', train_df, 'Enshi, Xiangyang, Wuda'),
        ('Validation', val_df, 'Enshi, Xiangyang, Wuda'),
        ('External Test', external_df, 'Shiyan, Jingzhou')
    ]
    
    for name, df, centers_str in datasets_info:
        pos = (df['label'] == 1).sum()
        neg = (df['label'] == 0).sum()
        total = len(df)
        pos_rate = pos / total * 100
        
        ax7.text(0.08, y_pos, name, fontsize=11, transform=ax7.transAxes)
        ax7.text(0.25, y_pos, f'{total}', fontsize=11, ha='center', transform=ax7.transAxes)
        ax7.text(0.40, y_pos, f'{pos}', fontsize=11, ha='center', transform=ax7.transAxes)
        ax7.text(0.55, y_pos, f'{neg}', fontsize=11, ha='center', transform=ax7.transAxes)
        ax7.text(0.70, y_pos, f'{pos_rate:.1f}%', fontsize=11, ha='center', transform=ax7.transAxes)
        ax7.text(0.85, y_pos, centers_str, fontsize=10, transform=ax7.transAxes)
        y_pos -= 0.10
    
    # Total row
    y_pos -= 0.05
    total_samples = len(train_df) + len(val_df) + len(external_df)
    total_pos = (train_df['label'] == 1).sum() + (val_df['label'] == 1).sum() + (external_df['label'] == 1).sum()
    total_neg = (train_df['label'] == 0).sum() + (val_df['label'] == 0).sum() + (external_df['label'] == 0).sum()
    total_pos_rate = total_pos / total_samples * 100
    
    ax7.text(0.08, y_pos, 'Total', fontsize=12, fontweight='bold', transform=ax7.transAxes)
    ax7.text(0.25, y_pos, f'{total_samples}', fontsize=12, fontweight='bold', ha='center', transform=ax7.transAxes)
    ax7.text(0.40, y_pos, f'{total_pos}', fontsize=12, fontweight='bold', ha='center', transform=ax7.transAxes)
    ax7.text(0.55, y_pos, f'{total_neg}', fontsize=12, fontweight='bold', ha='center', transform=ax7.transAxes)
    ax7.text(0.70, y_pos, f'{total_pos_rate:.1f}%', fontsize=12, fontweight='bold', ha='center', transform=ax7.transAxes)
    ax7.text(0.85, y_pos, 'All 5 centers', fontsize=11, fontweight='bold', transform=ax7.transAxes)
    
    # Add note
    y_pos -= 0.12
    ax7.text(0.5, y_pos, 'Note: External test set (Shiyan + Jingzhou) never participated in training',
             fontsize=10, ha='center', style='italic', color=colors[6], transform=ax7.transAxes)
    
    # Save figure
    plt.suptitle('Leave-Centers-Out Dataset Distribution Visualization', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Save as PDF (for paper)
    output_path_pdf = project_root / 'cursor_file' / 'leave_centers_out_dataset_visualization.pdf'
    plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight', facecolor='white', dpi=300)
    print(f"\n✅ Visualization saved (PDF): {output_path_pdf}")
    
    # Save as PNG (backup)
    output_path_png = project_root / 'cursor_file' / 'leave_centers_out_dataset_visualization.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Visualization saved (PNG): {output_path_png}")
    
    plt.close()
    
    return output_path_pdf

def main():
    """Main function"""
    print("="*80)
    print("Create Symbolic Links and Generate Visualization")
    print("for Leave-Centers-Out Dataset")
    print("="*80)
    
    original_dataset_path = Path('/data2/hmy/5Center_datas/5centers_multi')
    split_dataset_path = Path('/data2/hmy/5Center_datas/5centers_multi_leave_centers_out')
    
    if not original_dataset_path.exists():
        print(f"❌ Original dataset path not found: {original_dataset_path}")
        return
    
    if not split_dataset_path.exists():
        print(f"❌ Split dataset path not found: {split_dataset_path}")
        return
    
    print(f"\nOriginal dataset: {original_dataset_path}")
    print(f"Split dataset: {split_dataset_path}")
    
    # Create symbolic links
    print("\n" + "="*80)
    print("Step 1: Creating Symbolic Links")
    print("="*80)
    
    train_labels = split_dataset_path / 'train_labels.csv'
    val_labels = split_dataset_path / 'val_labels.csv'
    external_test_labels = split_dataset_path / 'external_test_labels.csv'
    
    if train_labels.exists():
        create_symbolic_links(train_labels, original_dataset_path, split_dataset_path, 'train')
    
    if val_labels.exists():
        create_symbolic_links(val_labels, original_dataset_path, split_dataset_path, 'val')
    
    if external_test_labels.exists():
        create_symbolic_links(external_test_labels, original_dataset_path, split_dataset_path, 'external_test')
    
    # Generate visualization
    print("\n" + "="*80)
    print("Step 2: Generating Visualization")
    print("="*80)
    
    output_path = visualize_leave_centers_out_dataset(split_dataset_path)
    
    print("\n" + "="*80)
    print("✅ All tasks completed!")
    print("="*80)
    print(f"\n📁 Dataset path: {split_dataset_path}")
    print(f"📊 Visualization: {output_path}")
    print(f"\n💡 Next steps:")
    print(f"  1. Update data paths in training scripts")
    print(f"  2. Use the visualization in your paper")

if __name__ == '__main__':
    main()

