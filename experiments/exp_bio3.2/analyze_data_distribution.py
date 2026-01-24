#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据分布分析脚本
分析OCT、Colposcopy和语义信息的分布情况，生成论文用统计报告
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
DATA_ROOT = Path('/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal')
TRAIN_CSV = DATA_ROOT / 'internal_train' / 'labels.csv'
VAL_CSV = DATA_ROOT / 'internal_val' / 'labels.csv'

def load_data():
    """加载训练集和验证集数据"""
    print("📊 正在加载数据...")
    
    # 尝试不同的编码
    try:
        train_df = pd.read_csv(TRAIN_CSV, encoding='utf-8')
    except:
        train_df = pd.read_csv(TRAIN_CSV, encoding='gbk')
    
    try:
        val_df = pd.read_csv(VAL_CSV, encoding='utf-8')
    except:
        val_df = pd.read_csv(VAL_CSV, encoding='gbk')
    
    print(f"  ✅ 训练集: {len(train_df)} 个样本")
    print(f"  ✅ 验证集: {len(val_df)} 个样本")
    
    return train_df, val_df

def extract_center_id(oct_id):
    """从oct_id提取中心ID"""
    if pd.isna(oct_id):
        return 'unknown'
    oct_id_str = str(oct_id)
    if oct_id_str.startswith('M'):
        match = re.match(r'M(\d+)', oct_id_str)
        if match:
            return match.group(1)
        return oct_id_str[1:6] if len(oct_id_str) > 6 else oct_id_str[1:]
    return 'unknown'

def parse_tct(tct_str):
    """解析TCT结果"""
    if pd.isna(tct_str) or str(tct_str).strip() == '':
        return 'NILM'
    tct_str = str(tct_str).upper().strip()
    
    if 'ASC-US' in tct_str:
        return 'ASC-US'
    elif 'ASC-H' in tct_str:
        return 'ASC-H'
    elif 'LSIL' in tct_str:
        return 'LSIL'
    elif 'HSIL' in tct_str:
        return 'HSIL'
    elif 'SCC' in tct_str or '癌' in tct_str:
        return 'SCC'
    else:
        return 'NILM'

def parse_hpv(hpv_str):
    """解析HPV结果"""
    if pd.isna(hpv_str) or str(hpv_str).strip() == '' or str(hpv_str).strip() == '-':
        return 'Negative'
    
    hpv_str = str(hpv_str).strip()
    # 如果包含数字，认为是阳性
    if re.search(r'\d+', hpv_str):
        return 'Positive'
    elif any(k in hpv_str.lower() for k in ['positive', '阳性', '高危', '16', '18']):
        return 'Positive'
    else:
        return 'Negative'

def analyze_oct_distribution(df, split_name):
    """分析OCT数据分布"""
    print(f"\n{'='*60}")
    print(f"📸 OCT数据分布分析 - {split_name}")
    print(f"{'='*60}")
    
    # 基本统计
    total_samples = len(df)
    positive_samples = sum(df['label'] == 1)
    negative_samples = sum(df['label'] == 0)
    
    print(f"\n1. 样本标签分布:")
    print(f"   总样本数: {total_samples}")
    print(f"   阳性样本 (label=1): {positive_samples} ({positive_samples/total_samples*100:.2f}%)")
    print(f"   阴性样本 (label=0): {negative_samples} ({negative_samples/total_samples*100:.2f}%)")
    
    # OCT帧数分布
    print(f"\n2. OCT帧数分布:")
    oct_counts = df['oct_count'].dropna()
    print(f"   总样本数: {len(oct_counts)}")
    print(f"   平均帧数: {oct_counts.mean():.2f}")
    print(f"   中位数帧数: {oct_counts.median():.2f}")
    print(f"   最小帧数: {oct_counts.min()}")
    print(f"   最大帧数: {oct_counts.max()}")
    print(f"   标准差: {oct_counts.std():.2f}")
    
    # 按标签分组统计
    if 'label' in df.columns:
        pos_oct = df[df['label'] == 1]['oct_count'].dropna()
        neg_oct = df[df['label'] == 0]['oct_count'].dropna()
        
        print(f"\n   阳性样本OCT帧数:")
        print(f"     平均: {pos_oct.mean():.2f}, 中位数: {pos_oct.median():.2f}, 范围: [{pos_oct.min()}, {pos_oct.max()}]")
        print(f"   阴性样本OCT帧数:")
        print(f"     平均: {neg_oct.mean():.2f}, 中位数: {neg_oct.median():.2f}, 范围: [{neg_oct.min()}, {neg_oct.max()}]")
    
    # 帧数区间分布
    print(f"\n3. OCT帧数区间分布:")
    bins = [0, 20, 40, 60, 80, 100, 120, 200, float('inf')]
    labels = ['0-20', '21-40', '41-60', '61-80', '81-100', '101-120', '121-200', '200+']
    df['oct_range'] = pd.cut(df['oct_count'], bins=bins, labels=labels, right=False)
    range_counts = df['oct_range'].value_counts().sort_index()
    for range_name, count in range_counts.items():
        print(f"   {range_name}帧: {count} 个样本 ({count/len(df)*100:.2f}%)")
    
    return {
        'total_samples': total_samples,
        'positive_samples': positive_samples,
        'negative_samples': negative_samples,
        'oct_mean': oct_counts.mean(),
        'oct_median': oct_counts.median(),
        'oct_min': oct_counts.min(),
        'oct_max': oct_counts.max(),
        'oct_std': oct_counts.std(),
        'range_distribution': range_counts.to_dict()
    }

def analyze_colposcopy_distribution(df, split_name):
    """分析Colposcopy数据分布"""
    print(f"\n{'='*60}")
    print(f"🔬 Colposcopy数据分布分析 - {split_name}")
    print(f"{'='*60}")
    
    # Colposcopy图像数分布
    col_counts = df['col_count'].dropna()
    print(f"\n1. Colposcopy图像数分布:")
    print(f"   总样本数: {len(col_counts)}")
    print(f"   平均图像数: {col_counts.mean():.2f}")
    print(f"   中位数图像数: {col_counts.median():.2f}")
    print(f"   最小图像数: {col_counts.min()}")
    print(f"   最大图像数: {col_counts.max()}")
    
    # 图像数分布统计
    col_dist = col_counts.value_counts().sort_index()
    print(f"\n2. 图像数详细分布:")
    for count, num_samples in col_dist.items():
        print(f"   {count}张图像: {num_samples} 个样本 ({num_samples/len(df)*100:.2f}%)")
    
    # 按标签分组统计
    if 'label' in df.columns:
        pos_col = df[df['label'] == 1]['col_count'].dropna()
        neg_col = df[df['label'] == 0]['col_count'].dropna()
        
        print(f"\n3. 按标签分组统计:")
        print(f"   阳性样本Colposcopy图像数:")
        print(f"     平均: {pos_col.mean():.2f}, 中位数: {pos_col.median():.2f}, 范围: [{pos_col.min()}, {pos_col.max()}]")
        print(f"   阴性样本Colposcopy图像数:")
        print(f"     平均: {neg_col.mean():.2f}, 中位数: {neg_col.median():.2f}, 范围: [{neg_col.min()}, {neg_col.max()}]")
    
    return {
        'col_mean': col_counts.mean(),
        'col_median': col_counts.median(),
        'col_min': col_counts.min(),
        'col_max': col_counts.max(),
        'col_distribution': col_dist.to_dict()
    }

def analyze_semantic_distribution(df, split_name):
    """分析语义信息分布（年龄、HPV、TCT）"""
    print(f"\n{'='*60}")
    print(f"📝 语义信息分布分析 - {split_name}")
    print(f"{'='*60}")
    
    # 年龄分布（过滤异常值，只保留18-100岁之间的合理年龄）
    print(f"\n1. 年龄分布:")
    ages_raw = df['age'].dropna()
    # 过滤异常值：只保留18-100岁之间的年龄
    ages = ages_raw[(ages_raw >= 18) & (ages_raw <= 100)]
    if len(ages) < len(ages_raw):
        print(f"   ⚠️ 过滤了 {len(ages_raw) - len(ages)} 个异常年龄值")
    print(f"   总样本数: {len(ages)}")
    print(f"   平均年龄: {ages.mean():.2f} 岁")
    print(f"   中位数年龄: {ages.median():.2f} 岁")
    print(f"   年龄范围: [{ages.min():.1f}, {ages.max():.1f}] 岁")
    print(f"   标准差: {ages.std():.2f} 岁")
    
    # 年龄区间分布（只对有效年龄进行分组）
    age_bins = [0, 30, 40, 50, 60, 70, 100]
    age_labels = ['<30', '30-40', '40-50', '50-60', '60-70', '70+']
    df_valid_age = df[(df['age'] >= 18) & (df['age'] <= 100)].copy()
    df_valid_age['age_range'] = pd.cut(df_valid_age['age'], bins=age_bins, labels=age_labels, right=False)
    age_range_counts = df_valid_age['age_range'].value_counts().sort_index()
    print(f"\n   年龄区间分布:")
    for range_name, count in age_range_counts.items():
        print(f"     {range_name}岁: {count} 个样本 ({count/len(df_valid_age)*100:.2f}%)")
    
    # HPV分布
    print(f"\n2. HPV结果分布:")
    df['hpv_parsed'] = df['hpv'].apply(parse_hpv)
    hpv_dist = df['hpv_parsed'].value_counts()
    for hpv_type, count in hpv_dist.items():
        print(f"   {hpv_type}: {count} 个样本 ({count/len(df)*100:.2f}%)")
    
    # TCT分布
    print(f"\n3. TCT结果分布:")
    df['tct_parsed'] = df['tct'].apply(parse_tct)
    tct_dist = df['tct_parsed'].value_counts()
    for tct_type, count in tct_dist.items():
        print(f"   {tct_type}: {count} 个样本 ({count/len(df)*100:.2f}%)")
    
    # 中心分布
    print(f"\n4. 中心分布:")
    df['center_id'] = df['oct_id'].apply(extract_center_id)
    center_dist = df['center_id'].value_counts().sort_index()
    for center_id, count in center_dist.items():
        print(f"   中心 {center_id}: {count} 个样本 ({count/len(df)*100:.2f}%)")
    
    # 阳性位点分布（仅阳性样本）
    if 'positive_sites' in df.columns:
        pos_df = df[df['label'] == 1]
        pos_sites = pos_df['positive_sites'].dropna()
        if len(pos_sites) > 0:
            print(f"\n5. 阳性位点分布 (仅阳性样本):")
            # 统计所有位点
            all_sites = []
            for sites_str in pos_sites:
                if pd.notna(sites_str) and str(sites_str).strip():
                    sites = [s.strip() for s in str(sites_str).split(',')]
                    all_sites.extend(sites)
            
            site_counts = Counter(all_sites)
            print(f"   总阳性位点数: {len(all_sites)}")
            print(f"   位点分布:")
            for site, count in sorted(site_counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
                print(f"     位点 {site}: {count} 次 ({count/len(pos_sites)*100:.2f}%)")
    
    return {
        'age_mean': ages.mean(),
        'age_median': ages.median(),
        'age_min': ages.min(),
        'age_max': ages.max(),
        'age_std': ages.std(),
        'hpv_distribution': hpv_dist.to_dict(),
        'tct_distribution': tct_dist.to_dict(),
        'center_distribution': center_dist.to_dict()
    }

def generate_summary_report(train_stats, val_stats):
    """生成汇总报告"""
    print(f"\n{'='*60}")
    print(f"📊 数据分布汇总报告")
    print(f"{'='*60}")
    
    print(f"\n【总体统计】")
    print(f"训练集总样本数: {train_stats['oct']['total_samples']}")
    print(f"验证集总样本数: {val_stats['oct']['total_samples']}")
    print(f"合计: {train_stats['oct']['total_samples'] + val_stats['oct']['total_samples']}")
    
    print(f"\n【OCT数据汇总】")
    print(f"训练集 - 平均帧数: {train_stats['oct']['oct_mean']:.2f}, 中位数: {train_stats['oct']['oct_median']:.2f}")
    print(f"验证集 - 平均帧数: {val_stats['oct']['oct_mean']:.2f}, 中位数: {val_stats['oct']['oct_median']:.2f}")
    
    print(f"\n【Colposcopy数据汇总】")
    print(f"训练集 - 平均图像数: {train_stats['col']['col_mean']:.2f}, 中位数: {train_stats['col']['col_median']:.2f}")
    print(f"验证集 - 平均图像数: {val_stats['col']['col_mean']:.2f}, 中位数: {val_stats['col']['col_median']:.2f}")
    
    print(f"\n【语义信息汇总】")
    print(f"训练集 - 平均年龄: {train_stats['semantic']['age_mean']:.2f} 岁")
    print(f"验证集 - 平均年龄: {val_stats['semantic']['age_mean']:.2f} 岁")
    
    # 生成LaTeX表格格式（用于论文）
    print(f"\n{'='*60}")
    print(f"📄 LaTeX表格格式（可直接用于论文）")
    print(f"{'='*60}")
    
    print(f"\n% 数据集分布统计表")
    print(f"\\begin{{table}}[h]")
    print(f"\\centering")
    print(f"\\caption{{数据集分布统计}}")
    print(f"\\label{{tab:data_distribution}}")
    print(f"\\begin{{tabular}}{{lcc}}")
    print(f"\\toprule")
    print(f"项目 & 训练集 & 验证集 \\\\")
    print(f"\\midrule")
    print(f"总样本数 & {train_stats['oct']['total_samples']} & {val_stats['oct']['total_samples']} \\\\")
    print(f"阳性样本数 & {train_stats['oct']['positive_samples']} & {val_stats['oct']['positive_samples']} \\\\")
    print(f"阴性样本数 & {train_stats['oct']['negative_samples']} & {val_stats['oct']['negative_samples']} \\\\")
    print(f"OCT平均帧数 & {train_stats['oct']['oct_mean']:.1f} & {val_stats['oct']['oct_mean']:.1f} \\\\")
    print(f"Colposcopy平均图像数 & {train_stats['col']['col_mean']:.1f} & {val_stats['col']['col_mean']:.1f} \\\\")
    print(f"平均年龄（岁） & {train_stats['semantic']['age_mean']:.1f} & {val_stats['semantic']['age_mean']:.1f} \\\\")
    print(f"\\bottomrule")
    print(f"\\end{{tabular}}")
    print(f"\\end{{table}}")

def main():
    """主函数"""
    print("="*60)
    print("数据分布分析工具")
    print("="*60)
    
    # 加载数据
    train_df, val_df = load_data()
    
    # 分析训练集
    train_oct_stats = analyze_oct_distribution(train_df, "训练集")
    train_col_stats = analyze_colposcopy_distribution(train_df, "训练集")
    train_semantic_stats = analyze_semantic_distribution(train_df, "训练集")
    
    # 分析验证集
    val_oct_stats = analyze_oct_distribution(val_df, "验证集")
    val_col_stats = analyze_colposcopy_distribution(val_df, "验证集")
    val_semantic_stats = analyze_semantic_distribution(val_df, "验证集")
    
    # 生成汇总报告
    train_stats = {
        'oct': train_oct_stats,
        'col': train_col_stats,
        'semantic': train_semantic_stats
    }
    val_stats = {
        'oct': val_oct_stats,
        'col': val_col_stats,
        'semantic': val_semantic_stats
    }
    
    generate_summary_report(train_stats, val_stats)
    
    print(f"\n{'='*60}")
    print("✅ 分析完成！")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

