#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建内部外部验证数据集
将5个多中心数据集分割为训练集和外部验证集
支持两种方案：
1. 中心分割：将1-2个中心作为外部验证集
2. 时间分割：按时间顺序分割，后期数据作为外部验证集
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import shutil

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class InternalExternalValidationCreator:
    """内部外部验证数据集创建器"""
    
    def __init__(self, data_root: str = '5centers_multi'):
        """
        Args:
            data_root: 数据根目录
        """
        self.data_root = Path(data_root)
        self.train_labels_file = self.data_root / 'train_labels.csv'
        self.test_labels_file = self.data_root / 'test_labels.csv'
        
        # 中心映射（从OCT ID提取）
        self.center_mapping = {
            'M22105': '恩施',
            'M0008': '荆州',
            'M22104': '十堰',
            'M22101': '十堰',
            'M20203': '武大',
            'M20105': '武大',
            'M22102': '襄阳'
        }
    
    def identify_center(self, oct_id: str) -> str:
        """从OCT ID识别中心"""
        if pd.isna(oct_id):
            return 'Unknown'
        
        oct_str = str(oct_id)
        for code, center in self.center_mapping.items():
            if code in oct_str:
                return center
        return 'Unknown'
    
    def analyze_center_distribution(self) -> Dict:
        """分析中心分布"""
        print("📊 分析中心分布...")
        
        # 加载所有数据
        train_df = pd.read_csv(self.train_labels_file)
        test_df = pd.read_csv(self.test_labels_file)
        
        # 识别中心
        train_df['center'] = train_df['OCT'].apply(self.identify_center)
        test_df['center'] = test_df['OCT'].apply(self.identify_center)
        
        # 统计分布
        distribution = {}
        all_centers = set(train_df['center'].unique()) | set(test_df['center'].unique())
        all_centers.discard('Unknown')
        
        for center in sorted(all_centers):
            train_center = train_df[train_df['center'] == center]
            test_center = test_df[test_df['center'] == center]
            
            distribution[center] = {
                'train_samples': len(train_center),
                'test_samples': len(test_center),
                'train_total': len(train_center),
                'test_total': len(test_center),
                'train_positive': int(train_center['label'].sum()) if 'label' in train_center.columns else 0,
                'train_negative': int((train_center['label'] == 0).sum()) if 'label' in train_center.columns else 0,
                'test_positive': int(test_center['label'].sum()) if 'label' in test_center.columns else 0,
                'test_negative': int((test_center['label'] == 0).sum()) if 'label' in test_center.columns else 0,
            }
        
        # 打印分布
        print("\n中心分布统计:")
        print(f"{'中心':<10} {'训练样本':<12} {'测试样本':<12} {'训练阳性':<12} {'测试阳性':<12}")
        print("-" * 70)
        for center, stats in distribution.items():
            print(f"{center:<10} {stats['train_total']:<12} {stats['test_total']:<12} "
                  f"{stats['train_positive']:<12} {stats['test_positive']:<12}")
        
        return distribution
    
    def create_center_split(self, external_centers: List[str],
                           output_dir: str = '5centers_multi_internal_external') -> Dict:
        """
        创建中心分割验证数据集
        
        Args:
            external_centers: 作为外部验证集的中心列表（如 ['恩施', '荆州']）
            output_dir: 输出目录
        
        Returns:
            分割结果字典
        """
        print(f"\n🔀 创建中心分割验证数据集...")
        print(f"   外部验证中心: {external_centers}")
        
        # 加载数据
        train_df = pd.read_csv(self.train_labels_file)
        test_df = pd.read_csv(self.test_labels_file)
        
        # 识别中心
        train_df['center'] = train_df['OCT'].apply(self.identify_center)
        test_df['center'] = test_df['OCT'].apply(self.identify_center)
        
        # 分割数据
        # 训练集：不在external_centers中的中心
        train_internal = train_df[~train_df['center'].isin(external_centers)].copy()
        test_internal = test_df[~test_df['center'].isin(external_centers)].copy()
        
        # 外部验证集：在external_centers中的中心
        train_external = train_df[train_df['center'].isin(external_centers)].copy()
        test_external = test_df[test_df['center'].isin(external_centers)].copy()
        
        # 合并训练和测试数据
        internal_train = pd.concat([train_internal, test_internal], ignore_index=True)
        external_val = pd.concat([train_external, test_external], ignore_index=True)
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 创建内部训练集目录结构
        internal_dir = output_path / 'internal_train'
        internal_dir.mkdir(exist_ok=True)
        (internal_dir / 'train').mkdir(exist_ok=True)
        (internal_dir / 'train' / 'oct').mkdir(exist_ok=True)
        (internal_dir / 'train' / 'col').mkdir(exist_ok=True)
        (internal_dir / 'test').mkdir(exist_ok=True)
        (internal_dir / 'test' / 'oct').mkdir(exist_ok=True)
        (internal_dir / 'test' / 'col').mkdir(exist_ok=True)
        
        # 创建外部验证集目录结构
        external_dir = output_path / 'external_validation'
        external_dir.mkdir(exist_ok=True)
        (external_dir / 'oct').mkdir(exist_ok=True)
        (external_dir / 'col').mkdir(exist_ok=True)
        
        # 保存标签文件
        internal_train.to_csv(internal_dir / 'train_labels.csv', index=False)
        external_val.to_csv(external_dir / 'external_labels.csv', index=False)
        
        # 复制图像文件（可选，如果数据量大可以只创建符号链接）
        print("\n📁 复制图像文件...")
        self._copy_images(internal_train, self.data_root, internal_dir / 'train')
        self._copy_images(external_val, self.data_root, external_dir)
        
        # 生成统计报告
        stats = {
            'split_method': 'center_split',
            'external_centers': external_centers,
            'internal_train': {
                'total_samples': len(internal_train),
                'positive': int(internal_train['label'].sum()) if 'label' in internal_train.columns else 0,
                'negative': int((internal_train['label'] == 0).sum()) if 'label' in internal_train.columns else 0,
                'centers': sorted(internal_train['center'].unique().tolist())
            },
            'external_validation': {
                'total_samples': len(external_val),
                'positive': int(external_val['label'].sum()) if 'label' in external_val.columns else 0,
                'negative': int((external_val['label'] == 0).sum()) if 'label' in external_val.columns else 0,
                'centers': sorted(external_val['center'].unique().tolist())
            },
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存统计报告
        stats_file = output_path / 'split_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 中心分割完成！")
        print(f"   内部训练集: {len(internal_train)} 样本")
        print(f"   外部验证集: {len(external_val)} 样本")
        print(f"   结果保存在: {output_dir}")
        
        return stats
    
    def create_time_split(self, split_date: str = None,
                         split_ratio: float = 0.2,
                         output_dir: str = '5centers_multi_internal_external') -> Dict:
        """
        创建时间分割验证数据集
        
        Args:
            split_date: 分割日期（格式：'2023-06-01'），如果为None则按比例分割
            split_ratio: 外部验证集比例（当split_date为None时使用）
            output_dir: 输出目录
        
        Returns:
            分割结果字典
        """
        print(f"\n🔀 创建时间分割验证数据集...")
        
        # 加载数据
        train_df = pd.read_csv(self.train_labels_file)
        test_df = pd.read_csv(self.test_labels_file)
        
        # 合并数据
        all_data = pd.concat([train_df, test_df], ignore_index=True)
        
        # 从OCT ID或患者ID提取日期
        # 假设日期格式为 YYYYMMDD_姓名
        def extract_date(oct_id):
            if pd.isna(oct_id):
                return None
            oct_str = str(oct_id)
            # 尝试提取日期（前8位数字）
            import re
            match = re.search(r'(\d{8})', oct_str)
            if match:
                date_str = match.group(1)
                try:
                    return pd.to_datetime(date_str, format='%Y%m%d')
                except:
                    return None
            return None
        
        all_data['date'] = all_data['OCT'].apply(extract_date)
        
        # 移除无法提取日期的样本
        all_data = all_data[all_data['date'].notna()].copy()
        
        # 按日期排序
        all_data = all_data.sort_values('date').reset_index(drop=True)
        
        # 分割数据
        if split_date:
            split_datetime = pd.to_datetime(split_date)
            internal_data = all_data[all_data['date'] < split_datetime].copy()
            external_data = all_data[all_data['date'] >= split_datetime].copy()
        else:
            split_idx = int(len(all_data) * (1 - split_ratio))
            internal_data = all_data.iloc[:split_idx].copy()
            external_data = all_data.iloc[split_idx:].copy()
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 创建内部训练集目录结构
        internal_dir = output_path / 'internal_train'
        internal_dir.mkdir(exist_ok=True)
        (internal_dir / 'train').mkdir(exist_ok=True)
        (internal_dir / 'train' / 'oct').mkdir(exist_ok=True)
        (internal_dir / 'train' / 'col').mkdir(exist_ok=True)
        (internal_dir / 'test').mkdir(exist_ok=True)
        (internal_dir / 'test' / 'oct').mkdir(exist_ok=True)
        (internal_dir / 'test' / 'col').mkdir(exist_ok=True)
        
        # 创建外部验证集目录结构
        external_dir = output_path / 'external_validation'
        external_dir.mkdir(exist_ok=True)
        (external_dir / 'oct').mkdir(exist_ok=True)
        (external_dir / 'col').mkdir(exist_ok=True)
        
        # 保存标签文件
        internal_data.to_csv(internal_dir / 'train_labels.csv', index=False)
        external_data.to_csv(external_dir / 'external_labels.csv', index=False)
        
        # 复制图像文件
        print("\n📁 复制图像文件...")
        self._copy_images(internal_data, self.data_root, internal_dir / 'train')
        self._copy_images(external_data, self.data_root, external_dir)
        
        # 生成统计报告
        stats = {
            'split_method': 'time_split',
            'split_date': split_date if split_date else f'ratio_{split_ratio}',
            'internal_train': {
                'total_samples': len(internal_data),
                'positive': int(internal_data['label'].sum()) if 'label' in internal_data.columns else 0,
                'negative': int((internal_data['label'] == 0).sum()) if 'label' in internal_data.columns else 0,
                'date_range': {
                    'start': internal_data['date'].min().strftime('%Y-%m-%d') if len(internal_data) > 0 else None,
                    'end': internal_data['date'].max().strftime('%Y-%m-%d') if len(internal_data) > 0 else None
                }
            },
            'external_validation': {
                'total_samples': len(external_data),
                'positive': int(external_data['label'].sum()) if 'label' in external_data.columns else 0,
                'negative': int((external_data['label'] == 0).sum()) if 'label' in external_data.columns else 0,
                'date_range': {
                    'start': external_data['date'].min().strftime('%Y-%m-%d') if len(external_data) > 0 else None,
                    'end': external_data['date'].max().strftime('%Y-%m-%d') if len(external_data) > 0 else None
                }
            },
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存统计报告
        stats_file = output_path / 'split_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 时间分割完成！")
        print(f"   内部训练集: {len(internal_data)} 样本")
        print(f"   外部验证集: {len(external_data)} 样本")
        print(f"   结果保存在: {output_dir}")
        
        return stats
    
    def _copy_images(self, df: pd.DataFrame, source_root: Path, target_dir: Path):
        """复制图像文件（创建符号链接以节省空间）"""
        print(f"   复制 {len(df)} 个样本的图像文件...")
        
        copied = 0
        for idx, row in df.iterrows():
            oct_id = str(row['OCT'])
            
            # 复制OCT图像
            source_oct = source_root / 'train' / 'oct' / oct_id
            if not source_oct.exists():
                source_oct = source_root / 'test' / 'oct' / oct_id
            
            if source_oct.exists():
                target_oct = target_dir / 'oct' / oct_id
                if not target_oct.exists():
                    target_oct.parent.mkdir(parents=True, exist_ok=True)
                    # 创建符号链接（节省空间）
                    try:
                        target_oct.symlink_to(source_oct.resolve())
                    except:
                        # 如果符号链接失败，则复制
                        shutil.copytree(source_oct, target_oct, dirs_exist_ok=True)
                copied += 1
            
            # 复制阴道镜图像
            # 从OCT ID提取患者ID（假设格式为 日期_姓名）
            patient_id = oct_id.split('_')[1] if '_' in oct_id else oct_id
            
            source_col = source_root / 'train' / 'col' / patient_id
            if not source_col.exists():
                source_col = source_root / 'test' / 'col' / patient_id
            
            if source_col.exists():
                target_col = target_dir / 'col' / patient_id
                if not target_col.exists():
                    target_col.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        target_col.symlink_to(source_col.resolve())
                    except:
                        shutil.copytree(source_col, target_col, dirs_exist_ok=True)
        
        print(f"   ✅ 已处理 {copied} 个样本")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='创建内部外部验证数据集')
    parser.add_argument('--method', type=str, required=True,
                       choices=['center', 'time'],
                       help='分割方法：center（中心分割）或time（时间分割）')
    parser.add_argument('--external_centers', type=str, nargs='+',
                       default=['恩施', '荆州'],
                       help='外部验证中心列表（仅用于center方法）')
    parser.add_argument('--split_date', type=str, default=None,
                       help='分割日期（格式：2023-06-01，仅用于time方法）')
    parser.add_argument('--split_ratio', type=float, default=0.2,
                       help='外部验证集比例（仅用于time方法，当split_date为None时使用）')
    parser.add_argument('--data_root', type=str, default='5centers_multi',
                       help='数据根目录')
    parser.add_argument('--output_dir', type=str,
                       default='5centers_multi_internal_external',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建分割器
    creator = InternalExternalValidationCreator(args.data_root)
    
    # 分析中心分布
    distribution = creator.analyze_center_distribution()
    
    # 执行分割
    if args.method == 'center':
        stats = creator.create_center_split(
            args.external_centers,
            args.output_dir
        )
    else:  # time
        stats = creator.create_time_split(
            args.split_date,
            args.split_ratio,
            args.output_dir
        )
    
    print(f"\n✅ 分割完成！")
    print(f"   统计报告: {args.output_dir}/split_statistics.json")


if __name__ == '__main__':
    main()

