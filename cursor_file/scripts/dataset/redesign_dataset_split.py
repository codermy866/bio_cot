#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件生成信息:
- 生成时间: 2025-12-25 10:35:08 (使用nvidia-smi获取CUDA时间)
- 生成需求: 用户要求作为专业的学术论文技术专家，重新设计合理的数据集划分方案
- 生成原因: 当前数据集划分存在严重问题（类别分布不一致），需要重新划分以符合SCI论文标准
- 相关任务: 数据集重新划分，SCI论文准备

文件功能: 重新设计并实现符合SCI论文标准的数据集划分方案，确保类别分布一致、样本量充足、中心代表性好
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import subprocess
import datetime
from sklearn.model_selection import train_test_split
import json

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
    Identify medical center from OCT ID
    
    Returns:
        (center_name, center_code) tuple
    """
    oct_str = str(oct_id)
    if 'M22105' in oct_str:
        return 'Enshi', 'M22105'
    elif 'M22102' in oct_str:
        return 'Xiangyang', 'M22102'
    elif 'M22104' in oct_str or 'M22101' in oct_str:
        return 'Shiyan', 'M22104'
    elif 'M0008' in oct_str:
        return 'Jingzhou', 'M0008'
    elif 'M20203' in oct_str or 'M20105' in oct_str:
        return 'Wuda', 'M20203'
    return 'Unknown', 'Unknown'

class ScientificDatasetSplitter:
    """
    Scientific dataset splitter for SCI paper standards
    
    Requirements:
    1. Class distribution consistency between internal and external sets
    2. Sufficient sample size (external ≥ 150)
    3. Center independence
    4. Balanced representation
    """
    
    def __init__(self, data_root: Path):
        """
        Args:
            data_root: Path to original dataset
        """
        self.data_root = Path(data_root)
        self.train_labels_file = self.data_root / 'train_labels.csv'
        self.test_labels_file = self.data_root / 'test_labels.csv'
        
    def load_all_data(self) -> pd.DataFrame:
        """Load all data from train and test"""
        print("="*80)
        print("Loading original dataset...")
        print("="*80)
        
        train_df = pd.read_csv(self.train_labels_file)
        test_df = pd.read_csv(self.test_labels_file)
        
        # Add source marker
        train_df['source'] = 'train'
        test_df['source'] = 'test'
        
        # Combine
        all_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Identify centers
        all_df['center_name'], all_df['center_code'] = zip(*all_df['OCT'].apply(identify_center))
        
        print(f"✅ Total samples: {len(all_df)}")
        print(f"✅ Positive: {(all_df['label'] == 1).sum()} ({(all_df['label'] == 1).sum()/len(all_df)*100:.1f}%)")
        print(f"✅ Negative: {(all_df['label'] == 0).sum()} ({(all_df['label'] == 0).sum()/len(all_df)*100:.1f}%)")
        
        return all_df
    
    def analyze_center_distribution(self, df: pd.DataFrame):
        """Analyze center distribution"""
        print("\n" + "="*80)
        print("Center Distribution Analysis")
        print("="*80)
        
        for center in sorted(df['center_name'].unique()):
            if center == 'Unknown':
                continue
            center_data = df[df['center_name'] == center]
            pos = (center_data['label'] == 1).sum()
            neg = (center_data['label'] == 0).sum()
            total = len(center_data)
            pos_rate = pos / total * 100
            
            print(f"\n{center}:")
            print(f"  Total: {total} samples")
            print(f"  Positive: {pos} ({pos_rate:.1f}%)")
            print(f"  Negative: {neg} ({100-pos_rate:.1f}%)")
    
    def design_split_strategy(self, df: pd.DataFrame) -> dict:
        """
        Design optimal split strategy
        
        Strategy:
        1. Use Jingzhou as external test set (70 samples, 32.9% positive, close to overall 32.7%)
        2. Use Enshi, Xiangyang, Shiyan, Wuda as internal development set
        3. Ensure class distribution consistency
        """
        print("\n" + "="*80)
        print("Designing Optimal Split Strategy")
        print("="*80)
        
        # Calculate overall positive rate
        overall_pos_rate = (df['label'] == 1).sum() / len(df)
        
        # Strategy: Jingzhou as external test set (best match to overall distribution)
        external_centers = ['Jingzhou']
        internal_centers = ['Enshi', 'Xiangyang', 'Shiyan', 'Wuda']
        
        # Split data
        external_data = df[df['center_name'].isin(external_centers)].copy()
        internal_data = df[df['center_name'].isin(internal_centers)].copy()
        
        # Check distributions
        external_pos_rate = (external_data['label'] == 1).sum() / len(external_data)
        internal_pos_rate = (internal_data['label'] == 1).sum() / len(internal_data)
        
        print(f"\n📊 Split Strategy:")
        print(f"  External test centers: {external_centers}")
        print(f"  Internal dev centers: {internal_centers}")
        print(f"\n📈 Distribution:")
        print(f"  Overall positive rate: {overall_pos_rate*100:.1f}%")
        print(f"  External positive rate: {external_pos_rate*100:.1f}% (difference: {abs(external_pos_rate - overall_pos_rate)*100:.1f} pp)")
        print(f"  Internal positive rate: {internal_pos_rate*100:.1f}% (difference: {abs(internal_pos_rate - overall_pos_rate)*100:.1f} pp)")
        
        # Check sample sizes
        print(f"\n📦 Sample Sizes:")
        print(f"  External test set: {len(external_data)} samples")
        print(f"  Internal dev set: {len(internal_data)} samples")
        
        # Check if external set has both classes
        external_pos = (external_data['label'] == 1).sum()
        external_neg = (external_data['label'] == 0).sum()
        
        if external_pos == 0 or external_neg == 0:
            print(f"\n⚠️  WARNING: External test set missing one class!")
            print(f"  Positive: {external_pos}, Negative: {external_neg}")
        else:
            print(f"  ✅ External test set has both classes (Pos: {external_pos}, Neg: {external_neg})")
        
        return {
            'external_centers': external_centers,
            'internal_centers': internal_centers,
            'external_data': external_data,
            'internal_data': internal_data,
            'overall_pos_rate': overall_pos_rate,
            'external_pos_rate': external_pos_rate,
            'internal_pos_rate': internal_pos_rate
        }
    
    def split_internal_data(self, internal_data: pd.DataFrame, val_ratio: float = 0.2, 
                           random_state: int = 42) -> tuple:
        """
        Split internal data into train and validation sets
        
        Uses stratified sampling to maintain class distribution
        """
        print("\n" + "="*80)
        print("Splitting Internal Development Set")
        print("="*80)
        
        # Stratified split to maintain class distribution
        train_df, val_df = train_test_split(
            internal_data,
            test_size=val_ratio,
            random_state=random_state,
            stratify=internal_data['label']
        )
        
        train_pos_rate = (train_df['label'] == 1).sum() / len(train_df)
        val_pos_rate = (val_df['label'] == 1).sum() / len(val_df)
        
        print(f"\n✅ Split Results:")
        print(f"  Train set: {len(train_df)} samples (positive rate: {train_pos_rate*100:.1f}%)")
        print(f"  Validation set: {len(val_df)} samples (positive rate: {val_pos_rate*100:.1f}%)")
        
        return train_df, val_df
    
    def save_split_results(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                          external_df: pd.DataFrame, output_dir: Path, strategy: dict):
        """Save split results"""
        print("\n" + "="*80)
        print("Saving Split Results")
        print("="*80)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Columns to save
        columns_to_save = ['ID', 'OCT', 'AGE', 'HPV清洗', 'TCT清洗', 'label']
        
        # Save train set
        train_output = train_df[columns_to_save].copy()
        train_file = output_dir / 'train_labels.csv'
        train_output.to_csv(train_file, index=False, encoding='utf-8-sig')
        print(f"✅ Train set saved: {train_file} ({len(train_output)} samples)")
        
        # Save validation set
        val_output = val_df[columns_to_save].copy()
        val_file = output_dir / 'val_labels.csv'
        val_output.to_csv(val_file, index=False, encoding='utf-8-sig')
        print(f"✅ Validation set saved: {val_file} ({len(val_output)} samples)")
        
        # Save external test set
        external_output = external_df[columns_to_save].copy()
        external_file = output_dir / 'external_test_labels.csv'
        external_output.to_csv(external_file, index=False, encoding='utf-8-sig')
        print(f"✅ External test set saved: {external_file} ({len(external_output)} samples)")
        
        # Save statistics
        stats = {
            'split_date': get_timestamp(),
            'total_samples': len(train_df) + len(val_df) + len(external_df),
            'overall_positive_rate': float(strategy['overall_pos_rate']),
            'split_strategy': {
                'external_centers': strategy['external_centers'],
                'internal_centers': strategy['internal_centers'],
                'external_positive_rate': float(strategy['external_pos_rate']),
                'internal_positive_rate': float(strategy['internal_pos_rate']),
                'distribution_difference_pp': float(abs(strategy['external_pos_rate'] - strategy['overall_pos_rate']) * 100)
            },
            'train_set': {
                'samples': len(train_df),
                'positive': int((train_df['label'] == 1).sum()),
                'negative': int((train_df['label'] == 0).sum()),
                'positive_rate': float((train_df['label'] == 1).sum() / len(train_df))
            },
            'validation_set': {
                'samples': len(val_df),
                'positive': int((val_df['label'] == 1).sum()),
                'negative': int((val_df['label'] == 0).sum()),
                'positive_rate': float((val_df['label'] == 1).sum() / len(val_df))
            },
            'external_test_set': {
                'samples': len(external_df),
                'positive': int((external_df['label'] == 1).sum()),
                'negative': int((external_df['label'] == 0).sum()),
                'positive_rate': float((external_df['label'] == 1).sum() / len(external_df))
            },
            'sci_paper_compliance': {
                'class_distribution_consistent': bool(abs(strategy['external_pos_rate'] - strategy['overall_pos_rate']) < 0.05),
                'external_sample_size_sufficient': bool(len(external_df) >= 150),
                'external_has_both_classes': bool((external_df['label'] == 1).sum() > 0 and (external_df['label'] == 0).sum() > 0),
                'center_independence': True
            }
        }
        
        stats_file = output_dir / 'split_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"✅ Statistics saved: {stats_file}")
        
        return stats
    
    def print_summary(self, stats: dict):
        """Print summary of split results"""
        print("\n" + "="*80)
        print("SPLIT SUMMARY - SCI PAPER COMPLIANCE")
        print("="*80)
        
        print(f"\n📊 Dataset Distribution:")
        print(f"  Overall positive rate: {stats['overall_positive_rate']*100:.1f}%")
        print(f"  External test positive rate: {stats['split_strategy']['external_positive_rate']*100:.1f}%")
        print(f"  Internal dev positive rate: {stats['split_strategy']['internal_positive_rate']*100:.1f}%")
        print(f"  Distribution difference: {stats['split_strategy']['distribution_difference_pp']:.1f} percentage points")
        
        print(f"\n📦 Sample Sizes:")
        print(f"  Train: {stats['train_set']['samples']} samples")
        print(f"  Validation: {stats['validation_set']['samples']} samples")
        print(f"  External test: {stats['external_test_set']['samples']} samples")
        print(f"  Total: {stats['total_samples']} samples")
        
        print(f"\n✅ SCI Paper Compliance:")
        compliance = stats['sci_paper_compliance']
        print(f"  Class distribution consistent: {'✅' if compliance['class_distribution_consistent'] else '❌'}")
        print(f"  External sample size sufficient: {'✅' if compliance['external_sample_size_sufficient'] else '⚠️ '}")
        print(f"  External has both classes: {'✅' if compliance['external_has_both_classes'] else '❌'}")
        print(f"  Center independence: {'✅' if compliance['center_independence'] else '❌'}")
        
        if all(compliance.values()):
            print(f"\n🎉 SUCCESS: Dataset split meets SCI paper standards!")
        else:
            print(f"\n⚠️  WARNING: Some requirements not met. Please review.")

def main():
    """Main function"""
    print("="*80)
    print("Scientific Dataset Splitter for SCI Paper")
    print("="*80)
    
    # Paths
    original_data_root = Path('/data2/hmy/5Center_datas/5centers_multi')
    output_dir = Path('/data2/hmy/5Center_datas/5centers_multi_internal_external_final_scientific')
    
    # Create splitter
    splitter = ScientificDatasetSplitter(original_data_root)
    
    # Load data
    all_df = splitter.load_all_data()
    
    # Analyze centers
    splitter.analyze_center_distribution(all_df)
    
    # Design split strategy
    strategy = splitter.design_split_strategy(all_df)
    
    # Split internal data
    train_df, val_df = splitter.split_internal_data(
        strategy['internal_data'], 
        val_ratio=0.2, 
        random_state=42
    )
    
    # Save results
    stats = splitter.save_split_results(
        train_df, 
        val_df, 
        strategy['external_data'],
        output_dir,
        strategy
    )
    
    # Print summary
    splitter.print_summary(stats)
    
    print("\n" + "="*80)
    print("✅ Dataset split completed!")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 Statistics file: {output_dir / 'split_statistics.json'}")
    print(f"\n💡 Next steps:")
    print(f"  1. Review the split statistics")
    print(f"  2. Update data paths in training scripts")
    print(f"  3. Recreate symbolic links if needed")

if __name__ == '__main__':
    main()

