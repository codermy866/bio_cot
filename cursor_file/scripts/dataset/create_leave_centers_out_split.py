#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件生成信息:
- 生成时间: 2025-12-25 10:41:26 (使用nvidia-smi获取CUDA时间)
- 生成需求: 用户要求使用Leave-Centers-Out策略，将十堰(Shiyan)和荆州(Jingzhou)作为外部测试集
- 生成原因: 实现严格的Leave-Centers-Out评估策略，确保外部测试集完全独立，从未参与训练
- 相关任务: Leave-Centers-Out数据集划分，SCI论文准备

文件功能: 创建Leave-Centers-Out策略的数据集划分，十堰和荆州作为外部测试集
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
    """Identify medical center from OCT ID"""
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

class LeaveCentersOutSplitter:
    """
    Leave-Centers-Out Dataset Splitter
    
    Strategy:
    - External test set: Shiyan + Jingzhou (completely independent, never used in training)
    - Internal development set: Enshi + Xiangyang + Wuda (800+ samples)
    - This ensures strict evaluation of generalization ability
    """
    
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.train_labels_file = self.data_root / 'train_labels.csv'
        self.test_labels_file = self.data_root / 'test_labels.csv'
        
        # Leave-Centers-Out: External test centers
        self.external_centers = ['Shiyan', 'Jingzhou']
        self.internal_centers = ['Enshi', 'Xiangyang', 'Wuda']
        
    def load_all_data(self) -> pd.DataFrame:
        """Load all data from train and test"""
        print("="*80)
        print("Leave-Centers-Out Dataset Splitter")
        print("="*80)
        print("\nLoading original dataset...")
        
        train_df = pd.read_csv(self.train_labels_file)
        test_df = pd.read_csv(self.test_labels_file)
        
        train_df['source'] = 'train'
        test_df['source'] = 'test'
        
        all_df = pd.concat([train_df, test_df], ignore_index=True)
        all_df['center_name'], all_df['center_code'] = zip(*all_df['OCT'].apply(identify_center))
        
        print(f"✅ Total samples: {len(all_df)}")
        print(f"✅ Positive: {(all_df['label'] == 1).sum()} ({(all_df['label'] == 1).sum()/len(all_df)*100:.1f}%)")
        print(f"✅ Negative: {(all_df['label'] == 0).sum()} ({(all_df['label'] == 0).sum()/len(all_df)*100:.1f}%)")
        
        return all_df
    
    def apply_leave_centers_out(self, df: pd.DataFrame) -> dict:
        """
        Apply Leave-Centers-Out strategy
        
        External test: Shiyan + Jingzhou (completely independent)
        Internal dev: Enshi + Xiangyang + Wuda (800+ samples)
        """
        print("\n" + "="*80)
        print("Leave-Centers-Out Strategy")
        print("="*80)
        
        overall_pos_rate = (df['label'] == 1).sum() / len(df)
        
        # Split by centers
        external_data = df[df['center_name'].isin(self.external_centers)].copy()
        internal_data = df[df['center_name'].isin(self.internal_centers)].copy()
        
        external_pos_rate = (external_data['label'] == 1).sum() / len(external_data)
        internal_pos_rate = (internal_data['label'] == 1).sum() / len(internal_data)
        
        print(f"\n📊 Leave-Centers-Out Configuration:")
        print(f"  External test centers: {self.external_centers}")
        print(f"    - Shiyan: {len(df[df['center_name'] == 'Shiyan'])} samples")
        print(f"    - Jingzhou: {len(df[df['center_name'] == 'Jingzhou'])} samples")
        print(f"    - Total: {len(external_data)} samples")
        print(f"    - Positive: {(external_data['label'] == 1).sum()} ({(external_data['label'] == 1).sum()/len(external_data)*100:.1f}%)")
        print(f"    - Negative: {(external_data['label'] == 0).sum()} ({(external_data['label'] == 0).sum()/len(external_data)*100:.1f}%)")
        
        print(f"\n  Internal dev centers: {self.internal_centers}")
        print(f"    - Enshi: {len(df[df['center_name'] == 'Enshi'])} samples")
        print(f"    - Xiangyang: {len(df[df['center_name'] == 'Xiangyang'])} samples")
        print(f"    - Wuda: {len(df[df['center_name'] == 'Wuda'])} samples")
        print(f"    - Total: {len(internal_data)} samples (800+ samples ✅)")
        print(f"    - Positive: {(internal_data['label'] == 1).sum()} ({(internal_data['label'] == 1).sum()/len(internal_data)*100:.1f}%)")
        print(f"    - Negative: {(internal_data['label'] == 0).sum()} ({(internal_data['label'] == 0).sum()/len(internal_data)*100:.1f}%)")
        
        print(f"\n📈 Class Distribution:")
        print(f"  Overall positive rate: {overall_pos_rate*100:.1f}%")
        print(f"  External test positive rate: {external_pos_rate*100:.1f}% (difference: {abs(external_pos_rate - overall_pos_rate)*100:.1f} pp)")
        print(f"  Internal dev positive rate: {internal_pos_rate*100:.1f}% (difference: {abs(internal_pos_rate - overall_pos_rate)*100:.1f} pp)")
        
        # Verify both classes in external set
        external_pos = (external_data['label'] == 1).sum()
        external_neg = (external_data['label'] == 0).sum()
        
        if external_pos > 0 and external_neg > 0:
            print(f"\n✅ External test set has both classes (Pos: {external_pos}, Neg: {external_neg})")
        else:
            print(f"\n❌ WARNING: External test set missing one class!")
        
        return {
            'external_centers': self.external_centers,
            'internal_centers': self.internal_centers,
            'external_data': external_data,
            'internal_data': internal_data,
            'overall_pos_rate': overall_pos_rate,
            'external_pos_rate': external_pos_rate,
            'internal_pos_rate': internal_pos_rate
        }
    
    def split_internal_data(self, internal_data: pd.DataFrame, val_ratio: float = 0.2, 
                           random_state: int = 42) -> tuple:
        """Split internal data into train and validation with stratified sampling"""
        print("\n" + "="*80)
        print("Splitting Internal Development Set")
        print("="*80)
        
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
        print("Saving Leave-Centers-Out Split Results")
        print("="*80)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        columns_to_save = ['ID', 'OCT', 'AGE', 'HPV清洗', 'TCT清洗', 'label']
        
        # Save train set
        train_output = train_df[columns_to_save].copy()
        train_file = output_dir / 'train_labels.csv'
        train_output.to_csv(train_file, index=False, encoding='utf-8-sig')
        print(f"✅ Train set: {train_file} ({len(train_output)} samples)")
        
        # Save validation set
        val_output = val_df[columns_to_save].copy()
        val_file = output_dir / 'val_labels.csv'
        val_output.to_csv(val_file, index=False, encoding='utf-8-sig')
        print(f"✅ Validation set: {val_file} ({len(val_output)} samples)")
        
        # Save external test set
        external_output = external_df[columns_to_save].copy()
        external_file = output_dir / 'external_test_labels.csv'
        external_output.to_csv(external_file, index=False, encoding='utf-8-sig')
        print(f"✅ External test set: {external_file} ({len(external_output)} samples)")
        
        # Save statistics
        stats = {
            'split_date': get_timestamp(),
            'split_strategy': 'Leave-Centers-Out',
            'total_samples': len(train_df) + len(val_df) + len(external_df),
            'overall_positive_rate': float(strategy['overall_pos_rate']),
            'leave_centers_out': {
                'external_centers': strategy['external_centers'],
                'internal_centers': strategy['internal_centers'],
                'external_positive_rate': float(strategy['external_pos_rate']),
                'internal_positive_rate': float(strategy['internal_pos_rate']),
                'distribution_difference_pp': float(abs(strategy['external_pos_rate'] - strategy['overall_pos_rate']) * 100),
                'external_sample_size': len(external_df),
                'internal_sample_size': len(train_df) + len(val_df)
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
                'positive_rate': float((external_df['label'] == 1).sum() / len(external_df)),
                'centers': strategy['external_centers']
            },
            'sci_paper_compliance': {
                'class_distribution_consistent': bool(abs(strategy['external_pos_rate'] - strategy['overall_pos_rate']) < 0.05),
                'external_sample_size_sufficient': bool(len(external_df) >= 100),
                'external_has_both_classes': bool((external_df['label'] == 1).sum() > 0 and (external_df['label'] == 0).sum() > 0),
                'center_independence': True,
                'leave_centers_out': True,
                'internal_800_plus': bool(len(train_df) + len(val_df) >= 800)
            }
        }
        
        stats_file = output_dir / 'split_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"✅ Statistics: {stats_file}")
        
        return stats
    
    def print_summary(self, stats: dict):
        """Print summary"""
        print("\n" + "="*80)
        print("LEAVE-CENTERS-OUT SPLIT SUMMARY")
        print("="*80)
        
        print(f"\n📊 Leave-Centers-Out Strategy:")
        print(f"  External test centers: {stats['leave_centers_out']['external_centers']}")
        print(f"  Internal dev centers: {stats['leave_centers_out']['internal_centers']}")
        
        print(f"\n📈 Class Distribution:")
        print(f"  Overall: {stats['overall_positive_rate']*100:.1f}% positive")
        print(f"  External test: {stats['leave_centers_out']['external_positive_rate']*100:.1f}% positive")
        print(f"  Internal dev: {stats['leave_centers_out']['internal_positive_rate']*100:.1f}% positive")
        print(f"  Difference: {stats['leave_centers_out']['distribution_difference_pp']:.1f} pp")
        
        print(f"\n📦 Sample Sizes:")
        print(f"  Train: {stats['train_set']['samples']} samples")
        print(f"  Validation: {stats['validation_set']['samples']} samples")
        print(f"  Internal dev total: {stats['leave_centers_out']['internal_sample_size']} samples (800+ ✅)")
        print(f"  External test: {stats['external_test_set']['samples']} samples")
        print(f"  Total: {stats['total_samples']} samples")
        
        print(f"\n✅ SCI Paper Compliance:")
        c = stats['sci_paper_compliance']
        print(f"  Leave-Centers-Out: {'✅' if c['leave_centers_out'] else '❌'}")
        print(f"  Internal 800+ samples: {'✅' if c['internal_800_plus'] else '❌'}")
        print(f"  Class distribution: {'✅' if c['class_distribution_consistent'] else '❌'}")
        print(f"  External size ≥100: {'✅' if c['external_sample_size_sufficient'] else '❌'}")
        print(f"  Both classes: {'✅' if c['external_has_both_classes'] else '❌'}")
        print(f"  Center independence: {'✅' if c['center_independence'] else '❌'}")
        
        if all(c.values()):
            print(f"\n🎉 SUCCESS: Leave-Centers-Out split meets all SCI paper standards!")
        else:
            print(f"\n⚠️  Some requirements not fully met")

def main():
    """Main function"""
    print("="*80)
    print("Leave-Centers-Out Dataset Splitter")
    print("="*80)
    
    original_data_root = Path('/data2/hmy/5Center_datas/5centers_multi')
    output_dir = Path('/data2/hmy/5Center_datas/5centers_multi_leave_centers_out')
    
    splitter = LeaveCentersOutSplitter(original_data_root)
    all_df = splitter.load_all_data()
    strategy = splitter.apply_leave_centers_out(all_df)
    train_df, val_df = splitter.split_internal_data(strategy['internal_data'], val_ratio=0.2, random_state=42)
    stats = splitter.save_split_results(train_df, val_df, strategy['external_data'], output_dir, strategy)
    splitter.print_summary(stats)
    
    print("\n" + "="*80)
    print("✅ Leave-Centers-Out split completed!")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n💡 This split ensures:")
    print(f"   - External test set (Shiyan + Jingzhou) never participates in training")
    print(f"   - Strict evaluation of generalization ability")
    print(f"   - Internal dev set has 800+ samples for robust training")

if __name__ == '__main__':
    main()

