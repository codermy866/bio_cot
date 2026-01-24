#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据划分检查脚本
确保训练/验证集不包含外部测试集的样本
"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple


class DataSplitChecker:
    """数据划分检查器"""
    
    def __init__(self):
        # 外部测试集中心代码
        self.external_center_codes = ['M0008', 'M20203', 'M20105']  # 荆州、武大
        
        # 内部开发集中心代码
        self.internal_center_codes = ['M22105', 'M22104', 'M22101', 'M22102']  # 恩施、十堰、襄阳
    
    def identify_center(self, oct_id: str) -> Tuple[str, str]:
        """
        从OCT ID识别中心
        
        Returns:
            (center_name, center_type): 中心名称和类型（internal/external/unknown）
        """
        if pd.isna(oct_id):
            return 'Unknown', 'unknown'
        
        oct_str = str(oct_id)
        
        # 检查外部测试集中心
        if 'M0008' in oct_str:
            return '荆州', 'external'
        elif 'M20203' in oct_str or 'M20105' in oct_str:
            return '武大', 'external'
        
        # 检查内部开发集中心
        if 'M22105' in oct_str:
            return '恩施', 'internal'
        elif 'M22104' in oct_str or 'M22101' in oct_str:
            return '十堰', 'internal'
        elif 'M22102' in oct_str:
            return '襄阳', 'internal'
        
        return 'Unknown', 'unknown'
    
    def check_labels_file(self, labels_file: Path, expected_type: str = 'internal') -> dict:
        """
        检查标签文件
        
        Args:
            labels_file: 标签文件路径
            expected_type: 期望的类型（'internal' 或 'external'）
        
        Returns:
            检查结果字典
        """
        if not labels_file.exists():
            return {
                'valid': False,
                'error': f'文件不存在: {labels_file}'
            }
        
        df = pd.read_csv(labels_file)
        
        # 识别中心
        center_info = df['OCT'].apply(self.identify_center)
        df['center_name'] = [x[0] for x in center_info]
        df['center_type'] = [x[1] for x in center_info]
        
        # 统计
        external_count = (df['center_type'] == 'external').sum()
        internal_count = (df['center_type'] == 'internal').sum()
        unknown_count = (df['center_type'] == 'unknown').sum()
        
        # 中心分布
        center_dist = df['center_name'].value_counts().to_dict()
        
        # 检查是否符合预期
        if expected_type == 'internal':
            is_valid = external_count == 0
            error_msg = None if is_valid else f'⚠️ 错误：发现 {external_count} 个外部测试集样本！'
        elif expected_type == 'external':
            is_valid = internal_count == 0
            error_msg = None if is_valid else f'⚠️ 错误：发现 {internal_count} 个内部开发集样本！'
        else:
            is_valid = True
            error_msg = None
        
        return {
            'valid': is_valid,
            'file': str(labels_file),
            'total_samples': len(df),
            'external_count': external_count,
            'internal_count': internal_count,
            'unknown_count': unknown_count,
            'center_distribution': center_dist,
            'positive_count': int(df['label'].sum()),
            'negative_count': int((df['label'] == 0).sum()),
            'positive_rate': float(df['label'].mean()),
            'error': error_msg
        }
    
    def check_all_splits(self, data_dir: Path) -> dict:
        """检查所有数据划分"""
        results = {}
        
        # 检查训练集
        train_file = data_dir / 'train_labels.csv'
        if train_file.exists():
            results['train'] = self.check_labels_file(train_file, expected_type='internal')
        
        # 检查验证集
        val_file = data_dir / 'val_labels.csv'
        if val_file.exists():
            results['val'] = self.check_labels_file(val_file, expected_type='internal')
        
        # 检查外部测试集
        external_file = data_dir / 'external_test_labels.csv'
        if external_file.exists():
            results['external_test'] = self.check_labels_file(external_file, expected_type='external')
        
        return results
    
    def print_check_results(self, results: dict):
        """打印检查结果"""
        print("=" * 80)
        print("🔍 数据划分检查结果")
        print("=" * 80)
        
        for split_name, result in results.items():
            print(f"\n📁 {split_name.upper()}")
            print(f"  文件: {result['file']}")
            print(f"  总样本数: {result['total_samples']}")
            print(f"  阳性样本: {result['positive_count']} ({result['positive_rate']*100:.2f}%)")
            print(f"  阴性样本: {result['negative_count']}")
            print(f"  中心分布: {result['center_distribution']}")
            
            if result['external_count'] > 0:
                print(f"  ⚠️  外部测试集样本: {result['external_count']}")
            if result['internal_count'] > 0:
                print(f"  ⚠️  内部开发集样本: {result['internal_count']}")
            if result['unknown_count'] > 0:
                print(f"  ⚠️  未知中心样本: {result['unknown_count']}")
            
            if result['valid']:
                print(f"  ✅ 检查通过")
            else:
                print(f"  ❌ 检查失败: {result['error']}")
        
        # 总结
        all_valid = all(r['valid'] for r in results.values())
        print("\n" + "=" * 80)
        if all_valid:
            print("✅ 所有数据划分检查通过！")
        else:
            print("❌ 数据划分存在问题，请检查！")
        print("=" * 80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='检查数据划分')
    parser.add_argument('--data_dir', type=str,
                       default='/data2/hmy/5Center_datas/5centers_multi_internal_external_final',
                       help='数据目录')
    
    args = parser.parse_args()
    
    checker = DataSplitChecker()
    results = checker.check_all_splits(Path(args.data_dir))
    checker.print_check_results(results)
    
    # 返回退出码
    all_valid = all(r['valid'] for r in results.values())
    exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()

