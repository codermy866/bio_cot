#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建合并的分类版本（3分类或4分类）
解决类别不平衡和样本不足问题
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt

class ClassificationMerger:
    def __init__(self, data_path='5centers_multi'):
        self.data_path = data_path
        
    def analyze_current_distribution(self):
        """分析当前分类分布"""
        print("📊 分析当前分类分布")
        print("=" * 60)
        
        train_df = pd.read_csv(f'{self.data_path}/train_labels.csv')
        test_df = pd.read_csv(f'{self.data_path}/test_labels.csv')
        
        # 创建5分类标签
        train_df_5class = self.create_tct_5class_labels(train_df)
        test_df_5class = self.create_tct_5class_labels(test_df)
        
        train_dist = train_df_5class['tct_5class'].value_counts().sort_index()
        
        class_info = {
            0: ('NILM', '正常/阴性'),
            1: ('ASC-US', '非典型鳞状细胞-意义不明'),
            2: ('LSIL', '低度鳞状上皮内病变'),
            3: ('HSIL', '高度鳞状上皮内病变'),
            4: ('Cancer', '癌变/恶性')
        }
        
        print("\n📋 当前5分类分布:")
        total = len(train_df_5class)
        for idx in range(5):
            count = train_dist.get(idx, 0)
            class_code, class_name = class_info[idx]
            percentage = count / total * 100
            print(f"  类别{idx} ({class_code}): {count}样本 ({percentage:.1f}%) - {class_name}")
        
        return train_df_5class, test_df_5class, class_info
    
    def create_tct_5class_labels(self, df):
        """创建TCT 5分类标签"""
        def map_tct_to_5class(row):
            tct_val = str(row.get('TCT清洗', '')) if pd.notna(row.get('TCT清洗')) else 'NILM'
            label_val = row.get('label', 0)
            
            if tct_val == 'NILM':
                return 0
            elif tct_val == 'ASC-US':
                return 1
            elif tct_val == 'LSIL':
                return 2
            elif tct_val == 'HSIL':
                return 3
            elif tct_val == '1' or tct_val in ['ASC-H', 'SCC'] or label_val == 1:
                return 4
            else:
                return 0
        
        df['tct_5class'] = df.apply(map_tct_to_5class, axis=1)
        return df
    
    def create_3class_labels(self, train_df, test_df):
        """创建3分类标签（推荐方案）"""
        print("\n🎯 创建3分类标签（推荐）")
        print("-" * 40)
        
        print("合并策略:")
        print("  类别0 (NILM) → 新类别0 (正常: 275样本)")
        print("  类别1+2 (ASC-US+LSIL) → 新类别1 (低度病变: 18样本)")
        print("  类别3+4 (HSIL+Cancer) → 新类别2 (高度病变: 492样本)")
        
        def map_to_3class(tct_5class):
            if tct_5class == 0:
                return 0  # 正常
            elif tct_5class in [1, 2]:
                return 1  # 低度病变
            elif tct_5class in [3, 4]:
                return 2  # 高度病变
            else:
                return 0
        
        train_df['merged_3class'] = train_df['tct_5class'].apply(map_to_3class)
        test_df['merged_3class'] = test_df['tct_5class'].apply(map_to_3class)
        
        # 统计分布
        train_dist = train_df['merged_3class'].value_counts().sort_index()
        test_dist = test_df['merged_3class'].value_counts().sort_index()
        
        print("\n📊 3分类分布:")
        class_names_3 = ['正常', '低度病变', '高度病变']
        for idx, name in enumerate(class_names_3):
            train_count = train_dist.get(idx, 0)
            test_count = test_dist.get(idx, 0)
            print(f"  类别{idx} ({name}): 训练{train_count}样本, 测试{test_count}样本")
        
        # 计算平衡性
        min_count = train_dist.min()
        max_count = train_dist.max()
        balance = min_count / max_count
        print(f"\n  平衡度: {balance:.3f} ({'较好' if balance > 0.05 else '需改进'})")
        
        return train_df, test_df
    
    def create_4class_labels(self, train_df, test_df):
        """创建4分类标签"""
        print("\n🎯 创建4分类标签")
        print("-" * 40)
        
        print("合并策略:")
        print("  类别0 (NILM) → 新类别0 (正常: 275样本)")
        print("  类别1 (ASC-US) → 新类别1 (轻度异常: 14样本)")
        print("  类别2+3 (LSIL+HSIL) → 新类别2 (中高度病变: 8样本)")
        print("  类别4 (Cancer) → 新类别3 (恶性病变: 488样本)")
        
        def map_to_4class(tct_5class):
            if tct_5class == 0:
                return 0  # 正常
            elif tct_5class == 1:
                return 1  # 轻度异常
            elif tct_5class in [2, 3]:
                return 2  # 中高度病变
            elif tct_5class == 4:
                return 3  # 恶性病变
            else:
                return 0
        
        train_df['merged_4class'] = train_df['tct_5class'].apply(map_to_4class)
        test_df['merged_4class'] = test_df['tct_5class'].apply(map_to_4class)
        
        # 统计分布
        train_dist = train_df['merged_4class'].value_counts().sort_index()
        
        print("\n📊 4分类分布:")
        class_names_4 = ['正常', '轻度异常', '中高度病变', '恶性病变']
        for idx, name in enumerate(class_names_4):
            count = train_dist.get(idx, 0)
            print(f"  类别{idx} ({name}): {count}样本")
        
        return train_df, test_df
    
    def compare_classifications(self):
        """比较不同分类方案"""
        print("\n📊 比较不同分类方案")
        print("=" * 60)
        
        train_df, test_df, class_info = self.analyze_current_distribution()
        
        # 3分类
        train_3class, test_3class = self.create_3class_labels(train_df.copy(), test_df.copy())
        
        # 4分类
        train_4class, test_4class = self.create_4class_labels(train_df.copy(), test_df.copy())
        
        # 保存结果
        self.save_merged_datasets(train_3class, test_3class, '3class')
        self.save_merged_datasets(train_4class, test_4class, '4class')
        
        # 绘制对比图
        self.plot_comparison(train_df, train_3class, train_4class)
        
        print("\n✅ 类别合并完成！")
        print("  已创建3分类和4分类数据集")
        
    def save_merged_datasets(self, train_df, test_df, version):
        """保存合并后的数据集"""
        output_dir = f'5centers_multi_{version}'
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/train', exist_ok=True)
        os.makedirs(f'{output_dir}/test', exist_ok=True)
        
        label_col = f'merged_{version}'
        
        # 保存标签文件
        train_df[[col for col in train_df.columns if col in ['ID', 'OCT', 'AGE', 'HPV清洗', 'TCT清洗', label_col]]].to_csv(
            f'{output_dir}/train/train_labels.csv', index=False
        )
        
        test_df[[col for col in test_df.columns if col in ['ID', 'OCT', 'AGE', 'HPV清洗', 'TCT清洗', label_col]]].to_csv(
            f'{output_dir}/test/test_labels.csv', index=False
        )
        
        # 保存标签映射
        if version == '3class':
            mapping = {
                '0': ('正常', 'NILM'),
                '1': ('低度病变', 'ASC-US+LSIL'),
                '2': ('高度病变', 'HSIL+Cancer')
            }
        else:  # 4class
            mapping = {
                '0': ('正常', 'NILM'),
                '1': ('轻度异常', 'ASC-US'),
                '2': ('中高度病变', 'LSIL+HSIL'),
                '3': ('恶性病变', 'Cancer')
            }
        
        with open(f'{output_dir}/label_mapping.json', 'w', encoding='utf-8') as f:
            json.dump({
                'version': version,
                'mapping': mapping,
                'description': f'合并后的{version}标签映射'
            }, f, indent=4, ensure_ascii=False)
        
        print(f"\n✅ {version}数据集已保存到: {output_dir}")
    
    def plot_comparison(self, train_5class, train_3class, train_4class):
        """绘制分类对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 5分类
        dist_5 = train_5class['tct_5class'].value_counts().sort_index()
        axes[0].bar(range(5), [dist_5.get(i, 0) for i in range(5)], color='steelblue')
        axes[0].set_title('5分类分布')
        axes[0].set_xlabel('类别')
        axes[0].set_ylabel('样本数')
        axes[0].set_xticks(range(5))
        axes[0].set_xticklabels(['NILM', 'ASC-US', 'LSIL', 'HSIL', 'Cancer'], rotation=45)
        
        # 3分类
        dist_3 = train_3class['merged_3class'].value_counts().sort_index()
        axes[1].bar(range(3), [dist_3.get(i, 0) for i in range(3)], color='coral')
        axes[1].set_title('3分类分布（推荐）')
        axes[1].set_xlabel('类别')
        axes[1].set_ylabel('样本数')
        axes[1].set_xticks(range(3))
        axes[1].set_xticklabels(['正常', '低度病变', '高度病变'])
        
        # 4分类
        dist_4 = train_4class['merged_4class'].value_counts().sort_index()
        axes[2].bar(range(4), [dist_4.get(i, 0) for i in range(4)], color='mediumseagreen')
        axes[2].set_title('4分类分布')
        axes[2].set_xlabel('类别')
        axes[2].set_ylabel('样本数')
        axes[2].set_xticks(range(4))
        axes[2].set_xticklabels(['正常', '轻度', '中高度', '恶性'], rotation=45)
        
        plt.tight_layout()
        plt.savefig('classification_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ 对比图已保存: classification_comparison.png")


def main():
    merger = ClassificationMerger()
    merger.compare_classifications()


if __name__ == "__main__":
    main()



