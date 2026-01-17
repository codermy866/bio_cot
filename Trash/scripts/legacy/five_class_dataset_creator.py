#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5分类数据集创建和标签映射
基于TCT结果的5分类方案，符合临床分级标准
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class FiveClassDatasetCreator:
    def __init__(self, data_path='5centers_multi'):
        self.data_path = data_path
        self.output_path = '5centers_multi_5class'
        
        # TCT 5分类标签映射（基于临床标准）
        self.tct_5class_mapping = {
            'NILM': 0,      # 正常/阴性
            'ASC-US': 1,    # 非典型鳞状细胞-意义不明
            'LSIL': 2,      # 低度鳞状上皮内病变
            'HSIL': 3,      # 高度鳞状上皮内病变
            'Cancer': 4     # 癌变/恶性
        }
        
        self.class_names = [
            'NILM(正常)',
            'ASC-US(非典型)', 
            'LSIL(低度病变)',
            'HSIL(高度病变)',
            'Cancer(癌变)'
        ]
        
    def create_tct_5class_labels(self, df):
        """创建基于TCT的5分类标签"""
        print("📊 创建TCT 5分类标签...")
        
        def map_tct_to_5class(tct_value, hpv_value, label_value):
            """智能映射TCT结果到5分类"""
            if pd.isna(tct_value) or tct_value == '':
                # 如果没有TCT信息，根据HPV和原始标签推断
                if str(hpv_value) == '1' and label_value == 1:
                    return 1  # HPV阳性且原始标签阳性 -> ASC-US
                elif label_value == 1:
                    return 2  # 原始标签阳性 -> LSIL
                else:
                    return 0  # 默认正常
                    
            tct_str = str(tct_value).strip()
            
            # 直接映射
            if tct_str == 'NILM':
                return 0
            elif tct_str == 'ASC-US':
                return 1
            elif tct_str == 'LSIL':
                return 2
            elif tct_str == 'HSIL':
                return 3
            elif tct_str in ['ASC-H', 'SCC']:  # 高度异常或鳞状细胞癌
                return 4
            elif tct_str == '1':  # 可能是癌变标记
                return 4
            else:
                # 其他情况根据HPV和原始标签推断
                if str(hpv_value) == '1' and label_value == 1:
                    return 1  # HPV阳性且原始标签阳性
                elif label_value == 1:
                    return 2  # 原始标签阳性
                else:
                    return 0  # 默认正常
        
        df['tct_5class'] = df.apply(
            lambda x: map_tct_to_5class(x['TCT清洗'], x['HPV清洗'], x['label']), 
            axis=1
        )
        
        return df
    
    def analyze_class_distribution(self, df, split_name=""):
        """分析类别分布"""
        print(f"\n📈 {split_name}类别分布分析:")
        
        class_dist = df['tct_5class'].value_counts().sort_index()
        total_samples = len(df)
        
        for class_id, count in class_dist.items():
            percentage = count / total_samples * 100
            print(f"  {self.class_names[class_id]}: {count}个样本 ({percentage:.1f}%)")
        
        # 计算类别平衡性
        min_count = class_dist.min()
        max_count = class_dist.max()
        balance_ratio = min_count / max_count if max_count > 0 else 0
        
        print(f"  类别平衡性: {balance_ratio:.3f}")
        print(f"  最少类别样本数: {min_count}")
        
        return class_dist, balance_ratio
    
    def create_balanced_dataset(self, df, min_samples_per_class=10):
        """创建平衡的数据集"""
        print(f"\n⚖️ 创建平衡数据集 (最少{min_samples_per_class}个样本/类)...")
        
        balanced_data = []
        
        for class_id in range(5):
            class_data = df[df['tct_5class'] == class_id]
            class_count = len(class_data)
            
            if class_count >= min_samples_per_class:
                balanced_data.append(class_data)
                print(f"  {self.class_names[class_id]}: 保留{class_count}个样本")
            else:
                print(f"  ⚠️ {self.class_names[class_id]}: 只有{class_count}个样本，跳过")
        
        if balanced_data:
            balanced_df = pd.concat(balanced_data, ignore_index=True)
            print(f"  平衡后总样本数: {len(balanced_df)}")
            return balanced_df
        else:
            print("  ❌ 无法创建平衡数据集")
            return df
    
    def split_dataset(self, df, test_size=0.2, stratify=True):
        """分割数据集"""
        print(f"\n✂️ 分割数据集 (测试集比例: {test_size})...")
        
        if stratify and len(df['tct_5class'].unique()) > 1:
            train_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                stratify=df['tct_5class'],
                random_state=42
            )
        else:
            train_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=42
            )
        
        print(f"  训练集: {len(train_df)} 样本")
        print(f"  测试集: {len(test_df)} 样本")
        
        return train_df, test_df
    
    def create_output_structure(self):
        """创建输出目录结构"""
        print(f"\n📁 创建输出目录结构...")
        
        os.makedirs(self.output_path, exist_ok=True)
        
        # 创建子目录
        subdirs = ['oct', 'col', 'cache']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_path, subdir), exist_ok=True)
        
        print(f"  输出目录: {self.output_path}")
    
    def save_datasets(self, train_df, test_df):
        """保存数据集"""
        print(f"\n💾 保存数据集...")
        
        # 保存标签文件
        train_df.to_csv(os.path.join(self.output_path, 'train_labels_5class.csv'), index=False)
        test_df.to_csv(os.path.join(self.output_path, 'test_labels_5class.csv'), index=False)
        
        # 保存标签映射
        mapping_info = {
            'class_mapping': self.tct_5class_mapping,
            'class_names': self.class_names,
            'num_classes': 5,
            'description': '基于TCT结果的5分类标签映射'
        }
        
        with open(os.path.join(self.output_path, 'label_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump(mapping_info, f, indent=4, ensure_ascii=False)
        
        print(f"  ✅ 训练集标签: {len(train_df)} 样本")
        print(f"  ✅ 测试集标签: {len(test_df)} 样本")
        print(f"  ✅ 标签映射文件已保存")
    
    def create_data_summary(self, train_df, test_df):
        """创建数据摘要"""
        print(f"\n📋 创建数据摘要...")
        
        summary = {
            'dataset_info': {
                'total_samples': len(train_df) + len(test_df),
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'num_classes': 5,
                'class_names': self.class_names
            },
            'train_distribution': train_df['tct_5class'].value_counts().sort_index().to_dict(),
            'test_distribution': test_df['tct_5class'].value_counts().sort_index().to_dict(),
            'center_distribution': self.analyze_center_distribution(train_df, test_df)
        }
        
        with open(os.path.join(self.output_path, 'dataset_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        print(f"  ✅ 数据摘要已保存")
    
    def analyze_center_distribution(self, train_df, test_df):
        """分析中心分布"""
        def identify_center(oct_id):
            if pd.isna(oct_id):
                return 'Unknown'
            oct_str = str(oct_id)
            if 'M22105' in oct_str:
                return '恩施'
            elif 'M0008' in oct_str:
                return '荆州'
            elif 'M22104' in oct_str or 'M22101' in oct_str:
                return '十堰'
            elif 'M20203' in oct_str or 'M20105' in oct_str:
                return '武大'
            elif 'M22102' in oct_str:
                return '襄阳'
            else:
                return 'Unknown'
        
        train_df['center'] = train_df['OCT'].apply(identify_center)
        test_df['center'] = test_df['OCT'].apply(identify_center)
        
        center_dist = {}
        for center in ['恩施', '荆州', '十堰', '武大', '襄阳']:
            train_center = train_df[train_df['center'] == center]
            test_center = test_df[test_df['center'] == center]
            
            center_dist[center] = {
                'train_samples': len(train_center),
                'test_samples': len(test_center),
                'train_class_dist': train_center['tct_5class'].value_counts().sort_index().to_dict(),
                'test_class_dist': test_center['tct_5class'].value_counts().sort_index().to_dict()
            }
        
        return center_dist
    
    def plot_class_distribution(self, train_df, test_df):
        """绘制类别分布图"""
        print(f"\n📊 绘制类别分布图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 训练集类别分布
        train_dist = train_df['tct_5class'].value_counts().sort_index()
        axes[0, 0].bar(range(5), [train_dist.get(i, 0) for i in range(5)])
        axes[0, 0].set_title('训练集类别分布')
        axes[0, 0].set_xlabel('类别')
        axes[0, 0].set_ylabel('样本数')
        axes[0, 0].set_xticks(range(5))
        axes[0, 0].set_xticklabels([name.split('(')[0] for name in self.class_names], rotation=45)
        
        # 2. 测试集类别分布
        test_dist = test_df['tct_5class'].value_counts().sort_index()
        axes[0, 1].bar(range(5), [test_dist.get(i, 0) for i in range(5)])
        axes[0, 1].set_title('测试集类别分布')
        axes[0, 1].set_xlabel('类别')
        axes[0, 1].set_ylabel('样本数')
        axes[0, 1].set_xticks(range(5))
        axes[0, 1].set_xticklabels([name.split('(')[0] for name in self.class_names], rotation=45)
        
        # 3. 类别比例对比
        train_props = [train_dist.get(i, 0) / len(train_df) for i in range(5)]
        test_props = [test_dist.get(i, 0) / len(test_df) for i in range(5)]
        
        x = np.arange(5)
        width = 0.35
        
        axes[1, 0].bar(x - width/2, train_props, width, label='训练集')
        axes[1, 0].bar(x + width/2, test_props, width, label='测试集')
        axes[1, 0].set_title('类别比例对比')
        axes[1, 0].set_xlabel('类别')
        axes[1, 0].set_ylabel('比例')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([name.split('(')[0] for name in self.class_names], rotation=45)
        axes[1, 0].legend()
        
        # 4. 中心分布
        train_df['center'] = train_df['OCT'].apply(lambda x: '恩施' if 'M22105' in str(x) else 
                                                      '荆州' if 'M0008' in str(x) else
                                                      '十堰' if 'M22104' in str(x) or 'M22101' in str(x) else
                                                      '武大' if 'M20203' in str(x) or 'M20105' in str(x) else
                                                      '襄阳' if 'M22102' in str(x) else 'Unknown')
        
        center_dist = train_df['center'].value_counts()
        axes[1, 1].pie(center_dist.values, labels=center_dist.index, autopct='%1.1f%%')
        axes[1, 1].set_title('中心分布')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'class_distribution_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 类别分布图已保存")
    
    def run_dataset_creation(self):
        """运行完整的数据集创建流程"""
        print("🚀 开始创建5分类数据集")
        print("=" * 60)
        
        # 1. 加载原始数据
        print("📂 加载原始数据...")
        train_df = pd.read_csv(os.path.join(self.data_path, 'train_labels.csv'))
        test_df = pd.read_csv(os.path.join(self.data_path, 'test_labels.csv'))
        
        print(f"  原始训练集: {len(train_df)} 样本")
        print(f"  原始测试集: {len(test_df)} 样本")
        
        # 2. 创建5分类标签
        train_df = self.create_tct_5class_labels(train_df)
        test_df = self.create_tct_5class_labels(test_df)
        
        # 3. 分析类别分布
        train_dist, train_balance = self.analyze_class_distribution(train_df, "训练集")
        test_dist, test_balance = self.analyze_class_distribution(test_df, "测试集")
        
        # 4. 创建平衡数据集（可选）
        print(f"\n⚖️ 数据集平衡性分析:")
        print(f"  训练集平衡性: {train_balance:.3f}")
        print(f"  测试集平衡性: {test_balance:.3f}")
        
        if train_balance < 0.1:  # 如果严重不平衡
            print("  ⚠️ 检测到严重类别不平衡，建议使用平衡策略")
            train_df = self.create_balanced_dataset(train_df, min_samples_per_class=5)
            test_df = self.create_balanced_dataset(test_df, min_samples_per_class=2)
        
        # 5. 创建输出结构
        self.create_output_structure()
        
        # 6. 保存数据集
        self.save_datasets(train_df, test_df)
        
        # 7. 创建数据摘要
        self.create_data_summary(train_df, test_df)
        
        # 8. 绘制分布图
        self.plot_class_distribution(train_df, test_df)
        
        print(f"\n🎉 5分类数据集创建完成！")
        print(f"📁 输出目录: {self.output_path}")
        print(f"📊 总样本数: {len(train_df) + len(test_df)}")
        print(f"🏷️ 类别数: 5")
        
        return train_df, test_df


def main():
    creator = FiveClassDatasetCreator()
    train_df, test_df = creator.run_dataset_creation()


if __name__ == "__main__":
    main()
