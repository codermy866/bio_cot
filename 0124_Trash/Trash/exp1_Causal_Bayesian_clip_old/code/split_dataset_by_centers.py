#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据医疗中心划分数据集
- 内部开发集（训练/验证）：恩施、襄阳、十堰（混合训练）
- 外部独立测试集：荆州、武大（严禁参与训练，作为最终考场）
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from sklearn.model_selection import train_test_split

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class CenterBasedDatasetSplitter:
    """基于医疗中心的数据集划分器"""
    
    def __init__(self, data_root: str = '5centers_multi'):
        """
        Args:
            data_root: 数据根目录
        """
        self.data_root = Path(data_root)
        self.train_labels_file = self.data_root / 'train_labels.csv'
        self.test_labels_file = self.data_root / 'test_labels.csv'
        
        # 中心代码到中心名称的映射
        self.center_code_mapping = {
            'M22105': '恩施',
            'M0008': '荆州',
            'M22104': '十堰',
            'M22101': '十堰',
            'M20203': '武大',
            'M20105': '武大',
            'M22102': '襄阳'
        }
        
        # 内部开发集中心（用于训练/验证）
        self.internal_centers = ['恩施', '襄阳', '十堰']
        
        # 外部测试集中心（严禁参与训练）
        self.external_centers = ['荆州', '武大']
    
    def identify_center(self, oct_id: str) -> str:
        """从OCT ID识别中心名称"""
        if pd.isna(oct_id):
            return 'Unknown'
        
        oct_str = str(oct_id)
        for code, center_name in self.center_code_mapping.items():
            if code in oct_str:
                return center_name
        return 'Unknown'
    
    def load_all_data(self) -> pd.DataFrame:
        """加载所有数据（train + test）"""
        print("📂 加载数据文件...")
        
        # 读取训练和测试标签
        train_df = pd.read_csv(self.train_labels_file)
        test_df = pd.read_csv(self.test_labels_file)
        
        # 添加来源标记
        train_df['source'] = 'train'
        test_df['source'] = 'test'
        
        # 合并数据
        all_data = pd.concat([train_df, test_df], ignore_index=True)
        
        print(f"✅ 训练集样本数: {len(train_df)}")
        print(f"✅ 测试集样本数: {len(test_df)}")
        print(f"✅ 总样本数: {len(all_data)}")
        
        return all_data
    
    def analyze_center_distribution(self, df: pd.DataFrame) -> Dict:
        """分析各中心的样本分布"""
        print("\n📊 分析中心分布...")
        
        # 识别中心
        df['center'] = df['OCT'].apply(self.identify_center)
        
        # 统计分布
        distribution = {}
        all_centers = df['center'].unique()
        
        print(f"\n{'中心':<10} {'样本数':<12} {'阳性数':<12} {'阳性率':<12} {'来源':<20}")
        print("-" * 80)
        
        for center in sorted(all_centers):
            if center == 'Unknown':
                continue
            
            center_data = df[df['center'] == center]
            positive_count = int(center_data['label'].sum())
            total_count = len(center_data)
            positive_rate = positive_count / total_count if total_count > 0 else 0
            
            sources = center_data['source'].unique()
            source_str = ', '.join(sources)
            
            distribution[center] = {
                'total': total_count,
                'positive': positive_count,
                'negative': total_count - positive_count,
                'positive_rate': positive_rate,
                'sources': list(sources)
            }
            
            print(f"{center:<10} {total_count:<12} {positive_count:<12} {positive_rate*100:.2f}%{'':<8} {source_str:<20}")
        
        return distribution
    
    def split_by_centers(self, df: pd.DataFrame, val_ratio: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        根据中心划分数据集
        
        Args:
            df: 所有数据
            val_ratio: 验证集比例（从内部开发集中划分）
            random_state: 随机种子
            
        Returns:
            train_df: 内部训练集（恩施、襄阳、十堰）
            val_df: 内部验证集（恩施、襄阳、十堰）
            external_test_df: 外部测试集（荆州、武大）
        """
        print("\n🔀 划分数据集...")
        
        # 识别中心
        df['center'] = df['OCT'].apply(self.identify_center)
        
        # 分离内部和外部数据
        internal_mask = df['center'].isin(self.internal_centers)
        external_mask = df['center'].isin(self.external_centers)
        
        internal_data = df[internal_mask].copy()
        external_data = df[external_mask].copy()
        
        print(f"\n内部开发集中心 ({', '.join(self.internal_centers)}):")
        print(f"  - 总样本数: {len(internal_data)}")
        print(f"  - 阳性样本: {int(internal_data['label'].sum())}")
        print(f"  - 阴性样本: {int((internal_data['label'] == 0).sum())}")
        
        print(f"\n外部测试集中心 ({', '.join(self.external_centers)}):")
        print(f"  - 总样本数: {len(external_data)}")
        print(f"  - 阳性样本: {int(external_data['label'].sum())}")
        print(f"  - 阴性样本: {int((external_data['label'] == 0).sum())}")
        
        # 从内部数据中划分训练集和验证集（保持类别平衡）
        if len(internal_data) > 0:
            train_df, val_df = train_test_split(
                internal_data,
                test_size=val_ratio,
                random_state=random_state,
                stratify=internal_data['label']  # 保持类别平衡
            )
        else:
            train_df = pd.DataFrame()
            val_df = pd.DataFrame()
        
        print(f"\n✅ 划分完成:")
        print(f"  - 内部训练集: {len(train_df)} 样本")
        print(f"  - 内部验证集: {len(val_df)} 样本")
        print(f"  - 外部测试集: {len(external_data)} 样本")
        
        return train_df, val_df, external_data
    
    def save_split_results(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          external_test_df: pd.DataFrame, output_dir: Path):
        """保存划分结果"""
        print("\n💾 保存划分结果...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 移除辅助列（center, source）
        columns_to_save = ['ID', 'OCT', 'AGE', 'HPV清洗', 'TCT清洗', 'label']
        
        # 保存训练集
        train_output = train_df[columns_to_save].copy()
        train_file = output_dir / 'train_labels.csv'
        train_output.to_csv(train_file, index=False, encoding='utf-8-sig')
        print(f"✅ 训练集已保存: {train_file} ({len(train_output)} 样本)")
        
        # 保存验证集
        val_output = val_df[columns_to_save].copy()
        val_file = output_dir / 'val_labels.csv'
        val_output.to_csv(val_file, index=False, encoding='utf-8-sig')
        print(f"✅ 验证集已保存: {val_file} ({len(val_output)} 样本)")
        
        # 保存外部测试集
        external_output = external_test_df[columns_to_save].copy()
        external_file = output_dir / 'external_test_labels.csv'
        external_output.to_csv(external_file, index=False, encoding='utf-8-sig')
        print(f"✅ 外部测试集已保存: {external_file} ({len(external_output)} 样本)")
        
        # 保存统计信息
        stats = {
            'split_date': pd.Timestamp.now().isoformat(),
            'internal_centers': self.internal_centers,
            'external_centers': self.external_centers,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'external_test_samples': len(external_test_df),
            'train_positive': int(train_df['label'].sum()),
            'train_negative': int((train_df['label'] == 0).sum()),
            'val_positive': int(val_df['label'].sum()),
            'val_negative': int((val_df['label'] == 0).sum()),
            'external_test_positive': int(external_test_df['label'].sum()),
            'external_test_negative': int((external_test_df['label'] == 0).sum()),
        }
        
        stats_file = output_dir / 'split_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"✅ 统计信息已保存: {stats_file}")
        
        return stats
    
    def create_summary_report(self, distribution: Dict, stats: Dict, output_dir: Path):
        """创建划分报告"""
        print("\n📝 生成划分报告...")
        
        report_file = output_dir / 'split_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 数据集划分报告\n\n")
            f.write(f"**划分时间**: {stats['split_date']}\n\n")
            
            f.write("## 划分策略\n\n")
            f.write("### 内部开发集（用于训练/验证）\n")
            f.write(f"- **中心**: {', '.join(stats['internal_centers'])}\n")
            f.write(f"- **训练集样本数**: {stats['train_samples']}\n")
            f.write(f"  - 阳性: {stats['train_positive']}\n")
            f.write(f"  - 阴性: {stats['train_negative']}\n")
            f.write(f"- **验证集样本数**: {stats['val_samples']}\n")
            f.write(f"  - 阳性: {stats['val_positive']}\n")
            f.write(f"  - 阴性: {stats['val_negative']}\n\n")
            
            f.write("### 外部独立测试集（严禁参与训练）\n")
            f.write(f"- **中心**: {', '.join(stats['external_centers'])}\n")
            f.write(f"- **样本数**: {stats['external_test_samples']}\n")
            f.write(f"  - 阳性: {stats['external_test_positive']}\n")
            f.write(f"  - 阴性: {stats['external_test_negative']}\n\n")
            
            f.write("## 中心分布详情\n\n")
            f.write("| 中心 | 总样本数 | 阳性数 | 阴性数 | 阳性率 |\n")
            f.write("|------|----------|--------|--------|--------|\n")
            
            for center, info in sorted(distribution.items()):
                if center == 'Unknown':
                    continue
                f.write(f"| {center} | {info['total']} | {info['positive']} | "
                       f"{info['negative']} | {info['positive_rate']*100:.2f}% |\n")
            
            f.write("\n## 重要提示\n\n")
            f.write("⚠️ **外部测试集（荆州、武大）严禁参与训练过程！**\n\n")
            f.write("外部测试集仅用于最终模型评估，确保结果的可信度和泛化能力。\n")
        
        print(f"✅ 划分报告已保存: {report_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='根据医疗中心划分数据集')
    parser.add_argument('--data_path', type=str, 
                       default='/data2/hmy/5Center_datas/5centers_multi',
                       help='数据路径')
    parser.add_argument('--output_dir', type=str,
                       default='/data2/hmy/5Center_datas/5centers_multi_internal_external_final',
                       help='输出目录')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例（从内部开发集中划分）')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔀 基于医疗中心的数据集划分")
    print("=" * 80)
    print(f"数据路径: {args.data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"验证集比例: {args.val_ratio}")
    print("=" * 80)
    
    # 创建划分器
    splitter = CenterBasedDatasetSplitter(data_root=args.data_path)
    
    # 加载所有数据
    all_data = splitter.load_all_data()
    
    # 分析中心分布
    distribution = splitter.analyze_center_distribution(all_data)
    
    # 划分数据集
    train_df, val_df, external_test_df = splitter.split_by_centers(
        all_data, 
        val_ratio=args.val_ratio,
        random_state=args.random_state
    )
    
    # 保存结果
    output_dir = Path(args.output_dir)
    stats = splitter.save_split_results(train_df, val_df, external_test_df, output_dir)
    
    # 生成报告
    splitter.create_summary_report(distribution, stats, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ 数据集划分完成！")
    print("=" * 80)
    print(f"\n📁 输出目录: {output_dir}")
    print(f"  - train_labels.csv: 内部训练集")
    print(f"  - val_labels.csv: 内部验证集")
    print(f"  - external_test_labels.csv: 外部测试集（严禁参与训练）")
    print(f"  - split_statistics.json: 统计信息")
    print(f"  - split_report.md: 划分报告")
    print("\n⚠️  重要提示：外部测试集（荆州、武大）严禁参与训练过程！")


if __name__ == '__main__':
    main()

