#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
外部验证数据质量检查脚本
用于检查外部验证数据的完整性、一致性和质量
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class ExternalDataQualityChecker:
    """外部验证数据质量检查器"""
    
    def __init__(self, data_root: str):
        """
        Args:
            data_root: 外部验证数据根目录
        """
        self.data_root = Path(data_root)
        self.issues = []
        self.warnings = []
        self.summary = {}
    
    def check_data_structure(self) -> Dict:
        """检查数据结构"""
        print("📁 检查数据结构...")
        
        structure_ok = True
        required_dirs = ['images', 'metadata.csv', 'labels.csv']
        
        for center_dir in self.data_root.iterdir():
            if not center_dir.is_dir():
                continue
            
            print(f"  检查中心: {center_dir.name}")
            
            for required in required_dirs:
                if required == 'images':
                    images_dir = center_dir / 'images'
                    if not images_dir.exists():
                        self.issues.append(f"{center_dir.name}: 缺少images目录")
                        structure_ok = False
                    else:
                        # 检查OCT和阴道镜子目录
                        oct_dir = images_dir / 'oct'
                        col_dir = images_dir / 'colposcopy'
                        if not oct_dir.exists():
                            self.warnings.append(f"{center_dir.name}: 缺少OCT图像目录")
                        if not col_dir.exists():
                            self.warnings.append(f"{center_dir.name}: 缺少阴道镜图像目录")
                else:
                    file_path = center_dir / required
                    if not file_path.exists():
                        self.issues.append(f"{center_dir.name}: 缺少{required}")
                        structure_ok = False
        
        self.summary['structure_ok'] = structure_ok
        return {'ok': structure_ok, 'issues': self.issues, 'warnings': self.warnings}
    
    def check_image_completeness(self) -> Dict:
        """检查图像完整性"""
        print("🖼️  检查图像完整性...")
        
        completeness_results = {}
        
        for center_dir in self.data_root.iterdir():
            if not center_dir.is_dir():
                continue
            
            center_name = center_dir.name
            images_dir = center_dir / 'images'
            
            if not images_dir.exists():
                continue
            
            # 检查OCT图像
            oct_dir = images_dir / 'oct'
            if oct_dir.exists():
                oct_patients = [d for d in oct_dir.iterdir() if d.is_dir()]
                oct_completeness = {}
                
                for patient_dir in oct_patients:
                    patient_id = patient_dir.name
                    # 检查是否有12个点位
                    point_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
                    if len(point_dirs) != 12:
                        self.warnings.append(
                            f"{center_name}/{patient_id}: OCT点位数量={len(point_dirs)} (期望12)"
                        )
                    
                    # 检查每个点位的帧数
                    total_frames = 0
                    for point_dir in point_dirs:
                        frames = list(point_dir.glob('*.jpg')) + list(point_dir.glob('*.png'))
                        total_frames += len(frames)
                        if len(frames) != 10:
                            self.warnings.append(
                                f"{center_name}/{patient_id}/{point_dir.name}: "
                                f"帧数={len(frames)} (期望10)"
                            )
                    
                    oct_completeness[patient_id] = {
                        'points': len(point_dirs),
                        'total_frames': total_frames,
                        'expected_frames': 120,
                        'complete': total_frames == 120
                    }
                
                completeness_results[center_name] = {
                    'oct': oct_completeness,
                    'n_patients': len(oct_patients)
                }
            
            # 检查阴道镜图像
            col_dir = images_dir / 'colposcopy'
            if col_dir.exists():
                col_patients = [d for d in col_dir.iterdir() if d.is_dir()]
                col_completeness = {}
                
                for patient_dir in col_patients:
                    patient_id = patient_dir.name
                    images = list(patient_dir.glob('*.jpg')) + list(patient_dir.glob('*.png'))
                    col_completeness[patient_id] = {
                        'n_images': len(images),
                        'expected': 3,
                        'complete': len(images) >= 3
                    }
                    
                    if len(images) < 3:
                        self.warnings.append(
                            f"{center_name}/{patient_id}: 阴道镜图像数量={len(images)} (期望≥3)"
                        )
        
        self.summary['completeness'] = completeness_results
        return completeness_results
    
    def check_metadata_completeness(self) -> Dict:
        """检查元数据完整性"""
        print("📋 检查元数据完整性...")
        
        metadata_results = {}
        required_fields = ['patient_id', 'age', 'hpv', 'tct', 'label']
        
        for center_dir in self.data_root.iterdir():
            if not center_dir.is_dir():
                continue
            
            center_name = center_dir.name
            metadata_file = center_dir / 'metadata.csv'
            
            if not metadata_file.exists():
                self.issues.append(f"{center_name}: 缺少metadata.csv")
                continue
            
            try:
                df = pd.read_csv(metadata_file)
                
                # 检查必需字段
                missing_fields = [f for f in required_fields if f not in df.columns]
                if missing_fields:
                    self.issues.append(
                        f"{center_name}: metadata.csv缺少字段: {missing_fields}"
                    )
                
                # 检查缺失值
                missing_values = df[required_fields].isnull().sum()
                if missing_values.sum() > 0:
                    self.warnings.append(
                        f"{center_name}: metadata.csv有缺失值:\n{missing_values.to_dict()}"
                    )
                
                metadata_results[center_name] = {
                    'n_samples': len(df),
                    'missing_fields': missing_fields,
                    'missing_values': missing_values.to_dict(),
                    'complete': len(missing_fields) == 0 and missing_values.sum() == 0
                }
            except Exception as e:
                self.issues.append(f"{center_name}: 读取metadata.csv失败: {e}")
        
        self.summary['metadata'] = metadata_results
        return metadata_results
    
    def check_label_consistency(self) -> Dict:
        """检查标注一致性"""
        print("🏷️  检查标注一致性...")
        
        label_results = {}
        
        for center_dir in self.data_root.iterdir():
            if not center_dir.is_dir():
                continue
            
            center_name = center_dir.name
            labels_file = center_dir / 'labels.csv'
            
            if not labels_file.exists():
                self.issues.append(f"{center_name}: 缺少labels.csv")
                continue
            
            try:
                df = pd.read_csv(labels_file)
                
                # 检查标签分布
                if 'label' in df.columns:
                    label_dist = df['label'].value_counts()
                    label_results[center_name] = {
                        'n_samples': len(df),
                        'positive': int(label_dist.get(1, 0)),
                        'negative': int(label_dist.get(0, 0)),
                        'distribution': label_dist.to_dict()
                    }
                    
                    # 检查类别平衡
                    if len(label_dist) == 2:
                        ratio = label_dist[1] / label_dist[0] if label_dist[0] > 0 else np.inf
                        if ratio < 0.3 or ratio > 3.0:
                            self.warnings.append(
                                f"{center_name}: 类别不平衡 (阳性/阴性={ratio:.2f})"
                            )
                else:
                    self.issues.append(f"{center_name}: labels.csv缺少'label'字段")
            except Exception as e:
                self.issues.append(f"{center_name}: 读取labels.csv失败: {e}")
        
        self.summary['labels'] = label_results
        return label_results
    
    def check_data_distribution(self) -> Dict:
        """检查数据分布"""
        print("📊 检查数据分布...")
        
        distribution_results = {}
        
        # 收集所有中心的元数据
        all_metadata = []
        all_labels = []
        
        for center_dir in self.data_root.iterdir():
            if not center_dir.is_dir():
                continue
            
            center_name = center_dir.name
            metadata_file = center_dir / 'metadata.csv'
            labels_file = center_dir / 'labels.csv'
            
            if metadata_file.exists() and labels_file.exists():
                try:
                    meta_df = pd.read_csv(metadata_file)
                    label_df = pd.read_csv(labels_file)
                    
                    if 'patient_id' in meta_df.columns and 'patient_id' in label_df.columns:
                        merged = pd.merge(meta_df, label_df, on='patient_id', how='inner')
                        merged['center'] = center_name
                        all_metadata.append(merged)
                        
                        if 'label' in label_df.columns:
                            all_labels.extend(label_df['label'].tolist())
                except Exception as e:
                    self.warnings.append(f"{center_name}: 合并元数据和标签失败: {e}")
        
        if all_metadata:
            combined_df = pd.concat(all_metadata, ignore_index=True)
            
            distribution_results = {
                'total_samples': len(combined_df),
                'n_centers': len(combined_df['center'].unique()) if 'center' in combined_df.columns else 0,
                'age_distribution': {
                    'mean': float(combined_df['age'].mean()) if 'age' in combined_df.columns else None,
                    'std': float(combined_df['age'].std()) if 'age' in combined_df.columns else None,
                    'min': float(combined_df['age'].min()) if 'age' in combined_df.columns else None,
                    'max': float(combined_df['age'].max()) if 'age' in combined_df.columns else None
                },
                'label_distribution': {
                    'positive': int(np.sum(all_labels == 1)) if all_labels else 0,
                    'negative': int(np.sum(all_labels == 0)) if all_labels else 0,
                    'ratio': float(np.sum(all_labels == 1) / np.sum(all_labels == 0)) 
                             if all_labels and np.sum(all_labels == 0) > 0 else None
                }
            }
            
            # 检查年龄分布
            if 'age' in combined_df.columns:
                age_outliers = combined_df[(combined_df['age'] < 18) | (combined_df['age'] > 65)]
                if len(age_outliers) > 0:
                    self.warnings.append(
                        f"年龄异常值: {len(age_outliers)}例 (期望18-65岁)"
                    )
        
        self.summary['distribution'] = distribution_results
        return distribution_results
    
    def generate_report(self, output_file: str = None):
        """生成质量检查报告"""
        if output_file is None:
            output_file = self.data_root / 'quality_check_report.json'
        
        report = {
            'check_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_root': str(self.data_root),
            'summary': self.summary,
            'issues': self.issues,
            'warnings': self.warnings,
            'overall_status': 'PASS' if len(self.issues) == 0 else 'FAIL'
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 质量检查报告已保存: {output_file}")
        print(f"\n📊 检查结果摘要:")
        print(f"   总体状态: {report['overall_status']}")
        print(f"   问题数: {len(self.issues)}")
        print(f"   警告数: {len(self.warnings)}")
        
        if self.issues:
            print(f"\n❌ 发现的问题:")
            for issue in self.issues:
                print(f"   - {issue}")
        
        if self.warnings:
            print(f"\n⚠️  警告:")
            for warning in self.warnings[:10]:  # 只显示前10个
                print(f"   - {warning}")
            if len(self.warnings) > 10:
                print(f"   ... 还有{len(self.warnings) - 10}个警告")
        
        return report


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='外部验证数据质量检查')
    parser.add_argument('--data_root', type=str, required=True,
                       help='外部验证数据根目录')
    parser.add_argument('--output_file', type=str, default=None,
                       help='输出报告文件路径')
    
    args = parser.parse_args()
    
    # 创建检查器
    checker = ExternalDataQualityChecker(args.data_root)
    
    # 执行所有检查
    print("=" * 60)
    print("🔍 开始外部验证数据质量检查")
    print("=" * 60)
    print()
    
    checker.check_data_structure()
    checker.check_image_completeness()
    checker.check_metadata_completeness()
    checker.check_label_consistency()
    checker.check_data_distribution()
    
    # 生成报告
    checker.generate_report(args.output_file)
    
    print("\n" + "=" * 60)
    print("✅ 质量检查完成")
    print("=" * 60)


if __name__ == '__main__':
    main()


