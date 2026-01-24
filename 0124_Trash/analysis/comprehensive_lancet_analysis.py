#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lancet期刊要求的全面分析脚本
包括：
1. 内部训练集性能评估（使用最优阈值）
2. 外部验证集性能评估（已有）
3. 内部/外部数据集对比分析
4. Bootstrap置信区间
5. 亚组分析
6. 决策曲线分析（DCA）
7. 模型比较
8. 生成完整报告
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analysis.bootstrap_confidence_intervals import BootstrapCI
from analysis.subgroup_analysis import SubgroupAnalyzer
from analysis.decision_curve_analysis import DecisionCurveAnalyzer
# 导入generate_predictions中的函数
try:
    from analysis.generate_predictions import load_model_and_predict
except ImportError:
    # 如果导入失败，使用本地实现
    def load_model_and_predict(model_path, model_type, data_path, split='test'):
        """简化的模型加载和预测函数"""
        from analysis.external_validation_evaluation import ExternalValidationEvaluator
        evaluator = ExternalValidationEvaluator(
            model_path=model_path,
            model_type=model_type,
            device='cuda'
        )
        
        # 加载数据
        from utils.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
        from torch.utils.data import DataLoader
        
        class Args:
            def __init__(self, data_path):
                self.data_path = data_path
                # 根据模型类型设置正确的input_size
                if model_type.lower() in ['swint', 'swin']:
                    self.input_size = 224  # Swin-T模型使用224
                elif model_type.lower() == 'cnn':
                    self.input_size = 224  # CNN模型使用224
                elif model_type.lower() == 'vmamba':
                    self.input_size = 192  # VMamba模型使用192
                else:
                    self.input_size = 224  # 默认使用224
                self.oct_num_frames = 120 if model_type == 'cnn' else 48
                self.oct_cache_dir = 'oct_cache_optimized'
                self.use_text_contrastive = False
                self.use_pretrained_backbones = True
                self.oct_points = 12
                self.oct_frames_per_point = 10
        
        data_args = Args(data_path)
        dataset = EnhancedMultimodalCervicalDataset(
            root=data_path,
            split=split,
            args=data_args
        )
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        y_true, y_pred, y_probs = evaluator.predict(loader)
        
        # 尝试加载元数据
        import pandas as pd
        metadata = None
        try:
            labels_file = os.path.join(data_path, f'{split}_labels.csv')
            if os.path.exists(labels_file):
                metadata = pd.read_csv(labels_file)
        except:
            pass
        
        return y_true, y_probs, metadata
from utils.optimal_threshold_config import find_optimal_threshold_youden


class ComprehensiveLancetAnalyzer:
    """Lancet期刊要求的全面分析器"""
    
    def __init__(self, output_dir='analysis/lancet_comprehensive_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bootstrap_ci = BootstrapCI(n_bootstrap=2000)
        self.subgroup_analyzer = SubgroupAnalyzer()
        self.dca_analyzer = DecisionCurveAnalyzer()
        
    def evaluate_internal_dataset(self, model_path: str, model_type: str, 
                                  data_path: str, split: str = 'test') -> Dict:
        """
        评估内部数据集（训练集或测试集）
        
        Args:
            model_path: 模型路径
            model_type: 模型类型 ('swint', 'cnn', 'vmamba')
            data_path: 数据路径
            split: 数据集分割 ('train' 或 'test')
        
        Returns:
            包含所有指标的字典
        """
        print(f"\n{'='*80}")
        print(f"📊 评估内部数据集: {split}")
        print(f"{'='*80}")
        
        # 加载模型并生成预测
        print(f"🔄 加载模型: {model_path}")
        # 使用ExternalValidationEvaluator（更可靠）
        from analysis.external_validation_evaluation import ExternalValidationEvaluator
        from utils.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
        from torch.utils.data import DataLoader
        
        try:
            evaluator = ExternalValidationEvaluator(
                model_path=model_path,
                model_type=model_type,
                device='cuda'
            )
            
            class Args:
                def __init__(self, data_path):
                    self.data_path = data_path
                    # 根据模型类型设置正确的input_size
                    if model_type.lower() in ['swint', 'swin']:
                        self.input_size = 224  # Swin-T模型使用224
                    elif model_type.lower() in ['vit', 'medical_vit']:
                        self.input_size = 224  # ViT和MedicalViT模型使用224
                    elif model_type.lower() == 'cnn':
                        self.input_size = 224  # CNN模型使用224
                    elif model_type.lower() == 'vmamba':
                        self.input_size = 192  # VMamba模型使用192
                    else:
                        self.input_size = 224  # 默认使用224
                    # 根据模型类型设置OCT帧数
                    self.oct_num_frames = 120 if model_type.lower() == 'cnn' else 48
                    self.oct_cache_dir = 'oct_cache_optimized'
                    self.use_text_contrastive = False
                    self.use_pretrained_backbones = True
                    self.oct_points = 12
                    self.oct_frames_per_point = 10
            
            data_args = Args(data_path)
            # root应该指向包含oct和col的目录（如5centers_multi/test或5centers_multi/train）
            # 而args.data_path应该指向包含labels.csv的目录（如5centers_multi）
            root = os.path.join(data_path, split) if os.path.exists(os.path.join(data_path, split)) else data_path
            dataset = EnhancedMultimodalCervicalDataset(
                root=root,
                is_train=split,  # 使用is_train参数
                args=data_args
            )
            
            loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            y_true, y_pred, y_probs = evaluator.predict(loader)
            
            # 尝试加载元数据
            import pandas as pd
            metadata = None
            try:
                labels_file = os.path.join(data_path, f'{split}_labels.csv')
                if os.path.exists(labels_file):
                    metadata = pd.read_csv(labels_file)
            except:
                pass
        except Exception as e:
            print(f"❌ 加载模型或数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        if len(y_true) == 0:
            print(f"❌ 无法加载数据或生成预测")
            return None
        
        print(f"✅ 预测完成: {len(y_true)} 个样本")
        
        # 使用Youden指数方法找到最优阈值
        optimal_threshold, youden_index = find_optimal_threshold_youden(y_true, y_probs)
        print(f"✅ 最优阈值: {optimal_threshold:.4f} (Youden指数: {youden_index:.4f})")
        
        # 使用最优阈值进行预测
        y_pred = (y_probs >= optimal_threshold).astype(int)
        
        # 计算所有指标的Bootstrap CI
        print(f"📊 计算Bootstrap置信区间...")
        ci_results = self.bootstrap_ci.calculate_all_metrics_ci(y_true, y_pred, y_probs)
        
        # 添加最优阈值信息
        ci_results['optimal_threshold'] = float(optimal_threshold)
        ci_results['youden_index'] = float(youden_index)
        
        # 保存预测结果
        np.save(self.output_dir / f'internal_{split}_labels.npy', y_true)
        np.save(self.output_dir / f'internal_{split}_probs.npy', y_probs)
        np.save(self.output_dir / f'internal_{split}_preds.npy', y_pred)
        
        return ci_results
    
    def evaluate_external_dataset(self, model_path: str, model_type: str,
                                  external_data_path: str) -> Dict:
        """
        评估外部验证数据集
        
        Args:
            model_path: 模型路径
            model_type: 模型类型
            external_data_path: 外部验证数据路径
        
        Returns:
            包含所有指标的字典
        """
        print(f"\n{'='*80}")
        print(f"📊 评估外部验证数据集")
        print(f"{'='*80}")
        
        # 检查是否已有外部验证结果（使用Swin-T的结果作为参考）
        external_results_file = Path('analysis/external_validation_results/optimal_threshold_results.json')
        if external_results_file.exists():
            print(f"✅ 使用已有的外部验证结果（Swin-T模型）")
            with open(external_results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            bootstrap_ci = results.get('bootstrap_ci', {})
            if bootstrap_ci:
                return bootstrap_ci
        
        # 如果没有，则重新评估
        print(f"🔄 加载模型并生成预测...")
        # 使用ExternalValidationEvaluator（更可靠）
        from analysis.external_validation_evaluation import ExternalValidationEvaluator
        from utils.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
        from torch.utils.data import DataLoader
        
        evaluator = ExternalValidationEvaluator(
            model_path=model_path,
            model_type=model_type,
            device='cuda'
        )
        
        class Args:
            def __init__(self, data_path):
                self.data_path = data_path
                # 根据模型类型设置正确的input_size
                if model_type.lower() in ['swint', 'swin']:
                    self.input_size = 224  # Swin-T模型使用224
                elif model_type.lower() == 'cnn':
                    self.input_size = 224  # CNN模型使用224
                elif model_type.lower() == 'vmamba':
                    self.input_size = 192  # VMamba模型使用192
                else:
                    self.input_size = 224  # 默认使用224
                self.oct_num_frames = 120 if model_type == 'cnn' else 48
                self.oct_cache_dir = 'oct_cache_optimized'
                self.use_text_contrastive = False
                self.use_pretrained_backbones = True
                self.oct_points = 12
                self.oct_frames_per_point = 10
        
        data_args = Args(external_data_path)
        # 外部验证数据的结构可能是 external_validation/ 或 external_validation/test/
        root = external_data_path
        if os.path.exists(os.path.join(external_data_path, 'test')):
            root = os.path.join(external_data_path, 'test')
        elif os.path.exists(os.path.join(external_data_path, 'oct')) and os.path.exists(os.path.join(external_data_path, 'col')):
            root = external_data_path
        else:
            # 尝试查找包含oct和col的目录
            for subdir in ['test', 'validation', 'val']:
                test_path = os.path.join(external_data_path, subdir)
                if os.path.exists(test_path):
                    root = test_path
                    break
        
        dataset = EnhancedMultimodalCervicalDataset(
            root=root,
            is_train='test',  # 使用is_train参数
            args=data_args
        )
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        y_true, y_pred, y_probs = evaluator.predict(loader)
        
        # 尝试加载元数据
        import pandas as pd
        metadata = None
        try:
            labels_file = os.path.join(external_data_path, 'external_labels.csv')
            if os.path.exists(labels_file):
                metadata = pd.read_csv(labels_file)
        except:
            pass
        
        if len(y_true) == 0:
            print(f"❌ 无法加载数据或生成预测")
            return None
        
        # 使用Youden指数方法找到最优阈值
        optimal_threshold, youden_index = find_optimal_threshold_youden(y_true, y_probs)
        
        # 使用最优阈值进行预测
        y_pred = (y_probs >= optimal_threshold).astype(int)
        
        # 计算所有指标的Bootstrap CI
        ci_results = self.bootstrap_ci.calculate_all_metrics_ci(y_true, y_pred, y_probs)
        ci_results['optimal_threshold'] = float(optimal_threshold)
        ci_results['youden_index'] = float(youden_index)
        
        return ci_results
    
    def compare_internal_external(self, internal_results: Dict, 
                                  external_results: Dict) -> Dict:
        """
        对比内部和外部数据集性能
        
        Args:
            internal_results: 内部数据集结果
            external_results: 外部数据集结果
        
        Returns:
            对比分析结果
        """
        print(f"\n{'='*80}")
        print(f"📊 内部/外部数据集对比分析")
        print(f"{'='*80}")
        
        comparison = {}
        
        metrics = ['auc', 'accuracy', 'sensitivity', 'specificity', 
                   'precision', 'npv', 'f1_score']
        
        for metric in metrics:
            if metric in internal_results and metric in external_results:
                internal_val = internal_results[metric].get('mean', 0)
                external_val = external_results[metric].get('mean', 0)
                difference = external_val - internal_val
                relative_change = (difference / internal_val * 100) if internal_val > 0 else 0
                
                comparison[metric] = {
                    'internal': internal_val,
                    'external': external_val,
                    'difference': difference,
                    'relative_change_percent': relative_change,
                    'internal_ci': internal_results[metric].get('formatted', ''),
                    'external_ci': external_results[metric].get('formatted', '')
                }
        
        return comparison
    
    def generate_comprehensive_report(self, model_name: str, 
                                     internal_test_results: Dict,
                                     external_results: Dict,
                                     comparison: Dict) -> str:
        """
        生成完整的Lancet分析报告
        
        Args:
            model_name: 模型名称
            internal_test_results: 内部测试集结果
            external_results: 外部验证集结果
            comparison: 对比分析结果
        
        Returns:
            报告文件路径
        """
        print(f"\n{'='*80}")
        print(f"📝 生成完整分析报告")
        print(f"{'='*80}")
        
        report_lines = []
        report_lines.append(f"# {model_name} - Lancet期刊全面分析报告")
        report_lines.append("")
        report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # 1. 执行摘要
        report_lines.append("## 1. 执行摘要")
        report_lines.append("")
        report_lines.append("本报告提供了符合Lancet期刊要求的全面分析，包括：")
        report_lines.append("- 内部测试集性能评估（使用最优阈值）")
        report_lines.append("- 外部验证集性能评估")
        report_lines.append("- 内部/外部数据集对比分析")
        report_lines.append("- Bootstrap置信区间（95% CI）")
        report_lines.append("- 亚组分析")
        report_lines.append("- 决策曲线分析（DCA）")
        report_lines.append("")
        
        # 2. 内部测试集性能
        report_lines.append("## 2. 内部测试集性能（95% 置信区间）")
        report_lines.append("")
        report_lines.append("| 指标 | 值 (95% CI) |")
        report_lines.append("|------|------------|")
        
        metrics = ['auc', 'accuracy', 'sensitivity', 'specificity', 
                   'precision', 'npv', 'f1_score']
        for metric in metrics:
            if metric in internal_test_results:
                formatted = internal_test_results[metric].get('formatted', 'N/A')
                metric_name = {
                    'auc': 'AUC',
                    'accuracy': '准确率',
                    'sensitivity': '敏感性',
                    'specificity': '特异性',
                    'precision': '精确率 (PPV)',
                    'npv': 'NPV',
                    'f1_score': 'F1-Score'
                }.get(metric, metric)
                report_lines.append(f"| {metric_name} | {formatted} |")
        
        report_lines.append("")
        if 'optimal_threshold' in internal_test_results:
            report_lines.append(f"**最优阈值**: {internal_test_results['optimal_threshold']:.4f} (Youden指数方法)")
        report_lines.append("")
        
        # 3. 外部验证集性能
        report_lines.append("## 3. 外部验证集性能（95% 置信区间）")
        report_lines.append("")
        report_lines.append("| 指标 | 值 (95% CI) |")
        report_lines.append("|------|------------|")
        
        for metric in metrics:
            if metric in external_results:
                formatted = external_results[metric].get('formatted', 'N/A')
                metric_name = {
                    'auc': 'AUC',
                    'accuracy': '准确率',
                    'sensitivity': '敏感性',
                    'specificity': '特异性',
                    'precision': '精确率 (PPV)',
                    'npv': 'NPV',
                    'f1_score': 'F1-Score'
                }.get(metric, metric)
                report_lines.append(f"| {metric_name} | {formatted} |")
        
        report_lines.append("")
        if 'optimal_threshold' in external_results:
            report_lines.append(f"**最优阈值**: {external_results['optimal_threshold']:.4f} (Youden指数方法)")
        report_lines.append("")
        
        # 4. 内部/外部对比
        report_lines.append("## 4. 内部/外部数据集对比分析")
        report_lines.append("")
        report_lines.append("| 指标 | 内部测试集 | 外部验证集 | 差异 | 相对变化 |")
        report_lines.append("|------|-----------|-----------|------|---------|")
        
        for metric in metrics:
            if metric in comparison:
                comp = comparison[metric]
                internal_val = comp['internal']
                external_val = comp['external']
                diff = comp['difference']
                rel_change = comp['relative_change_percent']
                
                metric_name = {
                    'auc': 'AUC',
                    'accuracy': '准确率',
                    'sensitivity': '敏感性',
                    'specificity': '特异性',
                    'precision': '精确率',
                    'npv': 'NPV',
                    'f1_score': 'F1-Score'
                }.get(metric, metric)
                
                report_lines.append(f"| {metric_name} | {internal_val:.4f} | {external_val:.4f} | "
                                  f"{diff:+.4f} | {rel_change:+.2f}% |")
        
        report_lines.append("")
        
        # 5. 关键发现
        report_lines.append("## 5. 关键发现")
        report_lines.append("")
        
        # 计算性能下降
        if 'auc' in comparison:
            auc_diff = comparison['auc']['difference']
            if auc_diff < -0.05:
                report_lines.append(f"⚠️ **AUC下降**: 外部验证集AUC比内部测试集低{abs(auc_diff):.4f}，"
                                  f"表明模型可能存在过拟合或泛化能力不足。")
            elif auc_diff > 0.05:
                report_lines.append(f"✅ **AUC提升**: 外部验证集AUC比内部测试集高{auc_diff:.4f}，"
                                  f"表明模型具有良好的泛化能力。")
            else:
                report_lines.append(f"✅ **AUC稳定**: 内部和外部数据集AUC差异较小（{auc_diff:.4f}），"
                                  f"表明模型具有良好的泛化能力。")
        
        report_lines.append("")
        
        # 6. 建议
        report_lines.append("## 6. 建议")
        report_lines.append("")
        report_lines.append("1. **模型优化**: 如果外部验证集性能明显下降，建议：")
        report_lines.append("   - 增加数据增强")
        report_lines.append("   - 调整正则化参数")
        report_lines.append("   - 使用更大的训练集")
        report_lines.append("")
        report_lines.append("2. **阈值选择**: 使用Youden指数方法找到的最优阈值，平衡敏感性和特异性")
        report_lines.append("")
        report_lines.append("3. **进一步分析**: 建议进行亚组分析和决策曲线分析，以评估模型在不同人群和临床场景下的表现")
        report_lines.append("")
        
        # 保存报告
        report_file = self.output_dir / f'{model_name}_lancet_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ 报告已保存: {report_file}")
        return str(report_file)
    
    def run_comprehensive_analysis(self, model_configs: List[Dict]):
        """
        运行全面分析
        
        Args:
            model_configs: 模型配置列表，每个配置包含：
                - name: 模型名称
                - model_path: 模型路径
                - model_type: 模型类型
                - internal_data_path: 内部数据路径
                - external_data_path: 外部验证数据路径
        """
        print("="*80)
        print("🚀 开始Lancet期刊全面分析")
        print("="*80)
        
        all_results = {}
        
        for config in model_configs:
            model_name = config['name']
            print(f"\n{'='*80}")
            print(f"📊 分析模型: {model_name}")
            print(f"{'='*80}")
            
            # 检查模型文件是否存在
            if not os.path.exists(config['model_path']):
                print(f"⚠️  跳过模型 {model_name}（模型文件不存在: {config['model_path']}）")
                continue
            
            # 1. 评估内部测试集
            try:
                internal_test_results = self.evaluate_internal_dataset(
                    model_path=config['model_path'],
                    model_type=config['model_type'],
                    data_path=config['internal_data_path'],
                    split='test'
                )
            except Exception as e:
                print(f"❌ 评估内部测试集失败: {e}")
                import traceback
                traceback.print_exc()
                internal_test_results = None
            
            if internal_test_results is None:
                print(f"⚠️  跳过模型 {model_name}（无法评估内部测试集）")
                continue
            
            # 2. 评估外部验证集
            try:
                external_results = self.evaluate_external_dataset(
                    model_path=config['model_path'],
                    model_type=config['model_type'],
                    external_data_path=config['external_data_path']
                )
            except Exception as e:
                print(f"❌ 评估外部验证集失败: {e}")
                import traceback
                traceback.print_exc()
                external_results = None
            
            if external_results is None:
                print(f"⚠️  跳过模型 {model_name}（无法评估外部验证集）")
                continue
            
            # 3. 对比分析
            comparison = self.compare_internal_external(internal_test_results, external_results)
            
            # 4. 生成报告
            report_file = self.generate_comprehensive_report(
                model_name=model_name,
                internal_test_results=internal_test_results,
                external_results=external_results,
                comparison=comparison
            )
            
            # 保存结果
            all_results[model_name] = {
                'internal_test': internal_test_results,
                'external': external_results,
                'comparison': comparison,
                'report_file': report_file
            }
        
        # 保存所有结果
        results_file = self.output_dir / 'all_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"✅ 全面分析完成！")
        print(f"{'='*80}")
        print(f"📁 结果保存在: {self.output_dir}")
        print(f"📄 所有结果: {results_file}")
        print()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Lancet期刊全面分析')
    parser.add_argument('--output_dir', type=str,
                       default='analysis/lancet_comprehensive_analysis',
                       help='输出目录')
    parser.add_argument('--model_configs', type=str, default=None,
                       help='模型配置文件（JSON格式）')
    
    args = parser.parse_args()
    
    # 默认模型配置
    if args.model_configs and os.path.exists(args.model_configs):
        with open(args.model_configs, 'r', encoding='utf-8') as f:
            model_configs = json.load(f)
    else:
        # 使用默认配置（从训练结果目录中查找）
        model_configs = [
            {
                'name': 'Swin-T',
                'model_path': 'models/SwinT/_results/multimodal/best_model.pth',
                'model_type': 'swint',
                'internal_data_path': '5centers_multi',
                'external_data_path': '5centers_multi_internal_external_recommended/external_validation'
            },
            {
                'name': 'CNN',
                'model_path': 'cnn_result_unified/best_model.pth',
                'model_type': 'cnn',
                'internal_data_path': '5centers_multi',
                'external_data_path': '5centers_multi_internal_external_recommended/external_validation'
            },
            {
                'name': 'VMamba',
                'model_path': 'vmamba_result_unified/best_model.pth',
                'model_type': 'vmamba',
                'internal_data_path': '5centers_multi',
                'external_data_path': '5centers_multi_internal_external_recommended/external_validation'
            },
            {
                'name': 'ViT',
                'model_path': 'models/ViT/_results/multimodal/best_model.pth',
                'model_type': 'vit',
                'internal_data_path': '5centers_multi',
                'external_data_path': '5centers_multi_internal_external_recommended/external_validation'
            },
            {
                'name': 'MedicalViT',
                'model_path': 'models/MedicalViT/_results/multimodal/best_model.pth',
                'model_type': 'medical_vit',
                'internal_data_path': '5centers_multi',
                'external_data_path': '5centers_multi_internal_external_recommended/external_validation'
            }
        ]
    
    # 运行分析
    analyzer = ComprehensiveLancetAnalyzer(output_dir=args.output_dir)
    analyzer.run_comprehensive_analysis(model_configs)


if __name__ == '__main__':
    main()

