#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
外部验证评估脚本
用于评估模型在独立外部验证集上的性能
符合Lancet期刊发表要求
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    brier_score_loss
)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    # 如果sklearn版本较旧，使用替代方法
    def calibration_curve(y_true, y_prob, n_bins=10):
        from sklearn.metrics import brier_score_loss
        import numpy as np
        # 简单的校准曲线计算
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        fraction_of_positives = []
        mean_predicted_value = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                fraction_of_positives.append(y_true[in_bin].mean())
                mean_predicted_value.append(y_prob[in_bin].mean())
            else:
                fraction_of_positives.append(0)
                mean_predicted_value.append((bin_lower + bin_upper) / 2)
        
        return np.array(fraction_of_positives), np.array(mean_predicted_value)
from scipy import stats
from analysis.bootstrap_confidence_intervals import BootstrapCI
from analysis.decision_curve_analysis import DecisionCurveAnalyzer
from analysis.subgroup_analysis import SubgroupAnalyzer


class ExternalValidationEvaluator:
    """外部验证评估器"""
    
    def __init__(self, model_path: str, model_type: str = 'swint',
                 device: str = 'cuda', n_bootstrap: int = 2000):
        """
        Args:
            model_path: 训练好的模型路径
            model_type: 模型类型 ('swint', 'cnn', 'vmamba')
            device: 设备
            n_bootstrap: Bootstrap重采样次数
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.n_bootstrap = n_bootstrap
        self.bootstrap_ci = BootstrapCI(n_bootstrap=n_bootstrap)
        
        # 加载模型
        self.model = self._load_model()
        
    def _load_model(self):
        """加载训练好的模型"""
        print(f"🔄 加载模型: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 根据模型类型创建模型
        if self.model_type.lower() in ['swint', 'swin']:
            from models.SwinT.swin_multimodal_model import SwinTMultimodalTransformer
            # 从checkpoint或默认配置创建模型
            model = SwinTMultimodalTransformer(
                num_classes=2,
                embed_dim=768,
                num_heads=12,
                dropout=0.2,
                clinical_dim=7,
                oct_num_frames=48,
                col_num_frames=3,
                swin_name='swin_tiny_patch4_window7_224',
                pretrained=False,
                input_size=224,
                use_frame_attention=False,
            ).to(self.device)
        elif self.model_type.lower() == 'vit':
            from models.ViT.vit_multimodal_model import ViTMultimodalTransformer
            model = ViTMultimodalTransformer(
                num_classes=2,
                embed_dim=768,
                num_heads=12,
                dropout=0.2,
                clinical_dim=7,
                oct_num_frames=48,
                col_num_frames=3,
                vit_name='vit_base_patch16_224',
                pretrained=False,
                input_size=224,
                use_frame_attention=False,
            ).to(self.device)
        elif self.model_type.lower() == 'medical_vit':
            from models.MedicalViT.medical_vit_multimodal_model import MedicalViTMultimodalTransformer
            model = MedicalViTMultimodalTransformer(
                num_classes=2,
                embed_dim=768,
                num_heads=12,
                dropout=0.2,
                clinical_dim=7,
                oct_num_frames=48,
                col_num_frames=3,
                vit_name='vit_base_patch16_224',
                pretrained=False,
                input_size=224,
                use_frame_attention=False,
                use_medical_pretrained=True,
            ).to(self.device)
        elif self.model_type.lower() == 'cnn':
            from models.cnn_multimodal_model import CNNMultimodalTransformer
            model = CNNMultimodalTransformer(
                num_classes=2,
                embed_dim=1280,
                num_heads=20,
                dropout=0.3,
                clinical_dim=7,
                oct_num_frames=120,
                col_num_frames=3
            ).to(self.device)
        elif self.model_type.lower() == 'vmamba':
            from models.vmamba_multimodal_model import VMambaMultimodalTransformer
            model = VMambaMultimodalTransformer(
                num_classes=2,
                embed_dim=1024,
                num_heads=16,
                dropout=0.1,
                clinical_dim=7,
                oct_num_frames=120,
                col_num_frames=3,
                img_size=224,
                patch_size=16,
                depth=10,
                d_state=16
            ).to(self.device)
        else:
            raise ValueError(f"未知的模型类型: {self.model_type}")
        
        # 加载权重
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 非严格加载
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                         if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        model.eval()
        
        print(f"✅ 模型加载成功")
        return model
    
    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        在外部验证集上预测
        
        Returns:
            (y_true, y_pred, y_probs)
        """
        print(f"🔮 生成预测结果...")
        
        all_labels = []
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    oct_images = batch.get('oct_images')
                    col_images = batch.get('col_images')
                    clinical_features = batch.get('clinical_features')
                    if clinical_features is None:
                        clinical_features = batch.get('clinical')
                    labels = batch.get('label')
                    if labels is None:
                        labels = batch.get('labels')
                else:
                    if len(batch) >= 4:
                        oct_images, col_images, clinical_features, labels = batch[:4]
                    else:
                        continue
                
                # 检查数据是否有效
                if oct_images is None or col_images is None or clinical_features is None or labels is None:
                    continue
                
                oct_images = oct_images.to(self.device)
                col_images = col_images.to(self.device)
                clinical_features = clinical_features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(oct_images, col_images, clinical_features)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        if len(all_labels) == 0:
            print("❌ 警告: 没有成功预测任何样本，请检查数据加载")
            return np.array([]), np.array([]), np.array([])
        
        y_true = np.array(all_labels)
        y_probs = np.array(all_probs)
        y_pred = np.array(all_preds)
        
        print(f"✅ 预测完成: {len(y_true)} 个样本")
        print(f"   预测概率范围: [{y_probs.min():.4f}, {y_probs.max():.4f}]")
        print(f"   真实标签分布: 阳性={np.sum(y_true==1)}, 阴性={np.sum(y_true==0)}")
        return y_true, y_pred, y_probs
    
    def evaluate_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_probs: np.ndarray) -> Dict:
        """
        评估模型性能（带95% CI）
        
        Returns:
            包含所有指标及其CI的字典
        """
        print(f"\n📊 计算性能指标（Bootstrap CI）...")
        
        # 计算所有指标的Bootstrap CI
        results = self.bootstrap_ci.calculate_all_metrics_ci(y_true, y_pred, y_probs)
        
        # 添加校准指标
        try:
            if len(y_probs) > 0 and len(y_true) > 0:
                brier = brier_score_loss(y_true, y_probs)
                results['brier_score'] = {
                    'mean': float(brier),
                    'ci_lower': None,  # Brier Score通常不报告CI
                    'ci_upper': None,
                    'formatted': f"{brier:.4f}"
                }
            else:
                results['brier_score'] = {
                    'mean': None,
                    'ci_lower': None,
                    'ci_upper': None,
                    'formatted': "N/A"
                }
        except Exception as e:
            print(f"⚠️  Brier Score计算失败: {e}")
            results['brier_score'] = {
                'mean': None,
                'ci_lower': None,
                'ci_upper': None,
                'formatted': "N/A"
            }
        
        # 计算校准曲线
        try:
            if len(y_probs) > 0 and len(y_true) > 0:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_probs, n_bins=10
                )
                results['calibration_curve'] = {
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist()
                }
            else:
                results['calibration_curve'] = None
        except Exception as e:
            print(f"⚠️  校准曲线计算失败: {e}")
            results['calibration_curve'] = None
        
        return results
    
    def compare_with_training(self, training_results: Dict,
                            external_results: Dict) -> Dict:
        """
        与训练集性能比较
        
        Args:
            training_results: 训练集性能结果
            external_results: 外部验证集性能结果
        
        Returns:
            比较结果
        """
        print(f"\n📈 与训练集性能比较...")
        
        comparison = {}
        
        for metric in ['auc', 'accuracy', 'sensitivity', 'specificity', 'precision', 'npv', 'f1_score']:
            if metric in training_results and metric in external_results:
                train_value = training_results[metric].get('mean', 0)
                ext_value = external_results[metric].get('mean', 0)
                diff = ext_value - train_value
                
                comparison[metric] = {
                    'training': train_value,
                    'external': ext_value,
                    'difference': diff,
                    'relative_change': (diff / train_value * 100) if train_value > 0 else 0,
                    'acceptable': abs(diff) < 0.05 if metric == 'auc' else abs(diff) < 0.10
                }
        
        return comparison
    
    def compare_with_clinical_baseline(self, y_true: np.ndarray,
                                     clinical_data: pd.DataFrame) -> Dict:
        """
        与临床基线方法比较
        
        Args:
            y_true: 真实标签
            clinical_data: 包含HPV、TCT等临床数据的DataFrame
        
        Returns:
            比较结果
        """
        print(f"\n🏥 与临床基线方法比较...")
        
        comparison = {}
        
        # HPV单独
        if 'HPV' in clinical_data.columns:
            hpv_pred = (clinical_data['HPV'] == 1).astype(int)
            hpv_auc = roc_auc_score(y_true, hpv_pred) if len(np.unique(hpv_pred)) > 1 else np.nan
            hpv_sens = recall_score(y_true, hpv_pred, zero_division=0)
            hpv_spec = (1 - recall_score(1 - y_true, 1 - hpv_pred, zero_division=0))
            
            comparison['HPV'] = {
                'auc': hpv_auc,
                'sensitivity': hpv_sens,
                'specificity': hpv_spec
            }
        
        # TCT单独
        if 'TCT' in clinical_data.columns:
            tct_pred = (clinical_data['TCT'] == 1).astype(int)
            tct_auc = roc_auc_score(y_true, tct_pred) if len(np.unique(tct_pred)) > 1 else np.nan
            tct_sens = recall_score(y_true, tct_pred, zero_division=0)
            tct_spec = (1 - recall_score(1 - y_true, 1 - tct_pred, zero_division=0))
            
            comparison['TCT'] = {
                'auc': tct_auc,
                'sensitivity': tct_sens,
                'specificity': tct_spec
            }
        
        # HPV+TCT联合（任一阳性即为阳性）
        if 'HPV' in clinical_data.columns and 'TCT' in clinical_data.columns:
            combined_pred = ((clinical_data['HPV'] == 1) | (clinical_data['TCT'] == 1)).astype(int)
            combined_auc = roc_auc_score(y_true, combined_pred) if len(np.unique(combined_pred)) > 1 else np.nan
            combined_sens = recall_score(y_true, combined_pred, zero_division=0)
            combined_spec = (1 - recall_score(1 - y_true, 1 - combined_pred, zero_division=0))
            
            comparison['HPV+TCT'] = {
                'auc': combined_auc,
                'sensitivity': combined_sens,
                'specificity': combined_spec
            }
        
        return comparison
    
    def generate_report(self, results: Dict, output_dir: str, model_name: str):
        """
        生成外部验证报告
        
        Args:
            results: 评估结果
            output_dir: 输出目录
            model_name: 模型名称
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON结果
        results_file = os.path.join(output_dir, 'external_validation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        report_file = os.path.join(output_dir, 'external_validation_report.md')
        self._generate_markdown_report(results, report_file, model_name)
        
        print(f"\n✅ 外部验证报告已保存到: {output_dir}")
    
    def _generate_markdown_report(self, results: Dict, output_file: str, model_name: str):
        """生成Markdown格式报告"""
        
        report = f"""# {model_name} 外部验证报告

## 1. 性能指标（95% 置信区间）

| 指标 | 值 (95% CI) |
|------|------------|
| AUC | {results.get('auc', {}).get('formatted', 'N/A')} |
| Accuracy | {results.get('accuracy', {}).get('formatted', 'N/A')} |
| Sensitivity | {results.get('sensitivity', {}).get('formatted', 'N/A')} |
| Specificity | {results.get('specificity', {}).get('formatted', 'N/A')} |
| Precision (PPV) | {results.get('precision', {}).get('formatted', 'N/A')} |
| NPV | {results.get('npv', {}).get('formatted', 'N/A')} |
| F1-Score | {results.get('f1_score', {}).get('formatted', 'N/A')} |
| Brier Score | {results.get('brier_score', {}).get('formatted', 'N/A')} |

## 2. 与训练集性能比较

{self._format_comparison_table(results.get('training_comparison', {}))}

## 3. 与临床基线比较

{self._format_clinical_comparison(results.get('clinical_comparison', {}))}

## 4. 亚组分析

见 `subgroup_analysis.csv` 和 `forest_plot_*.png`

## 5. 决策曲线分析

见 `decision_curve.png`

---

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
    
    def _format_comparison_table(self, comparison: Dict) -> str:
        """格式化比较表格"""
        if not comparison:
            return "无比较数据"
        
        table = "| 指标 | 训练集 | 外部验证集 | 差异 | 相对变化 | 可接受 |\n"
        table += "|------|--------|------------|------|----------|--------|\n"
        
        for metric, data in comparison.items():
            table += f"| {metric.upper()} | {data['training']:.4f} | {data['external']:.4f} | "
            table += f"{data['difference']:.4f} | {data['relative_change']:.2f}% | "
            table += f"{'✅' if data['acceptable'] else '❌'} |\n"
        
        return table
    
    def _format_clinical_comparison(self, comparison: Dict) -> str:
        """格式化临床基线比较"""
        if not comparison:
            return "无临床基线数据"
        
        table = "| 方法 | AUC | Sensitivity | Specificity |\n"
        table += "|------|-----|-------------|-------------|\n"
        
        for method, data in comparison.items():
            table += f"| {method} | {data.get('auc', 'N/A'):.4f} | "
            table += f"{data.get('sensitivity', 'N/A'):.4f} | "
            table += f"{data.get('specificity', 'N/A'):.4f} |\n"
        
        return table


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='外部验证评估')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径')
    parser.add_argument('--external_data_path', type=str, required=True,
                       help='外部验证数据路径')
    parser.add_argument('--model_type', type=str, default='swint',
                       choices=['swint', 'cnn', 'vmamba'],
                       help='模型类型')
    parser.add_argument('--training_results', type=str, default=None,
                       help='训练集结果JSON文件（用于比较）')
    parser.add_argument('--output_dir', type=str, default='analysis/external_validation_results',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = ExternalValidationEvaluator(
        args.model_path,
        args.model_type,
        args.device
    )
    
    # 加载外部验证数据
    from utils.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
    from torch.utils.data import DataLoader
    
    # 检查外部验证数据集结构
    # 可能是 external_validation/ 或 test/ 结构
    if os.path.exists(os.path.join(args.external_data_path, 'external_labels.csv')):
        # 结构: external_validation/external_labels.csv
        external_root = args.external_data_path
        labels_file = os.path.join(args.external_data_path, 'external_labels.csv')
        # 创建临时test目录结构
        temp_test_dir = os.path.join(args.external_data_path, 'test')
        if not os.path.exists(temp_test_dir):
            os.makedirs(temp_test_dir, exist_ok=True)
            # 创建符号链接
            if os.path.exists(os.path.join(args.external_data_path, 'oct')):
                oct_link = os.path.join(temp_test_dir, 'oct')
                if not os.path.exists(oct_link):
                    os.symlink(os.path.join(args.external_data_path, 'oct'), oct_link)
            if os.path.exists(os.path.join(args.external_data_path, 'col')):
                col_link = os.path.join(temp_test_dir, 'col')
                if not os.path.exists(col_link):
                    os.symlink(os.path.join(args.external_data_path, 'col'), col_link)
            # 复制标签文件
            test_labels_file = os.path.join(temp_test_dir, 'test_labels.csv')
            if not os.path.exists(test_labels_file):
                import shutil
                shutil.copy(labels_file, test_labels_file)
        data_path = args.external_data_path
        root = temp_test_dir
    elif os.path.exists(os.path.join(args.external_data_path, 'test')):
        # 结构: external_data_path/test/
        data_path = args.external_data_path
        root = os.path.join(args.external_data_path, 'test')
    else:
        # 直接是数据目录
        data_path = os.path.dirname(args.external_data_path)
        root = args.external_data_path
    
    class Args:
        def __init__(self, data_path):
            self.data_path = data_path
            self.input_size = 224
            self.oct_num_frames = 48 if args.model_type.lower() in ['swint', 'swin'] else 120
            self.oct_cache_dir = 'oct_cache_optimized'
            self.use_text_contrastive = False
            self.use_pretrained_backbones = True
            self.oct_points = 12
            self.oct_frames_per_point = 10
    
    data_args = Args(data_path)
    
    external_dataset = EnhancedMultimodalCervicalDataset(
        root=root,
        is_train='test',
        args=data_args,
        transform=None,
        use_enhanced_oct=True,
        cache_oct_features=True
    )
    
    external_loader = DataLoader(
        external_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 预测
    y_true, y_pred, y_probs = evaluator.predict(external_loader)
    
    # 保存预测结果（用于后续分析）
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'val_labels.npy'), y_true)
    np.save(os.path.join(args.output_dir, 'val_probs.npy'), y_probs)
    np.save(os.path.join(args.output_dir, 'val_preds.npy'), y_pred)
    print(f"✅ 预测结果已保存到: {args.output_dir}")
    
    # 评估性能
    performance_results = evaluator.evaluate_performance(y_true, y_pred, y_probs)
    
    # 与训练集比较
    training_comparison = None
    if args.training_results and os.path.exists(args.training_results):
        with open(args.training_results, 'r') as f:
            training_results = json.load(f)
        training_comparison = evaluator.compare_with_training(
            training_results, performance_results
        )
        performance_results['training_comparison'] = training_comparison
    
    # 与临床基线比较（如果有临床数据）
    clinical_comparison = None
    try:
        clinical_data_file = os.path.join(args.external_data_path, 'test_labels.csv')
        if os.path.exists(clinical_data_file):
            clinical_data = pd.read_csv(clinical_data_file)
            if len(clinical_data) == len(y_true):
                clinical_comparison = evaluator.compare_with_clinical_baseline(
                    y_true, clinical_data
                )
                performance_results['clinical_comparison'] = clinical_comparison
    except Exception as e:
        print(f"⚠️  临床基线比较失败: {e}")
    
    # 亚组分析（如果有元数据）
    try:
        metadata_file = os.path.join(args.external_data_path, 'test_labels.csv')
        if os.path.exists(metadata_file):
            metadata = pd.read_csv(metadata_file)
            if len(metadata) == len(y_true):
                subgroup_analyzer = SubgroupAnalyzer()
                subgroup_results = subgroup_analyzer.analyze_all_subgroups(
                    y_true, y_pred, y_probs, metadata
                )
                subgroup_file = os.path.join(args.output_dir, 'subgroup_analysis.csv')
                subgroup_results.to_csv(subgroup_file, index=False)
                performance_results['subgroup_analysis'] = subgroup_results.to_dict('records')
    except Exception as e:
        print(f"⚠️  亚组分析失败: {e}")
    
    # DCA分析
    try:
        dca_analyzer = DecisionCurveAnalyzer()
        dca_results = dca_analyzer.analyze(y_true, y_probs, args.model_type.upper())
        dca_file = os.path.join(args.output_dir, 'dca_results.csv')
        dca_results.to_csv(dca_file, index=False)
        
        dca_plot_file = os.path.join(args.output_dir, 'decision_curve.png')
        dca_analyzer.plot_decision_curve(dca_results, dca_plot_file,
                                        f"External Validation DCA - {args.model_type.upper()}")
    except Exception as e:
        print(f"⚠️  DCA分析失败: {e}")
    
    # 生成报告
    evaluator.generate_report(performance_results, args.output_dir, args.model_type.upper())
    
    print(f"\n✅ 外部验证评估完成！")
    print(f"   结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()


