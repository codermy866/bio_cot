#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Ensemble 和 Conformal Prediction 脚本
用于不确定性量化和预测区间估计

特点：
- Deep Ensemble 实现
- Conformal Prediction 预测区间
- MC-Dropout 不确定性估计
- 校准评估
- 不确定性可视化
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DeepEnsemble:
    """Deep Ensemble 实现"""
    
    def __init__(self, model_class, model_configs: List[Dict], device: str = 'cuda'):
        """
        初始化Deep Ensemble
        
        Args:
            model_class: 模型类
            model_configs: 模型配置列表
            device: 设备
        """
        self.model_class = model_class
        self.model_configs = model_configs
        self.device = device
        self.models = []
        
    def load_models(self, model_paths: List[str]):
        """加载多个模型"""
        self.models = []
        
        for i, (config, path) in enumerate(zip(self.model_configs, model_paths)):
            print(f"🔄 加载模型 {i+1}/{len(model_paths)}: {path}")
            
            # 创建模型
            model = self.model_class(**config)
            
            # 加载权重
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(self.device)
            model.eval()
            
            self.models.append(model)
        
        print(f"✅ 成功加载 {len(self.models)} 个模型")
    
    def predict_ensemble(self, oct_images: torch.Tensor, col_images: torch.Tensor, 
                       clinical_features: torch.Tensor, n_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        Ensemble预测
        
        Args:
            oct_images: OCT图像
            col_images: 阴道镜图像
            clinical_features: 临床特征
            n_samples: 每个模型的采样次数
            
        Returns:
            预测结果字典
        """
        all_logits = []
        all_probs = []
        
        with torch.no_grad():
            for model in self.models:
                model_logits = []
                model_probs = []
                
                # 多次前向传播（用于MC-Dropout）
                for _ in range(n_samples):
                    logits = model(oct_images, col_images, clinical_features)
                    probs = torch.softmax(logits, dim=1)
                    
                    model_logits.append(logits)
                    model_probs.append(probs)
                
                # 模型内平均
                model_logits = torch.stack(model_logits).mean(dim=0)
                model_probs = torch.stack(model_probs).mean(dim=0)
                
                all_logits.append(model_logits)
                all_probs.append(model_probs)
        
        # Ensemble平均
        ensemble_logits = torch.stack(all_logits).mean(dim=0)
        ensemble_probs = torch.stack(all_probs).mean(dim=0)
        
        # 计算不确定性（方差）
        prob_variance = torch.stack(all_probs).var(dim=0)
        uncertainty = prob_variance.sum(dim=1)  # 总不确定性
        
        return {
            'logits': ensemble_logits,
            'probabilities': ensemble_probs,
            'uncertainty': uncertainty,
            'individual_logits': all_logits,
            'individual_probs': all_probs
        }
    
    def evaluate_ensemble(self, dataloader: DataLoader) -> Dict[str, Any]:
        """评估Ensemble性能"""
        all_labels = []
        all_probs = []
        all_uncertainties = []
        
        print("🔄 评估Deep Ensemble...")
        
        for batch_idx, batch in enumerate(dataloader):
            oct_img, col_img, clinical, label = batch
            oct_img = oct_img.to(self.device).float()
            col_img = col_img.to(self.device).float()
            clinical = clinical.to(self.device).float()
            label = label.long().to(self.device)
            
            # Ensemble预测
            results = self.predict_ensemble(oct_img, col_img, clinical)
            
            all_labels.append(label.cpu())
            all_probs.append(results['probabilities'].cpu())
            all_uncertainties.append(results['uncertainty'].cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  批次 {batch_idx + 1}/{len(dataloader)}")
        
        # 合并结果
        labels = torch.cat(all_labels, dim=0).numpy()
        probs = torch.cat(all_probs, dim=0).numpy()
        uncertainties = torch.cat(all_uncertainties, dim=0).numpy()
        
        # 计算指标
        preds = probs.argmax(axis=1)
        pos_probs = probs[:, 1]
        
        # 基础指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        # AUC
        try:
            auc = roc_auc_score(labels, pos_probs)
        except ValueError:
            auc = 0.5
        
        # Brier Score
        brier = brier_score_loss(labels, pos_probs)
        
        # 不确定性统计
        uncertainty_mean = np.mean(uncertainties)
        uncertainty_std = np.std(uncertainties)
        
        return {
            'labels': labels,
            'probabilities': probs,
            'uncertainties': uncertainties,
            'predictions': preds,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'brier_score': brier,
            'uncertainty_mean': uncertainty_mean,
            'uncertainty_std': uncertainty_std
        }


class ConformalPredictor:
    """Conformal Prediction 实现"""
    
    def __init__(self, alpha: float = 0.1):
        """
        初始化Conformal Predictor
        
        Args:
            alpha: 显著性水平 (1 - 置信度)
        """
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile = None
        
    def fit(self, labels: np.ndarray, probabilities: np.ndarray):
        """
        拟合Conformal Predictor
        
        Args:
            labels: 真实标签
            probabilities: 预测概率
        """
        # 计算conformity scores (使用预测概率的置信度)
        pos_probs = probabilities[:, 1]
        conformity_scores = np.where(labels == 1, pos_probs, 1 - pos_probs)
        
        # 计算分位数
        self.quantile = np.quantile(conformity_scores, 1 - self.alpha)
        
        print(f"✅ Conformal Predictor拟合完成，分位数: {self.quantile:.3f}")
    
    def predict_intervals(self, probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测置信区间
        
        Args:
            probabilities: 预测概率
            
        Returns:
            置信区间下界和上界
        """
        pos_probs = probabilities[:, 1]
        
        # 计算置信区间
        lower_bounds = np.maximum(0, pos_probs - self.quantile)
        upper_bounds = np.minimum(1, pos_probs + self.quantile)
        
        return lower_bounds, upper_bounds
    
    def evaluate_coverage(self, labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
        """
        评估覆盖率
        
        Args:
            labels: 真实标签
            probabilities: 预测概率
            
        Returns:
            覆盖率统计
        """
        lower_bounds, upper_bounds = self.predict_intervals(probabilities)
        
        # 计算覆盖率
        coverage = np.mean((labels >= lower_bounds) & (labels <= upper_bounds))
        
        # 计算区间宽度
        interval_width = np.mean(upper_bounds - lower_bounds)
        
        return {
            'coverage': coverage,
            'expected_coverage': 1 - self.alpha,
            'interval_width': interval_width,
            'coverage_gap': abs(coverage - (1 - self.alpha))
        }


class CalibrationEvaluator:
    """校准评估器"""
    
    def __init__(self, n_bins: int = 10):
        """
        初始化校准评估器
        
        Args:
            n_bins: 分箱数量
        """
        self.n_bins = n_bins
        
    def evaluate_calibration(self, labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
        """
        评估校准性能
        
        Args:
            labels: 真实标签
            probabilities: 预测概率
            
        Returns:
            校准评估结果
        """
        pos_probs = probabilities[:, 1]
        
        # 分箱
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 找到属于当前bin的样本
            in_bin = (pos_probs > bin_lower) & (pos_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # 计算bin内的准确率和置信度
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = pos_probs[in_bin].mean()
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        # 计算ECE (Expected Calibration Error)
        ece = 0
        for i in range(self.n_bins):
            if bin_counts[i] > 0:
                ece += abs(bin_accuracies[i] - bin_confidences[i]) * bin_counts[i]
        ece /= len(labels)
        
        # 计算MCE (Maximum Calibration Error)
        mce = max([abs(acc - conf) for acc, conf in zip(bin_accuracies, bin_confidences)])
        
        return {
            'ece': ece,
            'mce': mce,
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'bin_boundaries': bin_boundaries
        }
    
    def plot_reliability_diagram(self, labels: np.ndarray, probabilities: np.ndarray,
                               save_path: str = 'reliability_diagram.png'):
        """绘制可靠性图"""
        results = self.evaluate_calibration(labels, probabilities)
        
        plt.figure(figsize=(8, 8))
        
        # 绘制可靠性图
        bin_centers = [(results['bin_boundaries'][i] + results['bin_boundaries'][i+1]) / 2 
                      for i in range(self.n_bins)]
        
        plt.bar(bin_centers, results['bin_accuracies'], width=0.1, alpha=0.7, 
               label='Accuracy')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Reliability Diagram (ECE={results["ece"]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可靠性图已保存到: {save_path}")


class UncertaintyAnalyzer:
    """不确定性分析器"""
    
    def __init__(self, ensemble: DeepEnsemble, conformal_predictor: ConformalPredictor,
                 calibration_evaluator: CalibrationEvaluator):
        """
        初始化不确定性分析器
        
        Args:
            ensemble: Deep Ensemble
            conformal_predictor: Conformal Predictor
            calibration_evaluator: 校准评估器
        """
        self.ensemble = ensemble
        self.conformal_predictor = conformal_predictor
        self.calibration_evaluator = calibration_evaluator
        
    def analyze_uncertainty(self, dataloader: DataLoader) -> Dict[str, Any]:
        """分析不确定性"""
        print("🔄 开始不确定性分析...")
        
        # Ensemble评估
        ensemble_results = self.ensemble.evaluate_ensemble(dataloader)
        
        # 拟合Conformal Predictor
        self.conformal_predictor.fit(ensemble_results['labels'], ensemble_results['probabilities'])
        
        # 评估覆盖率
        coverage_results = self.conformal_predictor.evaluate_coverage(
            ensemble_results['labels'], ensemble_results['probabilities']
        )
        
        # 校准评估
        calibration_results = self.calibration_evaluator.evaluate_calibration(
            ensemble_results['labels'], ensemble_results['probabilities']
        )
        
        # 预测区间
        lower_bounds, upper_bounds = self.conformal_predictor.predict_intervals(
            ensemble_results['probabilities']
        )
        
        return {
            'ensemble_results': ensemble_results,
            'coverage_results': coverage_results,
            'calibration_results': calibration_results,
            'prediction_intervals': {
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds
            }
        }
    
    def plot_uncertainty_analysis(self, results: Dict[str, Any], 
                                 save_path: str = 'uncertainty_analysis.png'):
        """绘制不确定性分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 不确定性分布
        ax1 = axes[0, 0]
        uncertainties = results['ensemble_results']['uncertainties']
        ax1.hist(uncertainties, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Uncertainty')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Uncertainty Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. 预测区间
        ax2 = axes[0, 1]
        labels = results['ensemble_results']['labels']
        probs = results['ensemble_results']['probabilities'][:, 1]
        lower_bounds = results['prediction_intervals']['lower_bounds']
        upper_bounds = results['prediction_intervals']['upper_bounds']
        
        # 按不确定性排序
        uncertainty_order = np.argsort(uncertainties)
        sorted_labels = labels[uncertainty_order]
        sorted_probs = probs[uncertainty_order]
        sorted_lower = lower_bounds[uncertainty_order]
        sorted_upper = upper_bounds[uncertainty_order]
        
        # 绘制预测区间
        n_samples = min(100, len(sorted_labels))  # 限制样本数量
        indices = np.linspace(0, len(sorted_labels)-1, n_samples, dtype=int)
        
        ax2.errorbar(range(n_samples), sorted_probs[indices], 
                    yerr=[sorted_probs[indices] - sorted_lower[indices],
                          sorted_upper[indices] - sorted_probs[indices]],
                    fmt='o', alpha=0.6, capsize=3)
        ax2.scatter(range(n_samples), sorted_labels[indices], 
                   color='red', alpha=0.8, s=20, label='True Labels')
        ax2.set_xlabel('Sample Index (sorted by uncertainty)')
        ax2.set_ylabel('Probability')
        ax2.set_title('Prediction Intervals')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 校准图
        ax3 = axes[1, 0]
        calibration_results = results['calibration_results']
        bin_centers = [(calibration_results['bin_boundaries'][i] + 
                       calibration_results['bin_boundaries'][i+1]) / 2 
                      for i in range(len(calibration_results['bin_accuracies']))]
        
        ax3.bar(bin_centers, calibration_results['bin_accuracies'], 
               width=0.1, alpha=0.7, label='Accuracy')
        ax3.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Accuracy')
        ax3.set_title(f'Calibration (ECE={calibration_results["ece"]:.3f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 不确定性vs准确率
        ax4 = axes[1, 1]
        correct_predictions = (results['ensemble_results']['predictions'] == labels)
        ax4.scatter(uncertainties[correct_predictions], 
                   probs[correct_predictions], 
                   alpha=0.6, label='Correct', color='green')
        ax4.scatter(uncertainties[~correct_predictions], 
                   probs[~correct_predictions], 
                   alpha=0.6, label='Incorrect', color='red')
        ax4.set_xlabel('Uncertainty')
        ax4.set_ylabel('Predicted Probability')
        ax4.set_title('Uncertainty vs Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 不确定性分析图已保存到: {save_path}")
    
    def save_results(self, results: Dict[str, Any], save_path: str = 'uncertainty_results.json'):
        """保存结果"""
        # 转换为可序列化格式
        serializable_results = {}
        
        for key, value in results.items():
            if key == 'ensemble_results':
                serializable_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                    if k not in ['labels', 'probabilities', 'uncertainties', 'predictions']
                }
            elif key == 'prediction_intervals':
                serializable_results[key] = {
                    k: v.tolist() for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ 不确定性分析结果已保存到: {save_path}")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='不确定性分析')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                      help='模型路径列表')
    parser.add_argument('--data_path', type=str, default='5centers_multi',
                      help='数据路径')
    parser.add_argument('--model_type', type=str, choices=['cnn', 'vmamba'], 
                      default='cnn', help='模型类型')
    parser.add_argument('--clinical_dim', type=int, default=8,
                      help='临床特征维度')
    parser.add_argument('--output_dir', type=str, default='uncertainty_output',
                      help='输出目录')
    parser.add_argument('--alpha', type=float, default=0.1,
                      help='Conformal prediction显著性水平')
    parser.add_argument('--n_bins', type=int, default=10,
                      help='校准评估分箱数量')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 选择模型类
    if args.model_type == 'cnn':
        from cnn_multimodal_model import CNNMultimodalTransformer
        model_class = CNNMultimodalTransformer
    else:
        from vmamba_multimodal_model import VMambaMultimodalTransformer
        model_class = VMambaMultimodalTransformer
    
    # 模型配置
    model_configs = [
        {'num_classes': 2, 'clinical_dim': args.clinical_dim, 'dropout': 0.1}
        for _ in range(len(args.model_paths))
    ]
    
    # 创建Deep Ensemble
    ensemble = DeepEnsemble(model_class, model_configs)
    ensemble.load_models(args.model_paths)
    
    # 加载数据
    from enhanced_multimodal_dataset import build_enhanced_dataset
    
    class Args:
        def __init__(self):
            self.data_path = args.data_path
            self.input_size = 224
            self.oct_num_frames = 48
            self.oct_cache_dir = 'oct_cache'
            self.use_text_contrastive = False
    
    args_data = Args()
    test_dataset = build_enhanced_dataset('test', args_data)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # 创建分析器
    conformal_predictor = ConformalPredictor(alpha=args.alpha)
    calibration_evaluator = CalibrationEvaluator(n_bins=args.n_bins)
    uncertainty_analyzer = UncertaintyAnalyzer(ensemble, conformal_predictor, calibration_evaluator)
    
    # 运行分析
    print("🚀 开始不确定性分析...")
    results = uncertainty_analyzer.analyze_uncertainty(test_loader)
    
    # 生成输出
    uncertainty_analyzer.plot_uncertainty_analysis(results,
                                                 os.path.join(args.output_dir, 'uncertainty_analysis.png'))
    
    calibration_evaluator.plot_reliability_diagram(
        results['ensemble_results']['labels'],
        results['ensemble_results']['probabilities'],
        os.path.join(args.output_dir, 'reliability_diagram.png')
    )
    
    uncertainty_analyzer.save_results(results,
                                    os.path.join(args.output_dir, 'uncertainty_results.json'))
    
    # 打印结果摘要
    print("\n📊 不确定性分析结果摘要:")
    print(f"Ensemble AUC: {results['ensemble_results']['auc']:.3f}")
    print(f"Ensemble Accuracy: {results['ensemble_results']['accuracy']:.3f}")
    print(f"Brier Score: {results['ensemble_results']['brier_score']:.3f}")
    print(f"ECE: {results['calibration_results']['ece']:.3f}")
    print(f"MCE: {results['calibration_results']['mce']:.3f}")
    print(f"Coverage: {results['coverage_results']['coverage']:.3f}")
    print(f"Expected Coverage: {results['coverage_results']['expected_coverage']:.3f}")
    print(f"Interval Width: {results['coverage_results']['interval_width']:.3f}")
    
    print("✅ 不确定性分析完成!")


if __name__ == '__main__':
    main()
