#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决策曲线分析 (Decision Curve Analysis, DCA) 脚本
用于评估临床决策的净获益，生成柳叶刀级别的决策曲线

特点：
- 计算不同阈值下的净获益
- 支持多模型对比
- 成本-效果分析
- 临床决策阈值推荐
- 敏感性分析
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DecisionCurveAnalyzer:
    """决策曲线分析器"""
    
    def __init__(self, cost_tp: float = 0.0, cost_fp: float = 0.0, 
                 cost_fn: float = 1.0, cost_tn: float = 0.0):
        """
        初始化DCA分析器
        
        Args:
            cost_tp: 真阳性成本 (通常为0，因为正确诊断)
            cost_fp: 假阳性成本 (过度治疗成本)
            cost_fn: 假阴性成本 (漏诊成本，通常设为1)
            cost_tn: 真阴性成本 (通常为0，因为正确排除)
        """
        self.cost_tp = cost_tp
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.cost_tn = cost_tn
        
    def calculate_net_benefit(self, labels: np.ndarray, probabilities: np.ndarray, 
                            threshold: float) -> float:
        """
        计算净获益
        
        Args:
            labels: 真实标签 (0/1)
            probabilities: 预测概率
            threshold: 决策阈值
            
        Returns:
            净获益值
        """
        # 预测结果
        predictions = (probabilities >= threshold).astype(int)
        
        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        # 总样本数
        n = len(labels)
        
        # 净获益计算
        # NB = (TP/n) - (FP/n) * (pt/(1-pt)) - (FN/n) * ((1-pt)/pt)
        # 其中 pt 是阈值概率
        
        if threshold == 0:
            # 当阈值为0时，所有样本都被预测为阳性
            net_benefit = (tp + fp) / n - (fp / n) * (threshold / (1 - threshold + 1e-8))
        elif threshold == 1:
            # 当阈值为1时，所有样本都被预测为阴性
            net_benefit = (tn + fn) / n - (fn / n) * ((1 - threshold) / (threshold + 1e-8))
        else:
            # 标准净获益计算
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold)) - \
                         (fn / n) * ((1 - threshold) / threshold)
        
        return net_benefit
    
    def calculate_net_benefit_all_treat(self, labels: np.ndarray) -> float:
        """计算全部治疗的净获益"""
        n = len(labels)
        tp = np.sum(labels)  # 所有阳性样本
        fp = n - tp  # 所有阴性样本
        return tp / n - fp / n
    
    def calculate_net_benefit_no_treat(self, labels: np.ndarray) -> float:
        """计算全部不治疗的净获益"""
        n = len(labels)
        tn = n - np.sum(labels)  # 所有阴性样本
        fn = np.sum(labels)  # 所有阳性样本
        return tn / n - fn / n
    
    def analyze_single_model(self, labels: np.ndarray, probabilities: np.ndarray,
                           model_name: str = "Model") -> Dict[str, Any]:
        """
        分析单个模型的决策曲线
        
        Args:
            labels: 真实标签
            probabilities: 预测概率
            model_name: 模型名称
            
        Returns:
            分析结果字典
        """
        # 阈值范围
        thresholds = np.linspace(0.01, 0.99, 99)
        
        # 计算净获益
        net_benefits = []
        for threshold in thresholds:
            nb = self.calculate_net_benefit(labels, probabilities, threshold)
            net_benefits.append(nb)
        
        net_benefits = np.array(net_benefits)
        
        # 计算参考线
        nb_all_treat = self.calculate_net_benefit_all_treat(labels)
        nb_no_treat = self.calculate_net_benefit_no_treat(labels)
        
        # 找到最优阈值
        optimal_idx = np.argmax(net_benefits)
        optimal_threshold = thresholds[optimal_idx]
        optimal_nb = net_benefits[optimal_idx]
        
        return {
            'model_name': model_name,
            'thresholds': thresholds,
            'net_benefits': net_benefits,
            'nb_all_treat': nb_all_treat,
            'nb_no_treat': nb_no_treat,
            'optimal_threshold': optimal_threshold,
            'optimal_nb': optimal_nb,
            'labels': labels,
            'probabilities': probabilities
        }
    
    def analyze_multiple_models(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析多个模型的决策曲线对比
        
        Args:
            model_results: 模型结果列表，每个元素包含labels和probabilities
            
        Returns:
            多模型分析结果
        """
        results = []
        
        for i, result in enumerate(model_results):
            model_name = result.get('model_name', f'Model_{i+1}')
            labels = result['labels']
            probabilities = result['probabilities']
            
            analysis = self.analyze_single_model(labels, probabilities, model_name)
            results.append(analysis)
        
        return {
            'model_results': results,
            'thresholds': results[0]['thresholds'] if results else None
        }
    
    def plot_decision_curve(self, results: Dict[str, Any], 
                          save_path: str = 'decision_curve.png',
                          title: str = 'Decision Curve Analysis',
                          figsize: Tuple[int, int] = (10, 8)):
        """
        绘制决策曲线
        
        Args:
            results: 分析结果
            save_path: 保存路径
            title: 图表标题
            figsize: 图形大小
        """
        plt.figure(figsize=figsize)
        
        # 阈值
        thresholds = results['thresholds']
        
        # 绘制每个模型的决策曲线
        for result in results['model_results']:
            plt.plot(thresholds, result['net_benefits'], 
                    label=result['model_name'], linewidth=2)
            
            # 标记最优阈值
            plt.plot(result['optimal_threshold'], result['optimal_nb'], 
                    'o', markersize=8, color=plt.gca().lines[-1].get_color())
        
        # 绘制参考线
        first_result = results['model_results'][0]
        plt.axhline(y=first_result['nb_all_treat'], color='red', 
                   linestyle='--', alpha=0.7, label='Treat All')
        plt.axhline(y=first_result['nb_no_treat'], color='blue', 
                   linestyle='--', alpha=0.7, label='Treat None')
        
        # 设置图形属性
        plt.xlabel('Threshold Probability', fontsize=12)
        plt.ylabel('Net Benefit', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # 设置坐标轴范围
        plt.xlim(0, 1)
        y_min = min([min(r['net_benefits']) for r in results['model_results']])
        y_max = max([max(r['net_benefits']) for r in results['model_results']])
        plt.ylim(y_min - 0.05, y_max + 0.05)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 决策曲线已保存到: {save_path}")
    
    def plot_cost_effectiveness(self, results: Dict[str, Any],
                              cost_scenarios: List[Dict[str, float]],
                              save_path: str = 'cost_effectiveness.png'):
        """
        绘制成本-效果分析
        
        Args:
            results: 分析结果
            cost_scenarios: 成本场景列表
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        scenario_names = ['Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost']
        
        for i, (scenario, name) in enumerate(zip(cost_scenarios, scenario_names)):
            ax = axes[i]
            
            # 重新计算净获益（使用新的成本设置）
            analyzer = DecisionCurveAnalyzer(**scenario)
            
            for result in results['model_results']:
                labels = result['labels']
                probabilities = result['probabilities']
                thresholds = result['thresholds']
                
                net_benefits = []
                for threshold in thresholds:
                    nb = analyzer.calculate_net_benefit(labels, probabilities, threshold)
                    net_benefits.append(nb)
                
                ax.plot(thresholds, net_benefits, 
                       label=result['model_name'], linewidth=2)
            
            ax.set_xlabel('Threshold Probability')
            ax.set_ylabel('Net Benefit')
            ax.set_title(f'{name} Scenario')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(0, 1)
        
        plt.suptitle('Cost-Effectiveness Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 成本-效果分析已保存到: {save_path}")
    
    def generate_threshold_recommendations(self, results: Dict[str, Any],
                                         clinical_context: str = "General") -> Dict[str, Any]:
        """
        生成阈值推荐
        
        Args:
            results: 分析结果
            clinical_context: 临床上下文
            
        Returns:
            阈值推荐结果
        """
        recommendations = {}
        
        for result in results['model_results']:
            model_name = result['model_name']
            
            # 找到不同净获益水平对应的阈值
            thresholds = result['thresholds']
            net_benefits = result['net_benefits']
            
            # 高净获益阈值（净获益 > 0.1）
            high_nb_indices = net_benefits > 0.1
            if np.any(high_nb_indices):
                high_nb_thresholds = thresholds[high_nb_indices]
                recommended_threshold = np.median(high_nb_thresholds)
            else:
                recommended_threshold = result['optimal_threshold']
            
            # 计算在该阈值下的性能指标
            optimal_idx = np.argmin(np.abs(thresholds - recommended_threshold))
            optimal_nb = net_benefits[optimal_idx]
            
            recommendations[model_name] = {
                'recommended_threshold': recommended_threshold,
                'net_benefit_at_threshold': optimal_nb,
                'optimal_threshold': result['optimal_threshold'],
                'max_net_benefit': result['optimal_nb'],
                'clinical_context': clinical_context
            }
        
        return recommendations
    
    def save_results(self, results: Dict[str, Any], 
                    recommendations: Dict[str, Any],
                    save_path: str = 'dca_results.json'):
        """保存分析结果"""
        output = {
            'analysis_results': results,
            'recommendations': recommendations,
            'cost_settings': {
                'cost_tp': self.cost_tp,
                'cost_fp': self.cost_fp,
                'cost_fn': self.cost_fn,
                'cost_tn': self.cost_tn
            }
        }
        
        # 转换为可序列化格式
        serializable_output = {}
        for key, value in output.items():
            if key == 'analysis_results':
                serializable_output[key] = {}
                for model_result in value['model_results']:
                    model_name = model_result['model_name']
                    serializable_output[key][model_name] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in model_result.items()
                        if k not in ['labels', 'probabilities']  # 排除大数组
                    }
            else:
                serializable_output[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(serializable_output, f, indent=2)
        
        print(f"✅ DCA结果已保存到: {save_path}")
    
    def generate_summary_table(self, results: Dict[str, Any],
                             recommendations: Dict[str, Any],
                             save_path: str = 'dca_summary.csv'):
        """生成汇总表格"""
        table_data = []
        
        for result in results['model_results']:
            model_name = result['model_name']
            rec = recommendations.get(model_name, {})
            
            table_data.append({
                'Model': model_name,
                'Optimal_Threshold': f"{result['optimal_threshold']:.3f}",
                'Max_Net_Benefit': f"{result['optimal_nb']:.3f}",
                'Recommended_Threshold': f"{rec.get('recommended_threshold', 0):.3f}",
                'NB_at_Recommended': f"{rec.get('net_benefit_at_threshold', 0):.3f}",
                'NB_Treat_All': f"{result['nb_all_treat']:.3f}",
                'NB_Treat_None': f"{result['nb_no_treat']:.3f}"
            })
        
        df = pd.DataFrame(table_data)
        df.to_csv(save_path, index=False)
        
        print(f"✅ DCA汇总表格已保存到: {save_path}")
        
        # 打印表格
        print("\n📊 决策曲线分析汇总:")
        print(df.to_string(index=False))


def load_model_predictions(model_path: str, data_path: str, model_type: str = 'cnn') -> Dict[str, Any]:
    """
    加载模型预测结果
    
    Args:
        model_path: 模型路径
        data_path: 数据路径
        model_type: 模型类型
        
    Returns:
        预测结果字典
    """
    import torch
    from torch.utils.data import DataLoader
    
    # 选择模型类
    if model_type == 'cnn':
        from cnn_multimodal_model import CNNMultimodalTransformer
        model_class = CNNMultimodalTransformer
    else:
        from vmamba_multimodal_model import VMambaMultimodalTransformer
        model_class = VMambaMultimodalTransformer
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model = model_class(num_classes=2, clinical_dim=8)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    # 加载数据
    from enhanced_multimodal_dataset import build_enhanced_dataset
    
    class Args:
        def __init__(self):
            self.data_path = data_path
            self.input_size = 224
            self.oct_num_frames = 48
            self.oct_cache_dir = 'oct_cache'
            self.use_text_contrastive = False
    
    args = Args()
    test_dataset = build_enhanced_dataset('test', args)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # 获取预测结果
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            oct_img, col_img, clinical, label = batch
            oct_img = oct_img.to(device).float()
            col_img = col_img.to(device).float()
            clinical = clinical.to(device).float()
            label = label.long().to(device)
            
            logits = model(oct_img, col_img, clinical)
            probs = torch.softmax(logits, dim=1)
            
            all_labels.append(label.cpu())
            all_probs.append(probs.cpu())
    
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.cat(all_probs, dim=0).numpy()
    
    return {
        'labels': labels,
        'probabilities': probs[:, 1],  # 阳性概率
        'model_name': f'{model_type.upper()}_Model'
    }


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='决策曲线分析')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, 
                      help='模型路径列表')
    parser.add_argument('--data_path', type=str, default='5centers_multi', 
                      help='数据路径')
    parser.add_argument('--model_types', type=str, nargs='+', default=['cnn'], 
                      help='模型类型列表')
    parser.add_argument('--output_dir', type=str, default='dca_output', 
                      help='输出目录')
    parser.add_argument('--cost_tp', type=float, default=0.0, 
                      help='真阳性成本')
    parser.add_argument('--cost_fp', type=float, default=0.0, 
                      help='假阳性成本')
    parser.add_argument('--cost_fn', type=float, default=1.0, 
                      help='假阴性成本')
    parser.add_argument('--cost_tn', type=float, default=0.0, 
                      help='真阴性成本')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型预测结果
    model_results = []
    for model_path, model_type in zip(args.model_paths, args.model_types):
        print(f"🔄 加载模型: {model_path}")
        result = load_model_predictions(model_path, args.data_path, model_type)
        model_results.append(result)
    
    # 创建DCA分析器
    analyzer = DecisionCurveAnalyzer(
        cost_tp=args.cost_tp,
        cost_fp=args.cost_fp,
        cost_fn=args.cost_fn,
        cost_tn=args.cost_tn
    )
    
    # 运行分析
    print("🚀 开始决策曲线分析...")
    results = analyzer.analyze_multiple_models(model_results)
    
    # 生成阈值推荐
    recommendations = analyzer.generate_threshold_recommendations(results)
    
    # 生成输出
    analyzer.plot_decision_curve(results, 
                               os.path.join(args.output_dir, 'decision_curve.png'))
    
    # 成本-效果分析
    cost_scenarios = [
        {'cost_tp': 0.0, 'cost_fp': 0.1, 'cost_fn': 1.0, 'cost_tn': 0.0},  # 低成本
        {'cost_tp': 0.0, 'cost_fp': 0.3, 'cost_fn': 1.0, 'cost_tn': 0.0},  # 中等成本
        {'cost_tp': 0.0, 'cost_fp': 0.5, 'cost_fn': 1.0, 'cost_tn': 0.0},  # 高成本
        {'cost_tp': 0.0, 'cost_fp': 0.8, 'cost_fn': 1.0, 'cost_tn': 0.0}   # 很高成本
    ]
    
    analyzer.plot_cost_effectiveness(results, cost_scenarios,
                                   os.path.join(args.output_dir, 'cost_effectiveness.png'))
    
    analyzer.save_results(results, recommendations,
                        os.path.join(args.output_dir, 'dca_results.json'))
    
    analyzer.generate_summary_table(results, recommendations,
                                  os.path.join(args.output_dir, 'dca_summary.csv'))
    
    print("✅ 决策曲线分析完成!")


if __name__ == '__main__':
    main()
