#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文配图模板和表格导出功能
用于生成柳叶刀级别的论文配图和表格

特点：
- 标准化的图表样式
- 多语言支持（英文为主）
- 高分辨率输出
- 表格格式化
- 统计图表
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib样式
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class PaperFigureGenerator:
    """论文配图生成器"""
    
    def __init__(self, output_dir: str = 'paper_figures'):
        """
        初始化论文配图生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义颜色方案
        self.colors = {
            'primary': '#2E86AB',      # 主色调
            'secondary': '#A23B72',    # 次色调
            'accent': '#F18F01',       # 强调色
            'success': '#C73E1D',      # 成功色
            'warning': '#F4A261',      # 警告色
            'info': '#264653',         # 信息色
            'light': '#E9C46A',       # 浅色
            'dark': '#264653'          # 深色
        }
        
        # 定义线型
        self.line_styles = ['-', '--', '-.', ':']
        
        # 定义标记
        self.markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
    
    def create_roc_curve(self, results: Dict[str, Any], save_path: str = 'roc_curve.png'):
        """创建ROC曲线图"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制ROC曲线
        for i, (model_name, result) in enumerate(results.items()):
            fpr = result['fpr']
            tpr = result['tpr']
            auc = result['auc']
            
            color = list(self.colors.values())[i % len(self.colors)]
            linestyle = self.line_styles[i % len(self.line_styles)]
            marker = self.markers[i % len(self.markers)]
            
            ax.plot(fpr, tpr, color=color, linestyle=linestyle, marker=marker,
                   linewidth=2, markersize=6, label=f'{model_name} (AUC = {auc:.3f})')
        
        # 绘制对角线
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        # 设置图形属性
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # 添加统计信息
        ax.text(0.6, 0.2, f'Total Samples: {results[list(results.keys())[0]]["n_samples"]}', 
               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path))
        plt.close()
        
        print(f"✅ ROC曲线图已保存到: {save_path}")
    
    def create_precision_recall_curve(self, results: Dict[str, Any], 
                                    save_path: str = 'precision_recall_curve.png'):
        """创建精确率-召回率曲线图"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制PR曲线
        for i, (model_name, result) in enumerate(results.items()):
            precision = result['precision']
            recall = result['recall']
            ap = result['average_precision']
            
            color = list(self.colors.values())[i % len(self.colors)]
            linestyle = self.line_styles[i % len(self.line_styles)]
            marker = self.markers[i % len(self.markers)]
            
            ax.plot(recall, precision, color=color, linestyle=linestyle, marker=marker,
                   linewidth=2, markersize=6, label=f'{model_name} (AP = {ap:.3f})')
        
        # 设置图形属性
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title('Precision-Recall Curves', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path))
        plt.close()
        
        print(f"✅ 精确率-召回率曲线图已保存到: {save_path}")
    
    def create_confusion_matrix(self, cm: np.ndarray, labels: List[str] = None, 
                               save_path: str = 'confusion_matrix.png'):
        """创建混淆矩阵图"""
        if labels is None:
            labels = ['Negative', 'Positive']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # 绘制热力图
        im = ax.imshow(cm_percent, interpolation='nearest', cmap='Blues')
        
        # 添加数值标签
        thresh = cm_percent.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)',
                       ha="center", va="center", color="white" if cm_percent[i, j] > thresh else "black",
                       fontsize=14, fontweight='bold')
        
        # 设置图形属性
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_yticklabels(labels, fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Percentage (%)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path))
        plt.close()
        
        print(f"✅ 混淆矩阵图已保存到: {save_path}")
    
    def create_performance_comparison(self, results: Dict[str, Any], 
                                   save_path: str = 'performance_comparison.png'):
        """创建性能对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 准备数据
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        # 1. 条形图对比
        ax1 = axes[0, 0]
        x = np.arange(len(models))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            ax1.bar(x + i*width, values, width, label=metric, alpha=0.8)
        
        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 雷达图
        ax2 = axes[0, 1]
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        for model in models:
            values = [results[model][metric] for metric in metrics]
            values += values[:1]  # 闭合
            ax2.plot(angles, values, 'o-', linewidth=2, label=model)
            ax2.fill(angles, values, alpha=0.25)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics)
        ax2.set_ylim(0, 1)
        ax2.set_title('Performance Radar Chart', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 热力图
        ax3 = axes[1, 0]
        data = np.array([[results[model][metric] for metric in metrics] for model in models])
        im = ax3.imshow(data, cmap='YlOrRd', aspect='auto')
        
        ax3.set_xticks(range(len(metrics)))
        ax3.set_yticks(range(len(models)))
        ax3.set_xticklabels(metrics, rotation=45)
        ax3.set_yticklabels(models)
        ax3.set_title('Performance Heatmap', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for i in range(len(models)):
            for j in range(len(metrics)):
                ax3.text(j, i, f'{data[i, j]:.3f}', ha="center", va="center", 
                        color="white" if data[i, j] < 0.5 else "black", fontweight='bold')
        
        # 4. 箱线图
        ax4 = axes[1, 1]
        data_for_box = []
        labels_for_box = []
        
        for model in models:
            model_scores = [results[model][metric] for metric in metrics]
            data_for_box.append(model_scores)
            labels_for_box.append(model)
        
        bp = ax4.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        
        # 设置箱线图颜色
        for patch, color in zip(bp['boxes'], list(self.colors.values())[:len(models)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_title('Performance Distribution', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path))
        plt.close()
        
        print(f"✅ 性能对比图已保存到: {save_path}")
    
    def create_learning_curves(self, history: Dict[str, List[float]], 
                              save_path: str = 'learning_curves.png'):
        """创建学习曲线图"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 训练和验证损失
        ax1 = axes[0]
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 训练和验证准确率
        ax2 = axes[1]
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path))
        plt.close()
        
        print(f"✅ 学习曲线图已保存到: {save_path}")
    
    def create_statistical_plots(self, data: Dict[str, Any], 
                                save_path: str = 'statistical_plots.png'):
        """创建统计图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 分布图
        ax1 = axes[0, 0]
        for i, (name, values) in enumerate(data['distributions'].items()):
            ax1.hist(values, bins=30, alpha=0.7, label=name, 
                    color=list(self.colors.values())[i % len(self.colors)])
        ax1.set_xlabel('Value', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 散点图
        ax2 = axes[0, 1]
        x = data['scatter']['x']
        y = data['scatter']['y']
        ax2.scatter(x, y, alpha=0.6, color=self.colors['primary'])
        
        # 添加趋势线
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax2.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('X', fontsize=12)
        ax2.set_ylabel('Y', fontsize=12)
        ax2.set_title('Scatter Plot with Trend Line', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 箱线图
        ax3 = axes[1, 0]
        box_data = [data['boxplot'][key] for key in data['boxplot'].keys()]
        bp = ax3.boxplot(box_data, labels=list(data['boxplot'].keys()), patch_artist=True)
        
        for patch, color in zip(bp['boxes'], list(self.colors.values())[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_ylabel('Value', fontsize=12)
        ax3.set_title('Box Plot Comparison', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 相关性热力图
        ax4 = axes[1, 1]
        corr_matrix = data['correlation']
        im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        ax4.set_xticks(range(len(corr_matrix.columns)))
        ax4.set_yticks(range(len(corr_matrix.index)))
        ax4.set_xticklabels(corr_matrix.columns, rotation=45)
        ax4.set_yticklabels(corr_matrix.index)
        ax4.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha="center", va="center",
                        color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_path))
        plt.close()
        
        print(f"✅ 统计图表已保存到: {save_path}")


class PaperTableGenerator:
    """论文表格生成器"""
    
    def __init__(self, output_dir: str = 'paper_tables'):
        """
        初始化论文表格生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_performance_table(self, results: Dict[str, Any], 
                               save_path: str = 'performance_table.csv'):
        """创建性能表格"""
        # 准备数据
        table_data = []
        
        for model_name, metrics in results.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1_score']:.3f}",
                'AUC': f"{metrics['auc']:.3f}",
                'Sensitivity': f"{metrics['recall']:.3f}",
                'Specificity': f"{metrics['specificity']:.3f}",
                'NPV': f"{metrics['npv']:.3f}",
                'PPV': f"{metrics['ppv']:.3f}"
            }
            table_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(table_data)
        
        # 保存为CSV
        df.to_csv(os.path.join(self.output_dir, save_path), index=False)
        
        # 保存为LaTeX格式
        latex_path = save_path.replace('.csv', '.tex')
        with open(os.path.join(self.output_dir, latex_path), 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        
        print(f"✅ 性能表格已保存到: {save_path}")
        print(f"✅ LaTeX格式已保存到: {latex_path}")
        
        return df
    
    def create_demographic_table(self, data: Dict[str, Any], 
                               save_path: str = 'demographic_table.csv'):
        """创建人口统计学表格"""
        # 准备数据
        table_data = []
        
        for group, stats in data.items():
            row = {
                'Group': group,
                'N': stats['n'],
                'Age (Mean ± SD)': f"{stats['age_mean']:.1f} ± {stats['age_std']:.1f}",
                'Age (Range)': f"{stats['age_min']:.0f}-{stats['age_max']:.0f}",
                'Male (%)': f"{stats['male_percent']:.1f}",
                'Female (%)': f"{stats['female_percent']:.1f}",
                'Positive (%)': f"{stats['positive_percent']:.1f}",
                'Negative (%)': f"{stats['negative_percent']:.1f}"
            }
            table_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(table_data)
        
        # 保存
        df.to_csv(os.path.join(self.output_dir, save_path), index=False)
        
        # LaTeX格式
        latex_path = save_path.replace('.csv', '.tex')
        with open(os.path.join(self.output_dir, latex_path), 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        
        print(f"✅ 人口统计学表格已保存到: {save_path}")
        
        return df
    
    def create_comparison_table(self, results: Dict[str, Any], 
                              save_path: str = 'comparison_table.csv'):
        """创建对比表格"""
        # 准备数据
        table_data = []
        
        for model_name, metrics in results.items():
            row = {
                'Model': model_name,
                'Parameters (M)': f"{metrics['parameters']:.1f}",
                'Training Time (h)': f"{metrics['training_time']:.1f}",
                'Inference Time (ms)': f"{metrics['inference_time']:.1f}",
                'Memory Usage (GB)': f"{metrics['memory_usage']:.1f}",
                'AUC': f"{metrics['auc']:.3f}",
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'F1-Score': f"{metrics['f1_score']:.3f}"
            }
            table_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(table_data)
        
        # 保存
        df.to_csv(os.path.join(self.output_dir, save_path), index=False)
        
        # LaTeX格式
        latex_path = save_path.replace('.csv', '.tex')
        with open(os.path.join(self.output_dir, latex_path), 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        
        print(f"✅ 对比表格已保存到: {save_path}")
        
        return df
    
    def create_statistical_test_table(self, test_results: Dict[str, Any], 
                                    save_path: str = 'statistical_test_table.csv'):
        """创建统计检验表格"""
        # 准备数据
        table_data = []
        
        for test_name, result in test_results.items():
            row = {
                'Test': test_name,
                'Statistic': f"{result['statistic']:.3f}",
                'P-value': f"{result['p_value']:.3f}",
                'Significant': 'Yes' if result['p_value'] < 0.05 else 'No',
                'Effect Size': f"{result['effect_size']:.3f}",
                'Interpretation': result['interpretation']
            }
            table_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(table_data)
        
        # 保存
        df.to_csv(os.path.join(self.output_dir, save_path), index=False)
        
        # LaTeX格式
        latex_path = save_path.replace('.csv', '.tex')
        with open(os.path.join(self.output_dir, latex_path), 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        
        print(f"✅ 统计检验表格已保存到: {save_path}")
        
        return df


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='论文配图和表格生成')
    parser.add_argument('--results_file', type=str, required=True, help='结果文件路径')
    parser.add_argument('--output_dir', type=str, default='paper_output', help='输出目录')
    parser.add_argument('--figure_format', type=str, default='png', choices=['png', 'pdf', 'svg'], 
                       help='图片格式')
    parser.add_argument('--table_format', type=str, default='csv', choices=['csv', 'xlsx'], 
                       help='表格格式')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载结果数据
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    # 创建配图生成器
    figure_generator = PaperFigureGenerator(os.path.join(args.output_dir, 'figures'))
    
    # 创建表格生成器
    table_generator = PaperTableGenerator(os.path.join(args.output_dir, 'tables'))
    
    # 生成配图
    print("🔄 生成论文配图...")
    
    # ROC曲线
    if 'roc_results' in results:
        figure_generator.create_roc_curve(results['roc_results'])
    
    # 精确率-召回率曲线
    if 'pr_results' in results:
        figure_generator.create_precision_recall_curve(results['pr_results'])
    
    # 混淆矩阵
    if 'confusion_matrix' in results:
        figure_generator.create_confusion_matrix(results['confusion_matrix'])
    
    # 性能对比
    if 'performance_results' in results:
        figure_generator.create_performance_comparison(results['performance_results'])
    
    # 学习曲线
    if 'training_history' in results:
        figure_generator.create_learning_curves(results['training_history'])
    
    # 统计图表
    if 'statistical_data' in results:
        figure_generator.create_statistical_plots(results['statistical_data'])
    
    # 生成表格
    print("🔄 生成论文表格...")
    
    # 性能表格
    if 'performance_results' in results:
        table_generator.create_performance_table(results['performance_results'])
    
    # 人口统计学表格
    if 'demographic_data' in results:
        table_generator.create_demographic_table(results['demographic_data'])
    
    # 对比表格
    if 'comparison_results' in results:
        table_generator.create_comparison_table(results['comparison_results'])
    
    # 统计检验表格
    if 'statistical_tests' in results:
        table_generator.create_statistical_test_table(results['statistical_tests'])
    
    print("✅ 论文配图和表格生成完成!")


if __name__ == '__main__':
    main()
