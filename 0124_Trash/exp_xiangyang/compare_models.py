#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型对比分析脚本
对比ResNet50基线和Bio-COT（BIDA权重）模型在襄阳数据集上的表现
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results(result_file):
    """加载结果文件"""
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_models():
    """对比两个模型的结果"""
    results_dir = Path(__file__).parent / 'results'
    
    # 查找结果文件
    bida_results = None
    resnet_results = None
    
    # 查找BIDA结果（包含best_val_auc > 0.7的结果）
    for result_file in sorted(results_dir.glob('results_*.json')):
        data = load_results(result_file)
        if data.get('best_val_auc', 0) > 0.7:
            bida_results = (result_file, data)
            break
    
    # 查找ResNet50基线结果（最新的）
    for result_file in sorted(results_dir.glob('results_*.json'), reverse=True):
        data = load_results(result_file)
        # 如果AUC较低，可能是ResNet基线
        if data.get('best_val_auc', 0) < 0.75 or bida_results is None:
            if result_file != bida_results[0] if bida_results else True:
                resnet_results = (result_file, data)
                break
    
    if bida_results is None:
        print("⚠️ 未找到Bio-COT (BIDA)结果")
        return
    
    if resnet_results is None:
        print("⚠️ 未找到ResNet50基线结果")
        print("📊 Bio-COT (BIDA) 结果:")
        print_results(bida_results[1])
        return
    
    print("=" * 80)
    print("模型对比分析：ResNet50 vs Bio-COT (BIDA)")
    print("=" * 80)
    print()
    
    # 提取关键指标
    bida_data = bida_results[1]
    resnet_data = resnet_results[1]
    
    # 创建对比表格
    comparison = {
        '指标': ['最佳验证准确率', '最终验证准确率', '最佳验证AUC', '最终验证AUC', 
                '阴性Precision', '阴性Recall', '阴性F1', 
                '阳性Precision', '阳性Recall', '阳性F1',
                'Macro Avg F1', 'Weighted Avg F1'],
        'ResNet50基线': [
            f"{resnet_data.get('best_val_acc', 0):.4f}",
            f"{resnet_data.get('final_metrics', {}).get('acc', 0):.4f}",
            f"{resnet_data.get('best_val_auc', 0):.4f}",
            f"{resnet_data.get('final_metrics', {}).get('auc', 0):.4f}",
            f"{resnet_data.get('classification_report', {}).get('阴性', {}).get('precision', 0):.4f}",
            f"{resnet_data.get('classification_report', {}).get('阴性', {}).get('recall', 0):.4f}",
            f"{resnet_data.get('classification_report', {}).get('阴性', {}).get('f1-score', 0):.4f}",
            f"{resnet_data.get('classification_report', {}).get('阳性', {}).get('precision', 0):.4f}",
            f"{resnet_data.get('classification_report', {}).get('阳性', {}).get('recall', 0):.4f}",
            f"{resnet_data.get('classification_report', {}).get('阳性', {}).get('f1-score', 0):.4f}",
            f"{resnet_data.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0):.4f}",
            f"{resnet_data.get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0):.4f}",
        ],
        'Bio-COT (BIDA)': [
            f"{bida_data.get('best_val_acc', 0):.4f}",
            f"{bida_data.get('final_metrics', {}).get('acc', 0):.4f}",
            f"{bida_data.get('best_val_auc', 0):.4f}",
            f"{bida_data.get('final_metrics', {}).get('auc', 0):.4f}",
            f"{bida_data.get('classification_report', {}).get('阴性', {}).get('precision', 0):.4f}",
            f"{bida_data.get('classification_report', {}).get('阴性', {}).get('recall', 0):.4f}",
            f"{bida_data.get('classification_report', {}).get('阴性', {}).get('f1-score', 0):.4f}",
            f"{bida_data.get('classification_report', {}).get('阳性', {}).get('precision', 0):.4f}",
            f"{bida_data.get('classification_report', {}).get('阳性', {}).get('recall', 0):.4f}",
            f"{bida_data.get('classification_report', {}).get('阳性', {}).get('f1-score', 0):.4f}",
            f"{bida_data.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0):.4f}",
            f"{bida_data.get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0):.4f}",
        ]
    }
    
    df = pd.DataFrame(comparison)
    print(df.to_string(index=False))
    print()
    
    # 计算改进幅度
    print("=" * 80)
    print("性能改进分析")
    print("=" * 80)
    
    metrics_to_compare = [
        ('best_val_acc', '最佳验证准确率'),
        ('best_val_auc', '最佳验证AUC'),
        ('final_metrics.acc', '最终验证准确率'),
        ('final_metrics.auc', '最终验证AUC'),
    ]
    
    for metric_key, metric_name in metrics_to_compare:
        if '.' in metric_key:
            keys = metric_key.split('.')
            bida_val = bida_data
            resnet_val = resnet_data
            for k in keys:
                bida_val = bida_val.get(k, {})
                resnet_val = resnet_val.get(k, {})
        else:
            bida_val = bida_data.get(metric_key, 0)
            resnet_val = resnet_data.get(metric_key, 0)
        
        if isinstance(bida_val, dict) or isinstance(resnet_val, dict):
            continue
            
        improvement = ((bida_val - resnet_val) / resnet_val * 100) if resnet_val > 0 else 0
        print(f"{metric_name}:")
        print(f"  ResNet50: {resnet_val:.4f}")
        print(f"  Bio-COT:  {bida_val:.4f}")
        print(f"  改进:     {improvement:+.2f}%")
        print()
    
    # 绘制对比图
    plot_comparison(bida_data, resnet_data, results_dir)
    
    print("✅ 对比分析完成！")
    print(f"📊 对比图表已保存到: {results_dir / 'model_comparison.png'}")

def print_results(data):
    """打印结果"""
    print(f"最佳验证准确率: {data.get('best_val_acc', 0):.4f}")
    print(f"最佳验证AUC: {data.get('best_val_auc', 0):.4f}")
    print(f"最终验证准确率: {data.get('final_metrics', {}).get('acc', 0):.4f}")
    print(f"最终验证AUC: {data.get('final_metrics', {}).get('auc', 0):.4f}")

def plot_comparison(bida_data, resnet_data, output_dir):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 准确率对比
    ax = axes[0, 0]
    models = ['ResNet50\n基线', 'Bio-COT\n(BIDA)']
    best_acc = [
        resnet_data.get('best_val_acc', 0),
        bida_data.get('best_val_acc', 0)
    ]
    final_acc = [
        resnet_data.get('final_metrics', {}).get('acc', 0),
        bida_data.get('final_metrics', {}).get('acc', 0)
    ]
    x = range(len(models))
    width = 0.35
    ax.bar([i - width/2 for i in x], best_acc, width, label='最佳准确率', alpha=0.8)
    ax.bar([i + width/2 for i in x], final_acc, width, label='最终准确率', alpha=0.8)
    ax.set_ylabel('准确率')
    ax.set_title('验证集准确率对比')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. AUC对比
    ax = axes[0, 1]
    best_auc = [
        resnet_data.get('best_val_auc', 0),
        bida_data.get('best_val_auc', 0)
    ]
    final_auc = [
        resnet_data.get('final_metrics', {}).get('auc', 0),
        bida_data.get('final_metrics', {}).get('auc', 0)
    ]
    ax.bar([i - width/2 for i in x], best_auc, width, label='最佳AUC', alpha=0.8)
    ax.bar([i + width/2 for i in x], final_auc, width, label='最终AUC', alpha=0.8)
    ax.set_ylabel('AUC')
    ax.set_title('验证集AUC对比')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. F1分数对比（按类别）
    ax = axes[1, 0]
    categories = ['阴性', '阳性']
    resnet_f1 = [
        resnet_data.get('classification_report', {}).get('阴性', {}).get('f1-score', 0),
        resnet_data.get('classification_report', {}).get('阳性', {}).get('f1-score', 0)
    ]
    bida_f1 = [
        bida_data.get('classification_report', {}).get('阴性', {}).get('f1-score', 0),
        bida_data.get('classification_report', {}).get('阳性', {}).get('f1-score', 0)
    ]
    x_cat = range(len(categories))
    ax.bar([i - width/2 for i in x_cat], resnet_f1, width, label='ResNet50', alpha=0.8)
    ax.bar([i + width/2 for i in x_cat], bida_f1, width, label='Bio-COT', alpha=0.8)
    ax.set_ylabel('F1分数')
    ax.set_title('各类别F1分数对比')
    ax.set_xticks(x_cat)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 综合指标对比（雷达图风格）
    ax = axes[1, 1]
    metrics = ['准确率', 'AUC', '阴性F1', '阳性F1', 'Macro F1']
    resnet_scores = [
        resnet_data.get('final_metrics', {}).get('acc', 0),
        resnet_data.get('final_metrics', {}).get('auc', 0),
        resnet_data.get('classification_report', {}).get('阴性', {}).get('f1-score', 0),
        resnet_data.get('classification_report', {}).get('阳性', {}).get('f1-score', 0),
        resnet_data.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0)
    ]
    bida_scores = [
        bida_data.get('final_metrics', {}).get('acc', 0),
        bida_data.get('final_metrics', {}).get('auc', 0),
        bida_data.get('classification_report', {}).get('阴性', {}).get('f1-score', 0),
        bida_data.get('classification_report', {}).get('阳性', {}).get('f1-score', 0),
        bida_data.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0)
    ]
    x_met = range(len(metrics))
    ax.bar([i - width/2 for i in x_met], resnet_scores, width, label='ResNet50', alpha=0.8)
    ax.bar([i + width/2 for i in x_met], bida_scores, width, label='Bio-COT', alpha=0.8)
    ax.set_ylabel('分数')
    ax.set_title('综合指标对比')
    ax.set_xticks(x_met)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    compare_models()

