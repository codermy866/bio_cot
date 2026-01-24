#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型对比训练脚本
分别训练ResNet50基线和Bio-COT模型，并生成对比结果
"""

import subprocess
import sys
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def train_resnet50_baseline():
    """训练ResNet50基线模型"""
    print("=" * 80)
    print("开始训练 ResNet50 基线模型")
    print("=" * 80)
    
    script_path = Path(__file__).parent / 'train_xiangyang.py'
    cmd = [
        sys.executable,
        str(script_path)
    ]
    
    env = {'USE_BIDA_WEIGHTS': 'false', 'CUDA_VISIBLE_DEVICES': '1'}
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ ResNet50基线训练完成")
    else:
        print(f"❌ ResNet50基线训练失败: {result.stderr}")
    
    return result.returncode == 0


def train_bio_cot():
    """训练Bio-COT模型"""
    print("=" * 80)
    print("开始训练 Bio-COT 模型（单模态OCT）")
    print("=" * 80)
    
    script_path = Path(__file__).parent / 'train_bio_cot_xiangyang.py'
    cmd = [
        sys.executable,
        str(script_path)
    ]
    
    env = {'CUDA_VISIBLE_DEVICES': '1'}
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Bio-COT训练完成")
    else:
        print(f"❌ Bio-COT训练失败: {result.stderr}")
    
    return result.returncode == 0


def compare_results():
    """对比两个模型的结果"""
    results_dir = Path(__file__).parent / 'results'
    
    # 查找最新的结果文件
    resnet_results = None
    bio_cot_results = None
    
    # 查找ResNet50结果（不包含bio_cot的文件名）
    for result_file in sorted(results_dir.glob('results_*.json'), reverse=True):
        if 'bio_cot' not in result_file.name:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'model' not in data or data.get('model') != 'Bio-COT (单模态OCT)':
                    resnet_results = (result_file, data)
                    break
    
    # 查找Bio-COT结果
    for result_file in sorted(results_dir.glob('results_bio_cot_*.json'), reverse=True):
        with open(result_file, 'r', encoding='utf-8') as f:
            bio_cot_results = (result_file, json.load(f))
            break
    
    if resnet_results is None or bio_cot_results is None:
        print("⚠️ 未找到完整的结果文件")
        if resnet_results:
            print(f"✅ 找到ResNet50结果: {resnet_results[0].name}")
        if bio_cot_results:
            print(f"✅ 找到Bio-COT结果: {bio_cot_results[0].name}")
        return
    
    print("=" * 80)
    print("模型对比分析：ResNet50 vs Bio-COT")
    print("=" * 80)
    print()
    
    resnet_data = resnet_results[1]
    bio_cot_data = bio_cot_results[1]
    
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
        'Bio-COT': [
            f"{bio_cot_data.get('best_val_acc', 0):.4f}",
            f"{bio_cot_data.get('final_metrics', {}).get('acc', 0):.4f}",
            f"{bio_cot_data.get('best_val_auc', 0):.4f}",
            f"{bio_cot_data.get('final_metrics', {}).get('auc', 0):.4f}",
            f"{bio_cot_data.get('classification_report', {}).get('阴性', {}).get('precision', 0):.4f}",
            f"{bio_cot_data.get('classification_report', {}).get('阴性', {}).get('recall', 0):.4f}",
            f"{bio_cot_data.get('classification_report', {}).get('阴性', {}).get('f1-score', 0):.4f}",
            f"{bio_cot_data.get('classification_report', {}).get('阳性', {}).get('precision', 0):.4f}",
            f"{bio_cot_data.get('classification_report', {}).get('阳性', {}).get('recall', 0):.4f}",
            f"{bio_cot_data.get('classification_report', {}).get('阳性', {}).get('f1-score', 0):.4f}",
            f"{bio_cot_data.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0):.4f}",
            f"{bio_cot_data.get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0):.4f}",
        ]
    }
    
    df = pd.DataFrame(comparison)
    print(df.to_string(index=False))
    print()
    
    # 计算改进幅度
    print("=" * 80)
    print("性能改进分析")
    print("=" * 80)
    
    metrics = [
        ('best_val_acc', '最佳验证准确率'),
        ('best_val_auc', '最佳验证AUC'),
        ('final_metrics.acc', '最终验证准确率'),
        ('final_metrics.auc', '最终验证AUC'),
    ]
    
    improvements = []
    for metric_key, metric_name in metrics:
        if '.' in metric_key:
            keys = metric_key.split('.')
            bio_val = bio_cot_data
            resnet_val = resnet_data
            for k in keys:
                bio_val = bio_val.get(k, {})
                resnet_val = resnet_val.get(k, {})
        else:
            bio_val = bio_cot_data.get(metric_key, 0)
            resnet_val = resnet_data.get(metric_key, 0)
        
        if isinstance(bio_val, dict) or isinstance(resnet_val, dict):
            continue
            
        improvement = ((bio_val - resnet_val) / resnet_val * 100) if resnet_val > 0 else 0
        improvements.append({
            'metric': metric_name,
            'resnet': resnet_val,
            'bio_cot': bio_val,
            'improvement': improvement
        })
        print(f"{metric_name}:")
        print(f"  ResNet50: {resnet_val:.4f}")
        print(f"  Bio-COT:  {bio_val:.4f}")
        print(f"  改进:     {improvement:+.2f}%")
        print()
    
    # 绘制对比图
    plot_comparison(resnet_data, bio_cot_data, results_dir)
    
    print("✅ 对比分析完成！")
    print(f"📊 对比图表已保存到: {results_dir / 'model_comparison.png'}")


def plot_comparison(resnet_data, bio_cot_data, output_dir):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 准确率对比
    ax = axes[0, 0]
    models = ['ResNet50\n基线', 'Bio-COT']
    best_acc = [
        resnet_data.get('best_val_acc', 0),
        bio_cot_data.get('best_val_acc', 0)
    ]
    final_acc = [
        resnet_data.get('final_metrics', {}).get('acc', 0),
        bio_cot_data.get('final_metrics', {}).get('acc', 0)
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
        bio_cot_data.get('best_val_auc', 0)
    ]
    final_auc = [
        resnet_data.get('final_metrics', {}).get('auc', 0),
        bio_cot_data.get('final_metrics', {}).get('auc', 0)
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
    bio_cot_f1 = [
        bio_cot_data.get('classification_report', {}).get('阴性', {}).get('f1-score', 0),
        bio_cot_data.get('classification_report', {}).get('阳性', {}).get('f1-score', 0)
    ]
    x_cat = range(len(categories))
    ax.bar([i - width/2 for i in x_cat], resnet_f1, width, label='ResNet50', alpha=0.8)
    ax.bar([i + width/2 for i in x_cat], bio_cot_f1, width, label='Bio-COT', alpha=0.8)
    ax.set_ylabel('F1分数')
    ax.set_title('各类别F1分数对比')
    ax.set_xticks(x_cat)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 综合指标对比
    ax = axes[1, 1]
    metrics = ['准确率', 'AUC', '阴性F1', '阳性F1', 'Macro F1']
    resnet_scores = [
        resnet_data.get('final_metrics', {}).get('acc', 0),
        resnet_data.get('final_metrics', {}).get('auc', 0),
        resnet_data.get('classification_report', {}).get('阴性', {}).get('f1-score', 0),
        resnet_data.get('classification_report', {}).get('阳性', {}).get('f1-score', 0),
        resnet_data.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0)
    ]
    bio_cot_scores = [
        bio_cot_data.get('final_metrics', {}).get('acc', 0),
        bio_cot_data.get('final_metrics', {}).get('auc', 0),
        bio_cot_data.get('classification_report', {}).get('阴性', {}).get('f1-score', 0),
        bio_cot_data.get('classification_report', {}).get('阳性', {}).get('f1-score', 0),
        bio_cot_data.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0)
    ]
    x_met = range(len(metrics))
    ax.bar([i - width/2 for i in x_met], resnet_scores, width, label='ResNet50', alpha=0.8)
    ax.bar([i + width/2 for i in x_met], bio_cot_scores, width, label='Bio-COT', alpha=0.8)
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


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='模型对比训练')
    parser.add_argument('--skip-resnet', action='store_true', help='跳过ResNet50训练')
    parser.add_argument('--skip-bio-cot', action='store_true', help='跳过Bio-COT训练')
    parser.add_argument('--only-compare', action='store_true', help='只进行对比，不训练')
    args = parser.parse_args()
    
    if not args.only_compare:
        # 训练ResNet50基线
        if not args.skip_resnet:
            train_resnet50_baseline()
        
        # 训练Bio-COT
        if not args.skip_bio_cot:
            train_bio_cot()
    
    # 对比结果
    compare_results()


if __name__ == '__main__':
    main()


