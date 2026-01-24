#!/usr/bin/env python3
"""
从训练日志中提取数据并生成可视化结果
"""
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def parse_training_log(log_file):
    """从日志文件中解析训练历史数据"""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取epoch信息
    epoch_pattern = r'Epoch (\d+)/\d+:\s*\n\s*Train - Loss: ([\d.]+), Acc: ([\d.]+)\s*\n\s*Val\s*-\s*Loss: ([\d.]+), Acc: ([\d.]+), AUC: ([\d.]+), F1: ([\d.]+)\s*\n\s*Loss Components - CLS: ([\d.]+), OT: ([\d.]+), Consist: ([\d.]+), Adv: ([\d.]+)'
    
    matches = re.findall(epoch_pattern, content)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_f1': [],
        'train_cls_loss': [],
        'train_ot_loss': [],
        'train_consist_loss': [],
        'train_adv_loss': []
    }
    
    for match in matches:
        epoch, train_loss, train_acc, val_loss, val_acc, val_auc, val_f1, cls_loss, ot_loss, consist_loss, adv_loss = match
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['val_auc'].append(float(val_auc))
        history['val_f1'].append(float(val_f1))
        history['train_cls_loss'].append(float(cls_loss))
        history['train_ot_loss'].append(float(ot_loss))
        history['train_consist_loss'].append(float(consist_loss))
        history['train_adv_loss'].append(float(adv_loss))
    
    # 提取最佳AUC
    best_auc_match = re.search(r'最佳AUC: ([\d.]+) \(Epoch (\d+)\)', content)
    if best_auc_match:
        best_auc = float(best_auc_match.group(1))
        best_epoch = int(best_auc_match.group(2))
    else:
        best_auc = max(history['val_auc']) if history['val_auc'] else 0
        best_epoch = history['val_auc'].index(best_auc) + 1 if history['val_auc'] else 0
    
    return history, best_auc, best_epoch


def load_results_json(json_file):
    """加载结果JSON文件"""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except:
        return None


def generate_visualizations_from_log(log_file, output_dir, timestamp):
    """从日志生成可视化"""
    print(f"📊 从日志文件解析数据: {log_file}")
    history, best_auc, best_epoch = parse_training_log(log_file)
    
    if not history['train_loss']:
        print("❌ 未能从日志中提取训练数据")
        return
    
    print(f"✅ 提取了 {len(history['train_loss'])} 个epoch的数据")
    print(f"   最佳AUC: {best_auc:.4f} (Epoch {best_epoch})")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 导入可视化函数
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from train_bio_cot_multimodal_balanced_xiangyang import (
            plot_training_curves,
            plot_loss_heatmap,
            plot_loss_boxplot,
            plot_metrics_comparison,
            plot_loss_component_analysis
        )
    except ImportError:
        # 如果新函数不存在，只导入已有的
        from train_bio_cot_multimodal_balanced_xiangyang import (
            plot_training_curves,
            plot_loss_heatmap,
            plot_loss_boxplot,
            plot_metrics_comparison
        )
        plot_loss_component_analysis = None
    
    # 生成可视化
    print("\n📊 生成训练曲线...")
    plot_training_curves(history, output_dir, timestamp)
    
    print("📊 生成损失热图...")
    plot_loss_heatmap(history, output_dir, timestamp)
    
    print("📊 生成损失箱线图...")
    plot_loss_boxplot(history, output_dir, timestamp)
    
    print("📊 生成指标对比图...")
    plot_metrics_comparison(history, output_dir, timestamp)
    
    if plot_loss_component_analysis:
        print("📊 生成损失组件详细分析...")
        plot_loss_component_analysis(history, output_dir, timestamp)
    else:
        print("⚠️  损失组件详细分析函数不可用，跳过")
    
    # 保存历史数据到JSON
    results = {
        'best_epoch': best_epoch,
        'best_auc': best_auc,
        'history': history
    }
    
    results_file = output_dir / f'results_bio_cot_multimodal_balanced_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ 结果已保存: {results_file}")
    
    print("\n✅ 所有可视化已生成完成！")


if __name__ == '__main__':
    log_file = '/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_xiangyang/logs/train_bio_cot_multimodal_balanced_20260107_093123.log'
    output_dir = '/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_xiangyang/results_multimodal_balanced'
    timestamp = '20260107_093123'
    
    generate_visualizations_from_log(log_file, output_dir, timestamp)

