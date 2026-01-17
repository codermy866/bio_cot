#!/usr/bin/env python3
"""
重新生成小提琴图（使用已保存的预测结果）
"""
import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train_bio_cot_multimodal_balanced_xiangyang import (
    plot_advanced_violin_analysis,
    plot_prediction_distribution
)

def regenerate_violin_plots():
    """重新生成小提琴图"""
    timestamp = '20260107_155330'
    results_file = Path(f'results_multimodal_balanced/results_bio_cot_multimodal_balanced_{timestamp}.json')
    output_dir = Path('results_multimodal_balanced')
    
    # 加载结果
    print(f"📂 加载结果文件: {results_file}")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # 检查是否有预测概率数据
    if 'final_val_probs' in results:
        y_probs = np.array(results['final_val_probs'])
        y_labels = np.array(results['final_val_labels'])
        print(f"✅ 找到预测概率数据: {len(y_probs)} 个样本")
    elif 'val_probs' in results:
        y_probs = np.array(results['val_probs'])
        y_labels = np.array(results['val_labels'])
        print(f"✅ 找到预测概率数据: {len(y_probs)} 个样本")
    else:
        print("❌ JSON文件中没有找到预测概率数据")
        print("   需要重新运行模型推理来生成预测概率")
        return
    
    print(f"   阴性样本: {np.sum(y_labels == 0)}, 阳性样本: {np.sum(y_labels == 1)}")
    
    # 重新生成预测分布图（包含小提琴图）
    print("\n📊 重新生成预测分布分析图（包含小提琴图）...")
    try:
        plot_prediction_distribution(y_labels, y_probs, output_dir, timestamp)
        print("✅ 预测分布分析图已重新生成")
    except Exception as e:
        print(f"❌ 生成预测分布图失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 重新生成高级小提琴图
    print("\n📊 重新生成高级小提琴图分析...")
    try:
        plot_advanced_violin_analysis(y_labels, y_probs, output_dir, timestamp)
        print("✅ 高级小提琴图分析已重新生成")
    except Exception as e:
        print(f"❌ 生成高级小提琴图失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 所有小提琴图已重新生成！")

if __name__ == '__main__':
    regenerate_violin_plots()

