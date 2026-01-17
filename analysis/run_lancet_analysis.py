#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lancet期刊发表所需实验 - 综合分析脚本
一次性运行所有关键分析：Bootstrap CI, DCA, 亚组分析
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.bootstrap_confidence_intervals import BootstrapCI
from analysis.decision_curve_analysis import DecisionCurveAnalyzer
from analysis.subgroup_analysis import SubgroupAnalyzer


def load_predictions_from_model(result_dir: str, model_type: str = 'swint'):
    """
    从模型结果目录加载预测结果
    注意：这需要模型重新预测，或者从保存的预测结果加载
    """
    print(f"⚠️  需要从模型重新预测或加载保存的预测结果")
    print(f"   结果目录: {result_dir}")
    print(f"   模型类型: {model_type}")
    
    # TODO: 实现实际的预测加载
    # 选项1: 从保存的npy文件加载
    val_labels_file = os.path.join(result_dir, 'val_labels.npy')
    val_probs_file = os.path.join(result_dir, 'val_probs.npy')
    val_preds_file = os.path.join(result_dir, 'val_preds.npy')
    val_metadata_file = os.path.join(result_dir, 'val_metadata.csv')
    
    if all(os.path.exists(f) for f in [val_labels_file, val_probs_file, val_preds_file]):
        print("✅ 从保存的文件加载预测结果")
        y_true = np.load(val_labels_file)
        y_probs = np.load(val_probs_file)
        y_pred = np.load(val_preds_file)
        
        if os.path.exists(val_metadata_file):
            metadata = pd.read_csv(val_metadata_file)
        else:
            metadata = None
        
        return y_true, y_pred, y_probs, metadata
    else:
        print("❌ 未找到保存的预测结果文件")
        print("   需要运行模型重新预测或修改训练脚本保存预测结果")
        return None, None, None, None


def run_comprehensive_analysis(result_dir: str, data_path: str, 
                              output_dir: str, model_name: str = "SwinT"):
    """
    运行综合分析
    
    Args:
        result_dir: 模型结果目录
        data_path: 数据路径
        output_dir: 输出目录
        model_name: 模型名称
    """
    print(f"\n{'='*80}")
    print(f"🚀 开始综合分析: {model_name}")
    print(f"{'='*80}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载预测结果
    print("📥 步骤1: 加载预测结果...")
    y_true, y_pred, y_probs, metadata = load_predictions_from_model(result_dir, model_name.lower())
    
    if y_true is None:
        print("\n❌ 无法加载预测结果")
        print("   请先运行以下操作之一:")
        print("   1. 修改训练脚本保存预测结果（推荐）")
        print("   2. 运行模型重新预测并保存结果")
        return False
    
    print(f"   ✅ 加载成功: {len(y_true)} 个样本")
    
    # 2. Bootstrap置信区间
    print(f"\n📊 步骤2: 计算Bootstrap置信区间...")
    bootstrap_ci = BootstrapCI(n_bootstrap=2000)
    ci_results = bootstrap_ci.calculate_all_metrics_ci(y_true, y_pred, y_probs)
    
    # 保存CI结果
    ci_output_file = os.path.join(output_dir, 'bootstrap_ci_results.json')
    with open(ci_output_file, 'w') as f:
        json.dump(ci_results, f, indent=2)
    
    print(f"   ✅ CI结果已保存: {ci_output_file}")
    print(f"   AUC: {ci_results['auc']['formatted']}")
    print(f"   Accuracy: {ci_results['accuracy']['formatted']}")
    print(f"   Sensitivity: {ci_results['sensitivity']['formatted']}")
    print(f"   Specificity: {ci_results['specificity']['formatted']}")
    
    # 3. 决策曲线分析
    print(f"\n📈 步骤3: 决策曲线分析 (DCA)...")
    dca_analyzer = DecisionCurveAnalyzer()
    dca_results = dca_analyzer.analyze(y_true, y_probs, model_name)
    
    # 找到最优阈值
    optimal_threshold, max_net_benefit = dca_analyzer.find_optimal_threshold(y_true, y_probs)
    
    # 保存DCA结果
    dca_output_file = os.path.join(output_dir, 'dca_results.csv')
    dca_results.to_csv(dca_output_file, index=False)
    
    # 绘制DCA图
    dca_plot_file = os.path.join(output_dir, 'decision_curve.png')
    dca_analyzer.plot_decision_curve(dca_results, dca_plot_file, 
                                     f"Decision Curve Analysis - {model_name}")
    
    print(f"   ✅ DCA结果已保存: {dca_output_file}")
    print(f"   ✅ DCA图已保存: {dca_plot_file}")
    print(f"   最优阈值: {optimal_threshold:.4f}")
    print(f"   最大净收益: {max_net_benefit:.4f}")
    
    # 4. 亚组分析（如果有元数据）
    if metadata is not None and len(metadata) == len(y_true):
        print(f"\n👥 步骤4: 亚组分析...")
        subgroup_analyzer = SubgroupAnalyzer()
        subgroup_results = subgroup_analyzer.analyze_all_subgroups(
            y_true, y_pred, y_probs, metadata
        )
        
        # 保存亚组结果
        subgroup_output_file = os.path.join(output_dir, 'subgroup_analysis.csv')
        subgroup_results.to_csv(subgroup_output_file, index=False)
        
        # 绘制森林图
        for metric in ['auc', 'sensitivity', 'specificity']:
            if metric in subgroup_results.columns:
                plot_file = os.path.join(output_dir, f'forest_plot_{metric}.png')
                subgroup_analyzer.plot_forest_plot(subgroup_results, metric, plot_file)
        
        print(f"   ✅ 亚组分析结果已保存: {subgroup_output_file}")
        print(f"   分析了 {len(subgroup_results)} 个亚组")
    else:
        print(f"\n⚠️  步骤4: 跳过亚组分析（缺少元数据）")
        print(f"   需要元数据文件: val_metadata.csv")
    
    # 5. 生成汇总报告
    print(f"\n📝 步骤5: 生成汇总报告...")
    generate_summary_report(ci_results, dca_results, optimal_threshold, 
                          max_net_benefit, output_dir, model_name)
    
    print(f"\n{'='*80}")
    print(f"✅ 综合分析完成！")
    print(f"   结果保存在: {output_dir}")
    print(f"{'='*80}\n")
    
    return True


def generate_summary_report(ci_results: dict, dca_results: pd.DataFrame,
                           optimal_threshold: float, max_net_benefit: float,
                           output_dir: str, model_name: str):
    """生成汇总报告"""
    
    report = f"""
# {model_name} 综合分析报告
## Lancet期刊发表所需分析结果

### 1. 模型性能指标 (95% 置信区间)

| 指标 | 值 (95% CI) |
|------|------------|
| AUC | {ci_results['auc']['formatted']} |
| Accuracy | {ci_results['accuracy']['formatted']} |
| Sensitivity | {ci_results['sensitivity']['formatted']} |
| Specificity | {ci_results['specificity']['formatted']} |
| Precision (PPV) | {ci_results['precision']['formatted']} |
| NPV | {ci_results['npv']['formatted']} |
| F1-Score | {ci_results['f1_score']['formatted']} |

### 2. 决策曲线分析 (DCA)

- **最优决策阈值**: {optimal_threshold:.4f}
- **最大净收益**: {max_net_benefit:.4f}
- **DCA曲线**: 见 `decision_curve.png`

### 3. 亚组分析

见 `subgroup_analysis.csv` 和 `forest_plot_*.png`

---

**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    report_file = os.path.join(output_dir, 'summary_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   ✅ 汇总报告已保存: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Lancet期刊发表所需综合分析')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='模型结果目录（包含预测结果或需要重新预测）')
    parser.add_argument('--data_path', type=str, default='5centers_multi',
                       help='数据路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认：result_dir/analysis）')
    parser.add_argument('--model_name', type=str, default='SwinT',
                       help='模型名称')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.result_dir, 'lancet_analysis')
    
    # 运行分析
    success = run_comprehensive_analysis(
        args.result_dir,
        args.data_path,
        args.output_dir,
        args.model_name
    )
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()

