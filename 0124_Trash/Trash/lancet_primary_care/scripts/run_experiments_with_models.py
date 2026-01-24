#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用真实模型运行The Lancet Primary Care的所有实验
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from lancet_primary_care.models.model_loader import (
    MultimodalModelLoader,
    create_data_loader,
    get_best_model_configs
)
from lancet_primary_care.experiments.experiment1_biopsy_reduction import BiopsyReductionExperiment
from lancet_primary_care.experiments.experiment2_oct_sensitivity import OCTSensitivityExperiment
from lancet_primary_care.experiments.experiment3_screening_comparison import ScreeningComparisonExperiment
from lancet_primary_care.experiments.experiment4_workflow_optimization import WorkflowOptimizationExperiment


def load_model_and_predict(data_path: str, split: str = 'test'):
    """
    加载模型并对数据集进行预测
    
    Returns:
        predictions: 预测标签
        probabilities: 预测概率
        labels: 真实标签
    """
    print("\n" + "="*60)
    print("加载模型和生成预测")
    print("="*60)
    
    # 初始化模型加载器
    loader = MultimodalModelLoader(device='cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu')
    
    # 获取模型配置
    configs = get_best_model_configs()
    
    if len(configs) == 0:
        raise ValueError("没有找到可用的模型文件！")
    
    print(f"\n找到 {len(configs)} 个模型配置")
    
    # 加载模型（创建集成）
    ensemble_weights = [0.4, 0.35, 0.25] if len(configs) >= 3 else None
    model = loader.load_best_models(
        configs,
        create_ensemble=len(configs) > 1,
        ensemble_weights=ensemble_weights[:len(configs)] if ensemble_weights else None
    )
    
    # 创建数据加载器
    print(f"\n创建数据加载器 (split={split})...")
    data_loader = create_data_loader(
        data_path=data_path,
        split=split,
        batch_size=8,
        num_workers=4,
        input_size=224,
        oct_num_frames=32,
        oct_frames_per_point=5
    )
    
    # 预测
    print(f"\n开始预测...")
    results = loader.predict_dataset(model, data_loader, return_labels=True)
    
    predictions = results['predictions']
    probabilities = results['probabilities']
    labels = results.get('labels', None)
    
    print(f"\n预测完成:")
    print(f"  预测样本数: {len(predictions)}")
    print(f"  预测概率范围: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    if labels is not None:
        print(f"  真实标签分布: {np.bincount(labels)}")
        print(f"  预测准确率: {np.mean(predictions == labels):.4f}")
    
    return predictions, probabilities, labels


def main():
    parser = argparse.ArgumentParser(description='使用真实模型运行The Lancet Primary Care实验')
    parser.add_argument('--data_path', type=str, default='5centers_multi',
                       help='数据路径')
    parser.add_argument('--experiments', type=str, nargs='+',
                       choices=['1', '2', '3', '4', 'all'],
                       default=['all'],
                       help='要运行的实验')
    parser.add_argument('--output_dir', type=str,
                       default='lancet_primary_care/results',
                       help='输出目录')
    parser.add_argument('--use_train_data', action='store_true',
                       help='使用训练集数据（默认使用测试集）')
    
    args = parser.parse_args()
    
    # 确定使用的数据集分割
    split = 'train' if args.use_train_data else 'test'
    
    print("\n" + "="*60)
    print("The Lancet Primary Care - 实验执行")
    print("="*60)
    print(f"数据路径: {args.data_path}")
    print(f"数据集分割: {split}")
    print(f"输出目录: {args.output_dir}")
    
    # 加载模型并生成预测
    try:
        predictions, probabilities, labels = load_model_and_predict(
            args.data_path,
            split=split
        )
    except Exception as e:
        print(f"\n❌ 模型加载或预测失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 运行实验
    if 'all' in args.experiments or '1' in args.experiments:
        print("\n" + "="*60)
        print("实验1: HPV+患者活检率降低实验")
        print("="*60)
        try:
            exp1 = BiopsyReductionExperiment(
                data_path=args.data_path,
                output_dir=os.path.join(args.output_dir, 'experiment1_results')
            )
            results1 = exp1.run_experiment(
                ai_predictions=predictions,
                ai_probabilities=probabilities,
                risk_threshold=0.5
            )
            exp1.plot_results(results1)
            print("✅ 实验1完成")
        except Exception as e:
            print(f"❌ 实验1失败: {e}")
            import traceback
            traceback.print_exc()
    
    if 'all' in args.experiments or '2' in args.experiments:
        print("\n" + "="*60)
        print("实验2: OCT灵敏度提升实验")
        print("="*60)
        try:
            exp2 = OCTSensitivityExperiment(
                data_path=args.data_path,
                output_dir=os.path.join(args.output_dir, 'experiment2_results')
            )
            results2 = exp2.run_experiment(
                ai_oct_probabilities=probabilities,
                threshold=0.5
            )
            exp2.plot_results(results2)
            print("✅ 实验2完成")
        except Exception as e:
            print(f"❌ 实验2失败: {e}")
            import traceback
            traceback.print_exc()
    
    if 'all' in args.experiments or '3' in args.experiments:
        print("\n" + "="*60)
        print("实验3: 筛查方案对比实验")
        print("="*60)
        try:
            exp3 = ScreeningComparisonExperiment(
                data_path=args.data_path,
                output_dir=os.path.join(args.output_dir, 'experiment3_results')
            )
            results3 = exp3.run_experiment(
                ai_predictions=predictions,
                ai_probabilities=probabilities,
                threshold=0.5
            )
            exp3.plot_results(results3)
            print("✅ 实验3完成")
        except Exception as e:
            print(f"❌ 实验3失败: {e}")
            import traceback
            traceback.print_exc()
    
    if 'all' in args.experiments or '4' in args.experiments:
        print("\n" + "="*60)
        print("实验4: 筛查流程优化实验")
        print("="*60)
        try:
            exp4 = WorkflowOptimizationExperiment(
                data_path=args.data_path,
                output_dir=os.path.join(args.output_dir, 'experiment4_results')
            )
            results4 = exp4.run_experiment(
                ai_probabilities=probabilities,
                high_risk_threshold=0.7,
                low_risk_threshold=0.3
            )
            exp4.plot_results(results4)
            print("✅ 实验4完成")
        except Exception as e:
            print(f"❌ 实验4失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("所有实验完成！")
    print("="*60)
    print(f"\n结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()

