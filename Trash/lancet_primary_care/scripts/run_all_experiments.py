#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行所有实验的主脚本
用于The Lancet Primary Care研究
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

# Import experiments
from lancet_primary_care.experiments.experiment1_biopsy_reduction import BiopsyReductionExperiment
from lancet_primary_care.experiments.experiment2_oct_sensitivity import OCTSensitivityExperiment
from lancet_primary_care.experiments.experiment3_screening_comparison import ScreeningComparisonExperiment
from lancet_primary_care.experiments.experiment4_workflow_optimization import WorkflowOptimizationExperiment


def load_model_predictions(model_path: str = None, data_path: str = None):
    """
    加载模型并生成预测
    实际使用时需要根据具体模型实现
    """
    # TODO: 实现模型加载和预测
    # 这里返回None，实验代码会使用模拟数据
    return None, None


def main():
    parser = argparse.ArgumentParser(description='运行The Lancet Primary Care所有实验')
    parser.add_argument('--data_path', type=str, default='5centers_multi',
                       help='数据路径')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径（可选）')
    parser.add_argument('--experiments', type=str, nargs='+',
                       choices=['1', '2', '3', '4', 'all'],
                       default=['all'],
                       help='要运行的实验（1=活检率降低, 2=OCT灵敏度, 3=筛查对比, 4=流程优化, all=全部）')
    parser.add_argument('--output_dir', type=str,
                       default='lancet_primary_care/results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 加载模型预测（如果提供）
    ai_predictions, ai_probabilities = load_model_predictions(
        args.model_path, args.data_path
    )
    
    if 'all' in args.experiments or '1' in args.experiments:
        print("\n" + "="*60)
        print("运行实验1: HPV+患者活检率降低实验")
        print("="*60)
        exp1 = BiopsyReductionExperiment(
            data_path=args.data_path,
            output_dir=os.path.join(args.output_dir, 'experiment1_results')
        )
        exp1.run_experiment(
            ai_predictions=ai_predictions,
            ai_probabilities=ai_probabilities
        )
    
    if 'all' in args.experiments or '2' in args.experiments:
        print("\n" + "="*60)
        print("运行实验2: OCT灵敏度提升实验")
        print("="*60)
        exp2 = OCTSensitivityExperiment(
            data_path=args.data_path,
            output_dir=os.path.join(args.output_dir, 'experiment2_results')
        )
        exp2.run_experiment(
            ai_oct_probabilities=ai_probabilities
        )
    
    if 'all' in args.experiments or '3' in args.experiments:
        print("\n" + "="*60)
        print("运行实验3: 筛查方案对比实验")
        print("="*60)
        exp3 = ScreeningComparisonExperiment(
            data_path=args.data_path,
            output_dir=os.path.join(args.output_dir, 'experiment3_results')
        )
        exp3.run_experiment(
            ai_predictions=ai_predictions,
            ai_probabilities=ai_probabilities
        )
    
    if 'all' in args.experiments or '4' in args.experiments:
        print("\n" + "="*60)
        print("运行实验4: 筛查流程优化实验")
        print("="*60)
        exp4 = WorkflowOptimizationExperiment(
            data_path=args.data_path,
            output_dir=os.path.join(args.output_dir, 'experiment4_results')
        )
        exp4.run_experiment(
            ai_probabilities=ai_probabilities
        )
    
    print("\n" + "="*60)
    print("所有实验完成！")
    print("="*60)
    print(f"\n结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()

