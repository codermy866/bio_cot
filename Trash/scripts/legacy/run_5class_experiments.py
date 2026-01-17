#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5分类实验完整流程
包含数据集创建、模型训练、评估和分析的完整流程
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import subprocess

class FiveClassExperimentRunner:
    def __init__(self, base_data_path='5centers_multi', output_base_dir='5class_experiments'):
        self.base_data_path = base_data_path
        self.output_base_dir = output_base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 实验配置
        self.experiments = {
            'baseline': {
                'name': '基线模型',
                'use_focal_loss': False,
                'use_label_smoothing': False,
                'use_causal_adjustment': False,
                'use_text_contrastive': False,
                'epochs': 15,
                'lr': 1e-4
            },
            'focal_loss': {
                'name': 'Focal Loss',
                'use_focal_loss': True,
                'use_label_smoothing': False,
                'use_causal_adjustment': False,
                'use_text_contrastive': False,
                'epochs': 15,
                'lr': 1e-4
            },
            'label_smoothing': {
                'name': '标签平滑',
                'use_focal_loss': False,
                'use_label_smoothing': True,
                'use_causal_adjustment': False,
                'use_text_contrastive': False,
                'epochs': 15,
                'lr': 1e-4
            },
            'causal_adjustment': {
                'name': '因果调整',
                'use_focal_loss': True,
                'use_label_smoothing': False,
                'use_causal_adjustment': True,
                'use_text_contrastive': False,
                'epochs': 20,
                'lr': 5e-5
            },
            'text_contrastive': {
                'name': '文本对比学习',
                'use_focal_loss': True,
                'use_label_smoothing': False,
                'use_causal_adjustment': False,
                'use_text_contrastive': True,
                'epochs': 20,
                'lr': 5e-5
            },
            'full_enhanced': {
                'name': '全增强模型',
                'use_focal_loss': True,
                'use_label_smoothing': True,
                'use_causal_adjustment': True,
                'use_text_contrastive': True,
                'epochs': 25,
                'lr': 3e-5
            }
        }
    
    def create_dataset(self):
        """创建5分类数据集"""
        print("📊 步骤1: 创建5分类数据集")
        print("=" * 50)
        
        cmd = f"python five_class_dataset_creator.py"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 5分类数据集创建成功")
            return True
        else:
            print(f"❌ 数据集创建失败: {result.stderr}")
            return False
    
    def train_model(self, exp_name, exp_config):
        """训练单个模型"""
        print(f"\n🚀 步骤2: 训练{exp_config['name']}模型")
        print("=" * 50)
        
        # 创建输出目录
        output_dir = os.path.join(self.output_base_dir, f"{exp_name}_{self.timestamp}")
        
        # 构建训练命令
        cmd_parts = [
            "python train_5class_model.py",
            f"--data_path 5centers_multi_5class",
            f"--output_dir {output_dir}",
            f"--epochs {exp_config['epochs']}",
            f"--lr {exp_config['lr']}"
        ]
        
        if exp_config['use_focal_loss']:
            cmd_parts.append("--use_focal_loss")
        if exp_config['use_label_smoothing']:
            cmd_parts.append("--use_label_smoothing")
        if exp_config['use_causal_adjustment']:
            cmd_parts.append("--use_causal_adjustment")
        if exp_config['use_text_contrastive']:
            cmd_parts.append("--use_text_contrastive")
        
        cmd = " ".join(cmd_parts)
        
        print(f"执行命令: {cmd}")
        
        # 执行训练
        start_time = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ {exp_config['name']}模型训练成功")
            print(f"⏱️ 训练时间: {end_time - start_time:.1f}秒")
            return output_dir
        else:
            print(f"❌ {exp_config['name']}模型训练失败: {result.stderr}")
            return None
    
    def evaluate_model(self, model_dir, exp_name):
        """评估模型"""
        print(f"\n📈 步骤3: 评估{exp_name}模型")
        print("=" * 50)
        
        # 创建评估脚本
        eval_script = f"""
import torch
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from cnn_multimodal_model_5class import create_5class_model
from enhanced_multimodal_dataset import build_enhanced_dataset

def evaluate_5class_model(model_path, data_path):
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    model = create_5class_model(
        num_classes=checkpoint['num_classes'],
        embed_dim=512,
        clinical_dim=8
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # 加载数据
    _, test_dataset = build_enhanced_dataset(data_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 评估
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for oct_data, col_data, clinical_data, labels, metadata in test_loader:
            outputs = model(oct_data, col_data, clinical_data)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算指标
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    try:
        auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        auc_weighted = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except:
        auc_macro = 0.0
        auc_weighted = 0.0
    
    # 分类报告
    class_names = checkpoint['class_names']
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 保存结果
    results = {{
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }}
    
    with open('{model_dir}/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig('{model_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"评估完成:")
    print(f"  准确率: {{accuracy:.4f}}")
    print(f"  F1(macro): {{f1_macro:.4f}}")
    print(f"  F1(weighted): {{f1_weighted:.4f}}")
    print(f"  AUC(macro): {{auc_macro:.4f}}")
    print(f"  AUC(weighted): {{auc_weighted:.4f}}")
    
    return results

if __name__ == "__main__":
    model_path = "{model_dir}/best_model_5class.pth"
    data_path = "5centers_multi_5class"
    evaluate_5class_model(model_path, data_path)
"""
        
        # 保存并执行评估脚本
        eval_script_path = f"{model_dir}/evaluate_model.py"
        with open(eval_script_path, 'w') as f:
            f.write(eval_script)
        
        cmd = f"python {eval_script_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {exp_name}模型评估完成")
            return True
        else:
            print(f"❌ {exp_name}模型评估失败: {result.stderr}")
            return False
    
    def compare_experiments(self, experiment_results):
        """比较所有实验结果"""
        print(f"\n📊 步骤4: 比较所有实验结果")
        print("=" * 50)
        
        # 收集所有实验结果
        comparison_data = []
        
        for exp_name, model_dir in experiment_results.items():
            if model_dir and os.path.exists(f"{model_dir}/evaluation_results.json"):
                with open(f"{model_dir}/evaluation_results.json", 'r') as f:
                    results = json.load(f)
                
                comparison_data.append({
                    'experiment': exp_name,
                    'accuracy': results['accuracy'],
                    'f1_macro': results['f1_macro'],
                    'f1_weighted': results['f1_weighted'],
                    'precision_macro': results['precision_macro'],
                    'recall_macro': results['recall_macro'],
                    'auc_macro': results['auc_macro'],
                    'auc_weighted': results['auc_weighted']
                })
        
        if not comparison_data:
            print("❌ 没有可比较的实验结果")
            return
        
        # 保存比较结果
        comparison_file = os.path.join(self.output_base_dir, f"experiment_comparison_{self.timestamp}.json")
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=4, ensure_ascii=False)
        
        # 打印比较结果
        print("📈 实验结果比较:")
        print("-" * 80)
        print(f"{'实验名称':<15} {'准确率':<8} {'F1(macro)':<10} {'F1(weighted)':<12} {'AUC(macro)':<10} {'AUC(weighted)':<12}")
        print("-" * 80)
        
        for data in comparison_data:
            print(f"{data['experiment']:<15} {data['accuracy']:<8.4f} {data['f1_macro']:<10.4f} "
                  f"{data['f1_weighted']:<12.4f} {data['auc_macro']:<10.4f} {data['auc_weighted']:<12.4f}")
        
        # 找出最佳模型
        best_model = max(comparison_data, key=lambda x: x['f1_macro'])
        print(f"\n🏆 最佳模型: {best_model['experiment']}")
        print(f"   F1(macro): {best_model['f1_macro']:.4f}")
        print(f"   AUC(macro): {best_model['auc_macro']:.4f}")
        
        print(f"\n📁 比较结果保存在: {comparison_file}")
    
    def run_all_experiments(self):
        """运行所有实验"""
        print("🚀 开始5分类完整实验流程")
        print("=" * 80)
        
        # 创建输出目录
        os.makedirs(self.output_base_dir, exist_ok=True)
        
        # 步骤1: 创建数据集
        if not self.create_dataset():
            print("❌ 数据集创建失败，终止实验")
            return
        
        # 步骤2: 训练所有模型
        experiment_results = {}
        
        for exp_name, exp_config in self.experiments.items():
            print(f"\n{'='*60}")
            print(f"🧪 实验: {exp_config['name']}")
            print(f"{'='*60}")
            
            model_dir = self.train_model(exp_name, exp_config)
            experiment_results[exp_name] = model_dir
            
            if model_dir:
                # 步骤3: 评估模型
                self.evaluate_model(model_dir, exp_name)
        
        # 步骤4: 比较实验结果
        self.compare_experiments(experiment_results)
        
        print(f"\n🎉 所有实验完成！")
        print(f"📁 结果保存在: {self.output_base_dir}")
        
        return experiment_results

def main():
    parser = argparse.ArgumentParser(description='5分类完整实验流程')
    parser.add_argument('--base_data_path', type=str, default='5centers_multi', help='基础数据路径')
    parser.add_argument('--output_base_dir', type=str, default='5class_experiments', help='输出基础目录')
    parser.add_argument('--experiment', type=str, default='all', 
                       choices=['all', 'baseline', 'focal_loss', 'label_smoothing', 'causal_adjustment', 'text_contrastive', 'full_enhanced'],
                       help='要运行的实验')
    
    args = parser.parse_args()
    
    runner = FiveClassExperimentRunner(args.base_data_path, args.output_base_dir)
    
    if args.experiment == 'all':
        runner.run_all_experiments()
    else:
        # 运行单个实验
        if args.experiment in runner.experiments:
            exp_config = runner.experiments[args.experiment]
            
            # 创建数据集
            runner.create_dataset()
            
            # 训练模型
            model_dir = runner.train_model(args.experiment, exp_config)
            
            if model_dir:
                # 评估模型
                runner.evaluate_model(model_dir, args.experiment)
        else:
            print(f"❌ 未知实验: {args.experiment}")

if __name__ == "__main__":
    main()
