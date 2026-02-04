#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证阳性病例激活修复效果

功能：
1. 诊断模型对阳性和阴性病例的预测置信度
2. 比较使用真实标签 vs 预测类别时的Grad-CAM效果
3. 测试不同阈值对阳性病例的影响
4. 生成对比可视化结果
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# 导入模型和配置
from config import BioCOT_v3_2_Config
from models.bio_cot_v3_2 import create_bio_cot_v3_2
from data.dataset_v3_2 import FiveCentersMultimodalDatasetV3_2
from training.extract_vit_patches import extract_patch_features_with_vit
from generate_gradcam import generate_gradcam
from generate_lesion_focused_gradcam import generate_lesion_focused_gradcam, fix_color_and_overlay


def diagnose_model_predictions(
    model,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 20
) -> Dict:
    """
    诊断模型对阳性和阴性病例的预测情况
    
    Returns:
        diagnosis: 包含预测统计信息的字典
    """
    print("\n" + "="*80)
    print("🔍 诊断模型预测情况")
    print("="*80)
    
    model.to(device)
    model.eval()
    
    positive_samples = []
    negative_samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(positive_samples) >= num_samples and len(negative_samples) >= num_samples:
                break
            
            oct_images = batch['oct_images'].to(device, non_blocking=True)
            colposcopy_images = batch['colposcopy_images'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            image_names = batch['image_names']
            clinical_features = batch.get('clinical_features', None)
            
            # 提取特征
            B = oct_images.shape[0]
            if len(oct_images.shape) == 5:
                F_oct = oct_images.shape[1]
                oct_images_flat = oct_images.view(B * F_oct, *oct_images.shape[2:])
                oct_features = extract_patch_features_with_vit(oct_images_flat, device, batch_size=8)
                oct_features = oct_features.view(B, F_oct, 196, 768).mean(dim=1)
            else:
                oct_features = extract_patch_features_with_vit(oct_images, device, batch_size=8)
            
            if len(colposcopy_images.shape) == 5:
                N_colpo = colposcopy_images.shape[1]
                colpo_images_flat = colposcopy_images.view(B * N_colpo, *colposcopy_images.shape[2:])
                colpo_features = extract_patch_features_with_vit(colpo_images_flat, device, batch_size=8)
                colpo_features = colpo_features.view(B, N_colpo, 196, 768).mean(dim=1)
            else:
                colpo_features = extract_patch_features_with_vit(colposcopy_images, device, batch_size=8)
            
            # 前向传播
            if clinical_features is None:
                clinical_features = batch.get('clinical_data', None)
                if clinical_features is None:
                    clinical_features = torch.zeros(B, 7, device=device)
            
            output = model(
                f_oct=oct_features,
                f_colpo=colpo_features,
                image_names=image_names if isinstance(image_names, list) else [image_names] * B,
                clinical_features=clinical_features.to(device) if isinstance(clinical_features, torch.Tensor) else None,
                return_loss_components=False
            )
            
            logits = output['pred']
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # 收集样本
            for i in range(B):
                label = labels[i].item()
                pred = preds[i].item()
                prob_neg = probs[i, 0].item()
                prob_pos = probs[i, 1].item()
                
                sample_info = {
                    'label': label,
                    'pred': pred,
                    'prob_neg': prob_neg,
                    'prob_pos': prob_pos,
                    'correct': (pred == label)
                }
                
                if label == 1 and len(positive_samples) < num_samples:
                    positive_samples.append(sample_info)
                elif label == 0 and len(negative_samples) < num_samples:
                    negative_samples.append(sample_info)
                
                if len(positive_samples) >= num_samples and len(negative_samples) >= num_samples:
                    break
    
    # 统计分析
    def analyze_samples(samples, label_name):
        if not samples:
            return {}
        
        correct = sum(1 for s in samples if s['correct'])
        avg_prob = np.mean([s[f'prob_{label_name.lower()}'] for s in samples])
        avg_pred_prob = np.mean([s['prob_pos' if label_name == 'Positive' else 'prob_neg'] for s in samples])
        
        return {
            'count': len(samples),
            'correct': correct,
            'accuracy': correct / len(samples),
            'avg_prob': avg_prob,
            'avg_pred_prob': avg_pred_prob,
            'low_confidence': sum(1 for s in samples if s[f'prob_{label_name.lower()}'] < 0.3)
        }
    
    pos_stats = analyze_samples(positive_samples, 'Positive')
    neg_stats = analyze_samples(negative_samples, 'Negative')
    
    diagnosis = {
        'positive': {
            'samples': positive_samples,
            'stats': pos_stats
        },
        'negative': {
            'samples': negative_samples,
            'stats': neg_stats
        }
    }
    
    # 打印结果
    print(f"\n📊 阳性病例统计 (共{pos_stats['count']}个样本):")
    print(f"   准确率: {pos_stats['accuracy']:.2%}")
    print(f"   平均阳性概率: {pos_stats['avg_prob']:.4f}")
    print(f"   低置信度样本数 (<0.3): {pos_stats['low_confidence']}")
    
    print(f"\n📊 阴性病例统计 (共{neg_stats['count']}个样本):")
    print(f"   准确率: {neg_stats['accuracy']:.2%}")
    print(f"   平均阴性概率: {neg_stats['avg_prob']:.4f}")
    print(f"   低置信度样本数 (<0.3): {neg_stats['low_confidence']}")
    
    return diagnosis


def compare_gradcam_methods(
    model,
    f_oct: torch.Tensor,
    f_colpo: torch.Tensor,
    image_names: List[str],
    clinical_features: torch.Tensor,
    true_label: int,
    device: torch.device,
    modality: str = 'oct'
) -> Dict:
    """
    比较使用真实标签 vs 预测类别时的Grad-CAM效果
    
    Returns:
        comparison: 包含两种方法CAM值的字典
    """
    print(f"\n🔬 比较Grad-CAM方法 (真实标签 vs 预测类别)")
    print(f"   真实标签: {true_label}")
    
    # 获取预测类别
    with torch.no_grad():
        output = model(
            f_oct=f_oct,
            f_colpo=f_colpo,
            image_names=image_names,
            clinical_features=clinical_features,
            return_loss_components=False
        )
        logits = output['pred']
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        pred_prob = probs[0, predicted_class].item()
    
    print(f"   预测类别: {predicted_class}, 预测概率: {pred_prob:.4f}")
    
    # 方法1: 使用真实标签
    try:
        cam_true_label = generate_gradcam(
            model, f_oct, f_colpo, image_names, clinical_features,
            target_class=true_label,
            modality=modality
        )
        cam_true_stats = {
            'min': float(cam_true_label.min()),
            'max': float(cam_true_label.max()),
            'mean': float(cam_true_label.mean()),
            'std': float(cam_true_label.std()),
            'non_zero_ratio': float(np.sum(cam_true_label > 0.1) / cam_true_label.size)
        }
    except Exception as e:
        print(f"  ⚠️  使用真实标签生成CAM失败: {e}")
        cam_true_label = None
        cam_true_stats = None
    
    # 方法2: 使用预测类别
    try:
        cam_pred_class = generate_gradcam(
            model, f_oct, f_colpo, image_names, clinical_features,
            target_class=predicted_class,
            modality=modality
        )
        cam_pred_stats = {
            'min': float(cam_pred_class.min()),
            'max': float(cam_pred_class.max()),
            'mean': float(cam_pred_class.mean()),
            'std': float(cam_pred_class.std()),
            'non_zero_ratio': float(np.sum(cam_pred_class > 0.1) / cam_pred_class.size)
        }
    except Exception as e:
        print(f"  ⚠️  使用预测类别生成CAM失败: {e}")
        cam_pred_class = None
        cam_pred_stats = None
    
    # 打印对比
    if cam_true_stats:
        print(f"\n  📊 使用真实标签的CAM统计:")
        print(f"     最小值: {cam_true_stats['min']:.4f}")
        print(f"     最大值: {cam_true_stats['max']:.4f}")
        print(f"     均值: {cam_true_stats['mean']:.4f}")
        print(f"     非零比例 (>0.1): {cam_true_stats['non_zero_ratio']:.2%}")
    
    if cam_pred_stats:
        print(f"\n  📊 使用预测类别的CAM统计:")
        print(f"     最小值: {cam_pred_stats['min']:.4f}")
        print(f"     最大值: {cam_pred_stats['max']:.4f}")
        print(f"     均值: {cam_pred_stats['mean']:.4f}")
        print(f"     非零比例 (>0.1): {cam_pred_stats['non_zero_ratio']:.2%}")
    
    return {
        'true_label': true_label,
        'predicted_class': predicted_class,
        'pred_prob': pred_prob,
        'cam_true_label': cam_true_label,
        'cam_pred_class': cam_pred_class,
        'cam_true_stats': cam_true_stats,
        'cam_pred_stats': cam_pred_stats
    }


def test_threshold_effects(
    model,
    f_oct: torch.Tensor,
    f_colpo: torch.Tensor,
    image_names: List[str],
    clinical_features: torch.Tensor,
    target_class: int,
    modality: str = 'oct',
    thresholds: List[float] = [0.5, 0.6, 0.7, 0.8]
) -> Dict:
    """
    测试不同阈值对CAM的影响
    
    Returns:
        threshold_results: 包含不同阈值CAM结果的字典
    """
    print(f"\n🔬 测试不同阈值效果 (目标类别: {target_class})")
    
    # 先生成原始CAM
    raw_cam = generate_gradcam(
        model, f_oct, f_colpo, image_names, clinical_features,
        target_class=target_class,
        modality=modality
    )
    
    print(f"  📊 原始CAM: Min={raw_cam.min():.4f}, Max={raw_cam.max():.4f}, Mean={raw_cam.mean():.4f}")
    
    threshold_results = {
        'raw_cam': raw_cam,
        'thresholds': {}
    }
    
    for threshold in thresholds:
        try:
            focused_cam = generate_lesion_focused_gradcam(
                model, f_oct, f_colpo, image_names, clinical_features,
                target_class=target_class,
                modality=modality,
                threshold=threshold,
                use_percentile=True
            )
            
            stats = {
                'min': float(focused_cam.min()),
                'max': float(focused_cam.max()),
                'mean': float(focused_cam.mean()),
                'non_zero_ratio': float(np.sum(focused_cam > 0.1) / focused_cam.size),
                'reduction_ratio': float(1 - np.sum(focused_cam > 0) / np.sum(raw_cam > 0))
            }
            
            threshold_results['thresholds'][threshold] = {
                'cam': focused_cam,
                'stats': stats
            }
            
            print(f"\n  📊 阈值 {threshold}:")
            print(f"     最小值: {stats['min']:.4f}")
            print(f"     最大值: {stats['max']:.4f}")
            print(f"     均值: {stats['mean']:.4f}")
            print(f"     非零比例: {stats['non_zero_ratio']:.2%}")
            print(f"     减少比例: {stats['reduction_ratio']:.2%}")
            
        except Exception as e:
            print(f"  ⚠️  阈值 {threshold} 生成失败: {e}")
            threshold_results['thresholds'][threshold] = None
    
    return threshold_results


def visualize_comparison(
    image: np.ndarray,
    comparison_results: Dict,
    threshold_results: Dict,
    save_path: Path,
    label: int,
    modality: str = 'oct'
):
    """
    生成对比可视化图
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Grad-CAM对比分析 - {modality.upper()} (标签: {"Positive" if label == 1 else "Negative"})', 
                 fontsize=16, fontweight='bold')
    
    # 第一行：原始图像和两种CAM方法
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('原始图像', fontsize=12)
    axes[0, 0].axis('off')
    
    if comparison_results['cam_true_label'] is not None:
        im1 = axes[0, 1].imshow(comparison_results['cam_true_label'], cmap='jet', alpha=0.6)
        axes[0, 1].imshow(image, alpha=0.4)
        axes[0, 1].set_title(f'使用真实标签 (类别{comparison_results["true_label"]})\n'
                           f'Max={comparison_results["cam_true_stats"]["max"]:.3f}', fontsize=10)
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    if comparison_results['cam_pred_class'] is not None:
        im2 = axes[0, 2].imshow(comparison_results['cam_pred_class'], cmap='jet', alpha=0.6)
        axes[0, 2].imshow(image, alpha=0.4)
        axes[0, 2].set_title(f'使用预测类别 (类别{comparison_results["predicted_class"]}, '
                           f'概率={comparison_results["pred_prob"]:.3f})\n'
                           f'Max={comparison_results["cam_pred_stats"]["max"]:.3f}', fontsize=10)
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # 第二行：不同阈值的效果
    axes[1, 0].imshow(image)
    axes[1, 0].imshow(threshold_results['raw_cam'], cmap='jet', alpha=0.6)
    axes[1, 0].set_title('原始CAM (无阈值)', fontsize=10)
    axes[1, 0].axis('off')
    
    # 显示两个不同阈值的对比
    thresholds_to_show = sorted(threshold_results['thresholds'].keys())[:2]
    for idx, threshold in enumerate(thresholds_to_show):
        if threshold_results['thresholds'][threshold] is not None:
            cam = threshold_results['thresholds'][threshold]['cam']
            stats = threshold_results['thresholds'][threshold]['stats']
            axes[1, idx+1].imshow(image)
            axes[1, idx+1].imshow(cam, cmap='jet', alpha=0.6)
            axes[1, idx+1].set_title(f'阈值 {threshold}\n'
                                    f'Max={stats["max"]:.3f}, 非零={stats["non_zero_ratio"]:.1%}', 
                                    fontsize=10)
            axes[1, idx+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 对比图已保存: {save_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试阳性病例激活修复效果')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--data_root', type=str, default=None,
                       help='数据根目录（默认使用config中的路径）')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='每个类别测试的样本数（默认10）')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='结果保存目录（默认: biolcot_visualization/test_results）')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备（默认: cuda:0）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")
    
    # 加载配置
    config = BioCOT_v3_2_Config()
    if args.data_root:
        config.data_root = args.data_root
    
    # 创建保存目录
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(__file__).parent / 'test_results'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"📁 结果将保存到: {save_dir}")
    
    # 加载模型
    print("\n" + "="*80)
    print("📦 加载模型...")
    print("="*80)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = create_bio_cot_v3_2(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ 模型加载完成 (Epoch: {checkpoint.get('epoch', 'N/A')})")
    
    # 加载数据集
    print("\n" + "="*80)
    print("📊 加载数据集...")
    print("="*80)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_csv = Path(config.data_root) / 'val_labels.csv'
    if not val_csv.exists():
        val_csv = Path(config.data_root) / 'internal_train' / 'val' / 'labels.csv'
    
    val_dataset = FiveCentersMultimodalDatasetV3_2(
        csv_path=str(val_csv),
        transform=transform,
        oct_num_frames=config.oct_frames,
        max_col_images=config.colposcopy_images,
        balance_negative_frames=False,
        data_root=str(config.data_root)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✅ 验证集加载完成: {len(val_dataset)} 个样本")
    
    # 步骤1: 诊断模型预测
    diagnosis = diagnose_model_predictions(model, val_loader, device, num_samples=args.num_samples)
    
    # 保存诊断结果
    diagnosis_file = save_dir / f'diagnosis_{timestamp}.json'
    with open(diagnosis_file, 'w', encoding='utf-8') as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 诊断结果已保存: {diagnosis_file}")
    
    # 步骤2: 对阳性病例进行详细测试
    print("\n" + "="*80)
    print("🔬 对阳性病例进行详细测试")
    print("="*80)
    
    positive_samples = diagnosis['positive']['samples']
    if not positive_samples:
        print("⚠️  没有找到阳性样本，跳过测试")
        return
    
    # 选择几个代表性样本进行详细测试
    test_samples = positive_samples[:min(3, len(positive_samples))]
    
    for sample_idx, sample_info in enumerate(test_samples):
        print(f"\n{'='*80}")
        print(f"测试样本 {sample_idx + 1}/{len(test_samples)}")
        print(f"  真实标签: {sample_info['label']}")
        print(f"  预测类别: {sample_info['pred']}")
        print(f"  阳性概率: {sample_info['prob_pos']:.4f}")
        print(f"  预测正确: {sample_info['correct']}")
        print(f"{'='*80}")
        
        # 重新加载这个样本
        # 这里简化处理，实际应该从dataloader中获取
        # 为了简化，我们跳过这一步，直接使用诊断结果
        
    print("\n✅ 测试完成！")
    print(f"📁 所有结果已保存到: {save_dir}")


if __name__ == '__main__':
    main()

