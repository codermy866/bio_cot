#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.0 综合可视化脚本
包括：CAM图、t-SNE、箱线图、小提琴图等
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
from PIL import Image
import cv2

# 设置matplotlib后端
matplotlib.use('Agg')

# 设置学术论文风格
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'SimSun'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# 添加项目路径
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from config import BioCOT_v3_Config
from models.bio_cot_v3 import BioCOT_v3
from data.dataset_v3 import FiveCentersMultimodalDatasetV3
from training.extract_vit_patches import extract_patch_features_with_vit
from torch.utils.data import DataLoader
from torchvision import transforms


class BioCOT3_GradCAM:
    """Bio-COT 3.0的GradCAM可视化工具"""
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.handlers = []
        
        # 注册hook
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向hook"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # 找到目标层（DualHeadImageEncoder的最后一个层）
        if hasattr(self.model, 'dual_head_encoder'):
            if hasattr(self.model.dual_head_encoder, 'causal_head'):
                # 找到causal_head的最后一个线性层
                for module in reversed(list(self.model.dual_head_encoder.causal_head.modules())):
                    if isinstance(module, nn.Linear):
                        target_layer = module
                        break
                else:
                    target_layer = list(self.model.dual_head_encoder.causal_head.children())[-1]
            else:
                target_layer = list(self.model.dual_head_encoder.children())[-1]
        elif hasattr(self.model, 'classifier'):
            target_layer = list(self.model.classifier.children())[0]
        else:
            # 最后备用：使用模型的最后一个模块
            target_layer = list(self.model.children())[-1]
        
        self.handlers.append(target_layer.register_forward_hook(forward_hook))
        self.handlers.append(target_layer.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """移除hook"""
        for handle in self.handlers:
            handle.remove()
    
    def generate_cam(self, oct_images, colpo_images, knowledge_embeds, center_labels, target_class=None):
        """生成CAM图"""
        self.model.zero_grad()
        self.gradients = None
        self.activations = None
        
        # 提取patch特征
        oct_feats = extract_patch_features_with_vit(oct_images, next(self.model.parameters()).device)
        colpo_feats = extract_patch_features_with_vit(colpo_images, next(self.model.parameters()).device)
        
        # 前向传播
        outputs = self.model(
            f_oct=oct_feats,
            f_colpo=colpo_feats,
            note_embeds=knowledge_embeds,
            center_labels=center_labels,
            return_loss_components=False
        )
        
        logits = outputs['pred']
        
        # 确定目标类别
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        
        # 反向传播
        one_hot = torch.zeros_like(logits)
        one_hot[0][target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # 获取特征图和梯度
        if self.gradients is None or self.activations is None:
            return None
        
        # 计算CAM
        grads = self.gradients
        fmaps = self.activations
        
        # Global Average Pooling over gradients
        weights = torch.mean(grads, dim=(2, 3), keepdim=True) if len(grads.shape) == 4 else torch.mean(grads, dim=1, keepdim=True)
        
        # 加权求和
        cam = torch.sum(weights * fmaps, dim=1, keepdim=True) if len(fmaps.shape) == 4 else torch.sum(weights * fmaps, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # 归一化
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam.cpu().numpy()


def extract_features_for_tsne(model, dataloader, device, config):
    """提取特征用于t-SNE可视化"""
    model.eval()
    
    features_list = []
    labels_list = []
    centers_list = []
    predictions_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征"):
            oct_images = batch['oct_images'].to(device)
            colposcopy_images = batch['colposcopy_images'].to(device)
            labels = batch['label'].to(device)
            center_labels = batch['center_idx'].to(device)
            knowledge_embeddings = batch['knowledge_embedding'].to(device)
            
            # 提取patch特征
            oct_feats = extract_patch_features_with_vit(oct_images, device)
            colpo_feats = extract_patch_features_with_vit(colposcopy_images, device)
            
            # 前向传播
            outputs = model(
                f_oct=oct_feats,
                f_colpo=colpo_feats,
                note_embeds=knowledge_embeddings,
                center_labels=center_labels,
                return_loss_components=False
            )
            
            # 获取融合特征（用于t-SNE）
            # Bio-COT 3.0的输出结构：使用z_causal作为特征表示
            if 'z_causal' in outputs:
                features = outputs['z_causal'].cpu().numpy()
            else:
                # 备用：使用预测logits（降维后用于t-SNE）
                logits = outputs['pred'].cpu().numpy()
                # 如果logits维度太小，使用PCA扩展到更高维
                if logits.shape[1] < 64:
                    # 对于2类分类，logits只有2维，需要扩展到更高维用于t-SNE
                    # 使用简单的特征扩展（重复并添加噪声）
                    expanded = np.repeat(logits, 128, axis=1)  # [B, 256]
                    noise = np.random.randn(*expanded.shape) * 0.01
                    features = expanded + noise
                else:
                    features = logits
            
            predictions = torch.softmax(outputs['pred'], dim=1).cpu().numpy()
            
            features_list.append(features)
            labels_list.append(labels.cpu().numpy())
            centers_list.append(center_labels.cpu().numpy())
            predictions_list.append(predictions)
    
    features = np.vstack(features_list)
    labels = np.hstack(labels_list)
    centers = np.hstack(centers_list)
    predictions = np.vstack(predictions_list)
    
    return features, labels, centers, predictions


def visualize_tsne(features, labels, centers, predictions, output_dir, timestamp):
    """生成t-SNE可视化"""
    print("📊 正在生成t-SNE可视化...")
    
    # 降维到2D
    print("  降维中...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 1. 按标签着色
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.set_title('(a) t-SNE by Label', fontsize=13, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, label='Label')
    ax1.grid(True, alpha=0.3)
    
    # 2. 按中心着色
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], c=centers, cmap='tab10', alpha=0.6, s=50)
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax2.set_title('(b) t-SNE by Center', fontsize=13, fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, label='Center ID')
    ax2.grid(True, alpha=0.3)
    
    # 3. 按预测概率着色
    ax3 = axes[1, 0]
    probs = predictions[:, 1]  # 阳性概率
    scatter3 = ax3.scatter(features_2d[:, 0], features_2d[:, 1], c=probs, cmap='RdYlGn', alpha=0.6, s=50)
    ax3.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax3.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax3.set_title('(c) t-SNE by Prediction Probability', fontsize=13, fontweight='bold')
    plt.colorbar(scatter3, ax=ax3, label='Positive Probability')
    ax3.grid(True, alpha=0.3)
    
    # 4. 组合视图（标签+中心）
    ax4 = axes[1, 1]
    for center_id in np.unique(centers):
        mask = centers == center_id
        ax4.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   label=f'Center {center_id}', alpha=0.6, s=50)
    ax4.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax4.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax4.set_title('(d) t-SNE by Center (Legend)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"tsne_visualization_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ t-SNE可视化已保存: {output_path}")
    return output_path


def visualize_boxplots(history, output_dir, timestamp):
    """生成箱线图"""
    print("📊 正在生成箱线图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = np.array(range(1, len(history['val_auc']) + 1))
    
    # 1. AUC箱线图（按epoch分组）
    ax1 = axes[0, 0]
    data_auc = [history['val_auc']]
    bp1 = ax1.boxplot(data_auc, labels=['All Epochs'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('(a) AUC Distribution', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 准确率箱线图
    ax2 = axes[0, 1]
    data_acc = [history['val_acc']]
    bp2 = ax2.boxplot(data_acc, labels=['All Epochs'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightgreen')
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('(b) Accuracy Distribution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. F1-Score箱线图
    ax3 = axes[1, 0]
    data_f1 = [history['val_f1']]
    bp3 = ax3.boxplot(data_f1, labels=['All Epochs'], patch_artist=True)
    bp3['boxes'][0].set_facecolor('lightcoral')
    ax3.set_ylabel('F1-Score', fontsize=12)
    ax3.set_title('(c) F1-Score Distribution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 损失组件箱线图
    ax4 = axes[1, 1]
    loss_data = []
    loss_labels = []
    if history.get('cls_loss'):
        loss_data.append(history['cls_loss'])
        loss_labels.append('Cls Loss')
    if history.get('ot_loss') and any(v > 0 for v in history['ot_loss']):
        loss_data.append(history['ot_loss'])
        loss_labels.append('OT Loss')
    if history.get('sparse_loss') and any(v > 0 for v in history['sparse_loss']):
        loss_data.append(history['sparse_loss'])
        loss_labels.append('Sparse Loss')
    
    if loss_data:
        bp4 = ax4.boxplot(loss_data, labels=loss_labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp4['boxes'], colors[:len(bp4['boxes'])]):
            patch.set_facecolor(color)
        ax4.set_ylabel('Loss Value', fontsize=12)
        ax4.set_title('(d) Loss Components Distribution', fontsize=13, fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / f"boxplots_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 箱线图已保存: {output_path}")
    return output_path


def visualize_violinplots(history, output_dir, timestamp):
    """生成小提琴图"""
    print("📊 正在生成小提琴图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. AUC小提琴图
    ax1 = axes[0, 0]
    data_auc = pd.DataFrame({'AUC': history['val_auc'], 'Type': 'Validation'})
    sns.violinplot(data=data_auc, y='AUC', ax=ax1, inner='box', palette='Blues')
    ax1.set_title('(a) AUC Distribution (Violin Plot)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 准确率小提琴图
    ax2 = axes[0, 1]
    data_acc = pd.DataFrame({'Accuracy': history['val_acc'], 'Type': 'Validation'})
    sns.violinplot(data=data_acc, y='Accuracy', ax=ax2, inner='box', palette='Greens')
    ax2.set_title('(b) Accuracy Distribution (Violin Plot)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. F1-Score小提琴图
    ax3 = axes[1, 0]
    data_f1 = pd.DataFrame({'F1-Score': history['val_f1'], 'Type': 'Validation'})
    sns.violinplot(data=data_f1, y='F1-Score', ax=ax3, inner='box', palette='Reds')
    ax3.set_title('(c) F1-Score Distribution (Violin Plot)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 训练/验证对比小提琴图
    ax4 = axes[1, 1]
    train_data = pd.DataFrame({
        'Value': history['train_acc'],
        'Type': 'Train'
    })
    val_data = pd.DataFrame({
        'Value': history['val_acc'],
        'Type': 'Validation'
    })
    combined_data = pd.concat([train_data, val_data])
    sns.violinplot(data=combined_data, x='Type', y='Value', ax=ax4, inner='box', palette='Set2')
    ax4.set_title('(d) Train vs Validation Accuracy', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / f"violinplots_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 小提琴图已保存: {output_path}")
    return output_path


def visualize_cam_samples(model, dataloader, device, config, output_dir, timestamp, num_samples=8):
    """生成CAM图样本"""
    print("📊 正在生成CAM图...")
    
    gradcam = BioCOT3_GradCAM(model)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples:
                break
            
            oct_images = batch['oct_images'].to(device)
            colposcopy_images = batch['colposcopy_images'].to(device)
            labels = batch['label'].to(device)
            center_labels = batch['center_idx'].to(device)
            knowledge_embeddings = batch['knowledge_embedding'].to(device)
            
            # 只处理第一个样本
            oct_img = oct_images[0:1]
            colpo_img = colposcopy_images[0:1]
            label = labels[0].item()
            center = center_labels[0].item()
            knowledge = knowledge_embeddings[0:1]
            center_label = center_labels[0:1]
            
            # 生成CAM
            cam = gradcam.generate_cam(oct_img, colpo_img, knowledge, center_label)
            
            if cam is not None:
                # 可视化（使用OCT图像）
                img = oct_img[0].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())
                
                # 上采样CAM到图像尺寸
                cam_resized = cv2.resize(cam[0, 0], (img.shape[1], img.shape[0]))
                cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())
                
                # 叠加
                ax = axes[count]
                ax.imshow(img)
                ax.imshow(cam_resized, alpha=0.5, cmap='jet')
                ax.set_title(f'Sample {count+1}\nLabel: {label}, Center: {center}', fontsize=10)
                ax.axis('off')
                
                count += 1
    
    gradcam.remove_hooks()
    
    # 隐藏多余的子图
    for i in range(count, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / f"cam_samples_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ CAM图已保存: {output_path}")
    return output_path


def main():
    """主函数"""
    print("=" * 80)
    print("Bio-COT 3.0 综合可视化")
    print("=" * 80)
    
    # 配置
    config = BioCOT_v3_Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载训练历史（用于箱线图和小提琴图）
    history_files = sorted(output_dir.glob("training_history_*.json"), 
                          key=lambda x: x.stat().st_mtime, reverse=True)
    if history_files:
        with open(history_files[0], 'r') as f:
            history = json.load(f)
        print(f"✅ 加载训练历史: {history_files[0].name}")
        
        # 生成箱线图
        visualize_boxplots(history, output_dir, timestamp)
        
        # 生成小提琴图
        visualize_violinplots(history, output_dir, timestamp)
    else:
        print("⚠️ 未找到训练历史文件，跳过箱线图和小提琴图")
    
    # 2. 加载模型和数据（用于t-SNE和CAM）
    print("\n📥 加载模型和数据...")
    
    # 加载最佳模型
    checkpoint_files = sorted(Path(config.checkpoint_dir).glob("best_model_v3_*.pth"),
                             key=lambda x: x.stat().st_mtime, reverse=True)
    if not checkpoint_files:
        print("❌ 未找到模型检查点，跳过t-SNE和CAM可视化")
        return
    
    checkpoint_path = checkpoint_files[0]
    print(f"✅ 加载模型: {checkpoint_path.name}")
    
    # 创建模型
    model = BioCOT_v3(
        embed_dim=config.embed_dim,
        num_classes=config.num_classes,
        num_centers=config.num_centers,
        use_visual_notes=config.use_visual_notes,
        use_ot=config.use_ot,
        use_dual=config.use_dual,
        use_cross_attn=config.use_cross_attn,
        warmup_epochs=config.warmup_epochs
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # 加载数据集
    dataset = FiveCentersMultimodalDatasetV3(
        data_root=config.data_root,
        split='val',
        knowledge_embed_path=config.knowledge_embed_path
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # 生成t-SNE可视化
    print("\n📊 生成t-SNE可视化...")
    features, labels, centers, predictions = extract_features_for_tsne(model, dataloader, device, config)
    visualize_tsne(features, labels, centers, predictions, output_dir, timestamp)
    
    # 生成CAM图
    print("\n📊 生成CAM图...")
    visualize_cam_samples(model, dataloader, device, config, output_dir, timestamp, num_samples=8)
    
    print("\n" + "=" * 80)
    print("✅ 所有可视化已完成！")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print(f"时间戳: {timestamp}")


if __name__ == '__main__':
    main()

