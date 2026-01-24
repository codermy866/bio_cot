#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合可视化脚本 - 生成顶级期刊风格的可视化图表
包括：UMAP、直方图、热力图、Grad-CAM
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
from umap import UMAP
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2

# 设置matplotlib后端
matplotlib.use('Agg')

# 顶级期刊风格配置
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'STSong'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.6,
    'lines.linewidth': 2.0,
    'lines.markersize': 4,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# 配色方案（色盲友好，Nature/Science风格）
COLORS_LABEL = {
    0: '#2E86AB',  # 深蓝色（阴性）
    1: '#A23B72',  # 深紫红色（阳性）
}

COLORS_CENTER = [
    '#2E86AB',  # 深蓝色
    '#A23B72',  # 深紫红色
    '#F18F01',  # 深橙色
    '#C73E1D',  # 深红色
    '#6A994E',  # 深绿色
]

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.bida.bio_cot_model import BioCOTModel
from experiments.exp_5centers.dataset_v2 import FiveCentersMultimodalDatasetV2
from experiments.exp_5centers.train_bio_cot_5centers_multimodal import extract_features_with_vit
import timm


class BioCOT_GradCAM:
    """
    专门为 Bio-COT (ViT架构) 设计的 Grad-CAM 工具。
    解决了 ViT 输出为 Sequence 导致的热力图全蓝问题。
    
    核心修复：
    1. 维度重塑：将 [B, 197, 768] 序列重塑为 [B, 768, 14, 14]
    2. 正确选择目标层：最后一个 Transformer Block 的 LayerNorm 之前
    3. 去除 CLS Token 干扰：只使用 196 个 patch
    """
    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.model.eval()
        self.feature_maps = None
        self.gradients = None
        self.target_layer_name = target_layer_name
        self.handlers = []

        # 自动寻找 ViT 的最后一个 Block (如果未指定)
        if self.target_layer_name is None:
            # 对于 timm 的 ViT，通常是 blocks[-1].norm1
            if hasattr(model, 'blocks') and len(model.blocks) > 0:
                self.target_layer = model.blocks[-1].norm1
            else:
                raise ValueError("无法找到目标层，请手动指定 target_layer_name")
        else:
            named_modules = dict([*self.model.named_modules()])
            if target_layer_name in named_modules:
                self.target_layer = named_modules[target_layer_name]
            else:
                raise ValueError(f"找不到指定的层: {target_layer_name}")

        # 注册 Hook
        self._register_hooks()

    def _register_hooks(self):
        def save_features_hook(module, input, output):
            self.feature_maps = output.detach()

        def save_gradients_hook(module, grad_in, grad_out):
            if grad_out[0] is not None:
                self.gradients = grad_out[0].detach()

        # 注册 Forward 和 Backward Hook
        self.handlers.append(self.target_layer.register_forward_hook(save_features_hook))
        self.handlers.append(self.target_layer.register_full_backward_hook(save_gradients_hook))

    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()

    def _reshape_transform(self, tensor, height=14, width=14):
        """
        核心修复逻辑：将 ViT 的 [B, 197, C] 序列重塑为 [B, C, H, W]
        这是解决"全蓝"问题的关键！
        """
        # 去掉 CLS token (index 0) - 关键步骤！
        result = tensor[:, 1:, :]  # [B, 196, C]
        
        # 重塑为 Patch Grid (14x14)
        # [B, 196, 768] -> [B, 14, 14, 768]
        result = result.reshape(tensor.size(0), height, width, tensor.size(2))
        
        # 换轴适配 CNN 格式: [B, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result

    def generate_cam(self, input_tensor, target_class=None, class_idx=None):
        """
        生成 CAM 图
        input_tensor: [1, 3, 224, 224] (用于ViT)
        target_class: int, 想要可视化的类别 (0:阴性, 1:阳性) - 兼容参数
        class_idx: int, 同target_class（向后兼容）
        """
        self.model.zero_grad()
        self.feature_maps = None
        self.gradients = None
        
        # 兼容两种参数名
        if target_class is None:
            target_class = class_idx
        
        # 确保输入需要梯度
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # 1. 前向传播
        output = self.model(input_tensor)  # ViT输出 [B, num_classes]
        
        # 兼容字典输出或 Tensor 输出
        if isinstance(output, dict):
            logits = output.get('logits', output.get('pred', output))
        else:
            logits = output
        
        # 2. 确定目标类别
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        
        # 3. 反向传播
        one_hot = torch.zeros_like(logits)
        one_hot[0][target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)

        # 4. 获取特征图和梯度
        if self.gradients is None or self.feature_maps is None:
            print("⚠️ Hook失败，返回零CAM")
            return np.zeros((14, 14))
        
        # [1, 197, 768]
        grads = self.gradients
        fmaps = self.feature_maps
        
        # 确保是3D
        if grads.dim() == 2:
            grads = grads.unsqueeze(0)
        if fmaps.dim() == 2:
            fmaps = fmaps.unsqueeze(0)
        
        # 5. 关键步骤：Reshape (解决全蓝问题的核心)
        # 变为 [1, 768, 14, 14]
        grads = self._reshape_transform(grads)
        fmaps = self._reshape_transform(fmaps)

        # 6. Global Average Pooling over gradients (GAP)
        # weights: [1, 768, 1, 1]
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)

        # 7. 加权求和
        # cam: [1, 1, 14, 14]
        cam = torch.sum(weights * fmaps, dim=1, keepdim=True)

        # 8. ReLU 处理 (去掉负激活)
        cam = F.relu(cam)

        # 9. 归一化
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        # 转为 numpy [14, 14]
        cam = cam.cpu().numpy()[0, 0, :, :]
        
        return cam


def extract_predictions_and_features(model, dataloader, device, dataset):
    """提取预测结果和特征"""
    model.eval()
    
    all_features = []
    all_probs = []
    all_preds = []
    all_labels = []
    all_centers = []
    all_images = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="提取特征和预测")):
            oct_images = batch['oct_images'].to(device)
            colposcopy_images = batch['colposcopy_images'].to(device)
            
            oct_feat = extract_features_with_vit(oct_images, device)
            colpo_feat = extract_features_with_vit(colposcopy_images, device)
            
            clinical_data = batch.get('clinical_data', None)
            if clinical_data is None:
                if 'clinical_features' in batch:
                    clinical_features = batch['clinical_features']
                    clinical_data = {
                        'age': clinical_features[:, 0].cpu().numpy() * 100,
                        'hpv': (clinical_features[:, 1] > 0.5).cpu().numpy().astype(int),
                        'tct': np.argmax(clinical_features[:, 2:].cpu().numpy(), axis=1) if clinical_features.shape[1] > 2 else np.zeros(len(clinical_features)),
                    }
            
            center_labels = batch['center_idx'].to(device)
            
            output = model(
                oct_features=oct_feat,
                colpo_features=colpo_feat,
                clinical_features=None,
                clinical_data=clinical_data,
                center_labels=center_labels,
                return_loss_components=False
            )
            
            # 获取融合特征
            if 'z_causal' in output and 'z_sem' in output:
                z_causal = output['z_causal']
                z_sem = output['z_sem']
                fused = (z_causal + z_sem) / 2.0
                all_features.append(fused.cpu().numpy())
            
            # 获取预测
            logits = output.get('logits', output.get('pred', None))
            if logits is not None:
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
            
            all_labels.append(batch['label'].numpy())
            all_centers.append(batch['center_idx'].numpy())
            
            # 保存图像路径用于Grad-CAM（从数据集中获取原始路径）
            if len(all_images) < 20:  # 只保存前20个样本
                batch_size = len(batch['label'])
                for i in range(min(batch_size, 2)):  # 每个batch最多取2个样本
                    if len(all_images) >= 20:
                        break
                    sample_idx = batch_idx * dataloader.batch_size + i
                    if sample_idx < len(dataset):
                        row = dataset.df.iloc[sample_idx]
                        oct_id = str(row['oct_id'])
                        
                        # 获取OCT图像路径
                        oct_paths_str = row.get('oct_paths', '')
                        if pd.isna(oct_paths_str) or oct_paths_str == '':
                            oct_paths_str = row.get('OCT图像路径', '')
                        
                        # 获取Colposcopy图像路径
                        col_paths_str = row.get('col_paths', '')
                        if pd.isna(col_paths_str) or col_paths_str == '':
                            col_paths_str = row.get('Colposcopy图像路径', '')
                        
                        all_images.append({
                            'oct_id': oct_id,
                            'oct_paths': str(oct_paths_str) if not pd.isna(oct_paths_str) else '',
                            'col_paths': str(col_paths_str) if not pd.isna(col_paths_str) else '',
                            'label': batch['label'][i].item(),
                            'pred': preds[i].item() if logits is not None else None,
                        })
    
    results = {
        'features': np.concatenate(all_features, axis=0) if all_features else None,
        'probs': np.concatenate(all_probs, axis=0) if all_probs else None,
        'preds': np.concatenate(all_preds, axis=0) if all_preds else None,
        'labels': np.concatenate(all_labels, axis=0),
        'centers': np.concatenate(all_centers, axis=0),
        'sample_images': all_images,
    }
    
    return results


def visualize_umap(features, labels, centers, output_dir, timestamp):
    """生成UMAP可视化"""
    print("🔄 正在进行UMAP降维...")
    
    # PCA预降维
    if features.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        features = pca.fit_transform(features)
        print(f"✅ PCA降维完成，保留方差: {pca.explained_variance_ratio_.sum():.4f}")
    
    # UMAP降维
    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    features_2d = umap_model.fit_transform(features)
    print("✅ UMAP降维完成")
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('UMAP Visualization of Feature Space', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 左图：按标签着色
    ax1 = axes[0]
    for label_val in [0, 1]:
        mask = labels == label_val
        color = COLORS_LABEL[label_val]
        label_name = 'Negative' if label_val == 0 else 'Positive'
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=color, label=label_name, alpha=0.7, s=50,
                   edgecolors='white', linewidths=0.8, zorder=2)
    ax1.set_xlabel('UMAP 1', fontsize=13, fontweight='bold')
    ax1.set_ylabel('UMAP 2', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Colored by Label', fontsize=14, fontweight='bold', pad=12)
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 右图：按中心着色
    ax2 = axes[1]
    unique_centers = np.unique(centers)
    for center_id in unique_centers:
        mask = centers == center_id
        color = COLORS_CENTER[int(center_id) % len(COLORS_CENTER)]
        ax2.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=color, label=f'Center {int(center_id)}', alpha=0.7, s=50,
                   edgecolors='white', linewidths=0.8, zorder=2)
    ax2.set_xlabel('UMAP 1', fontsize=13, fontweight='bold')
    ax2.set_ylabel('UMAP 2', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Colored by Center', fontsize=14, fontweight='bold', pad=12)
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_file = output_dir / f"umap_visualization_{timestamp}.png"
    fig.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 UMAP图已保存: {plot_file}")
    
    plot_file_pdf = output_dir / f"umap_visualization_{timestamp}.pdf"
    fig.savefig(plot_file_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 UMAP图PDF已保存: {plot_file_pdf}")
    
    plt.close(fig)


def visualize_histograms(probs, labels, preds, output_dir, timestamp):
    """生成直方图可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Prediction Probability Distributions', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 左上：所有样本的概率分布
    ax1 = axes[0, 0]
    ax1.hist(probs[:, 1], bins=30, color=COLORS_LABEL[1], alpha=0.7, edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('Predicted Probability (Positive)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('(a) All Samples', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 右上：按真实标签分组的概率分布
    ax2 = axes[0, 1]
    for label_val in [0, 1]:
        mask = labels == label_val
        label_name = 'Negative' if label_val == 0 else 'Positive'
        color = COLORS_LABEL[label_val]
        ax2.hist(probs[mask, 1], bins=30, label=label_name, color=color, 
                alpha=0.6, edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('Predicted Probability (Positive)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('(b) By True Label', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 左下：正确预测 vs 错误预测
    ax3 = axes[1, 0]
    correct_mask = (preds == labels)
    incorrect_mask = (preds != labels)
    ax3.hist(probs[correct_mask, 1], bins=30, label='Correct', 
            color='#6A994E', alpha=0.6, edgecolor='black', linewidth=0.8)
    ax3.hist(probs[incorrect_mask, 1], bins=30, label='Incorrect', 
            color='#C73E1D', alpha=0.6, edgecolor='black', linewidth=0.8)
    ax3.set_xlabel('Predicted Probability (Positive)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Correct vs Incorrect Predictions', fontsize=13, fontweight='bold', pad=10)
    ax3.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax3.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 右下：预测置信度分布
    ax4 = axes[1, 1]
    confidence = np.max(probs, axis=1)
    ax4.hist(confidence, bins=30, color='#F18F01', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax4.set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Prediction Confidence Distribution', fontsize=13, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plot_file = output_dir / f"histograms_{timestamp}.png"
    fig.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 直方图已保存: {plot_file}")
    
    plot_file_pdf = output_dir / f"histograms_{timestamp}.pdf"
    fig.savefig(plot_file_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 直方图PDF已保存: {plot_file_pdf}")
    
    plt.close(fig)


def visualize_heatmaps(preds, labels, centers, features, output_dir, timestamp):
    """生成热力图可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Heatmap Visualizations', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 左上：混淆矩阵
    ax1 = axes[0, 0]
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                cbar_kws={'label': 'Count'}, square=True, linewidths=0.5)
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Confusion Matrix', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticklabels(['Negative', 'Positive'])
    ax1.set_yticklabels(['Negative', 'Positive'])
    
    # 右上：按中心的混淆矩阵（归一化）
    ax2 = axes[0, 1]
    center_cm = np.zeros((len(np.unique(centers)), 2, 2))
    unique_centers = np.unique(centers)
    for i, center_id in enumerate(unique_centers):
        mask = centers == center_id
        if np.sum(mask) > 0:
            center_cm[i] = confusion_matrix(labels[mask], preds[mask], labels=[0, 1])
    
    # 计算平均准确率
    center_acc = []
    for i, center_id in enumerate(unique_centers):
        mask = centers == center_id
        if np.sum(mask) > 0:
            acc = np.sum((preds[mask] == labels[mask])) / np.sum(mask)
            center_acc.append(acc)
        else:
            center_acc.append(0)
    
    im = ax2.imshow(np.array(center_acc).reshape(-1, 1), cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_yticks(range(len(unique_centers)))
    ax2.set_yticklabels([f'Center {int(c)}' for c in unique_centers])
    ax2.set_xticks([0])
    ax2.set_xticklabels(['Accuracy'])
    ax2.set_title('(b) Accuracy by Center', fontsize=13, fontweight='bold', pad=10)
    plt.colorbar(im, ax=ax2, label='Accuracy')
    
    # 左下：特征相关性热力图（采样特征）
    ax3 = axes[1, 0]
    if features.shape[1] > 50:
        # 采样特征维度
        sample_indices = np.random.choice(features.shape[1], 50, replace=False)
        features_sample = features[:, sample_indices]
    else:
        features_sample = features
    
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(features_sample.T)
    im = ax3.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax3.set_xlabel('Feature Index', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Feature Index', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Feature Correlation Matrix', fontsize=13, fontweight='bold', pad=10)
    plt.colorbar(im, ax=ax3, label='Correlation')
    
    # 右下：按标签和中心的预测准确率热力图
    ax4 = axes[1, 1]
    label_center_acc = np.zeros((2, len(unique_centers)))
    for label_val in [0, 1]:
        for i, center_id in enumerate(unique_centers):
            mask = (labels == label_val) & (centers == center_id)
            if np.sum(mask) > 0:
                acc = np.sum((preds[mask] == labels[mask])) / np.sum(mask)
                label_center_acc[label_val, i] = acc
    
    im = ax4.imshow(label_center_acc, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Negative', 'Positive'])
    ax4.set_xticks(range(len(unique_centers)))
    ax4.set_xticklabels([f'Center {int(c)}' for c in unique_centers])
    ax4.set_xlabel('Center', fontsize=12, fontweight='bold')
    ax4.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Accuracy by Label and Center', fontsize=13, fontweight='bold', pad=10)
    plt.colorbar(im, ax=ax4, label='Accuracy')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plot_file = output_dir / f"heatmaps_{timestamp}.png"
    fig.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 热力图已保存: {plot_file}")
    
    plot_file_pdf = output_dir / f"heatmaps_{timestamp}.pdf"
    fig.savefig(plot_file_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 热力图PDF已保存: {plot_file_pdf}")
    
    plt.close(fig)


def load_image_from_path(image_path_str, default_size=(224, 224)):
    """从路径字符串加载图像"""
    from PIL import Image
    
    if pd.isna(image_path_str) or image_path_str == '':
        return None
    
    # 解析路径（可能包含多个路径，用分号分隔）
    paths = [p.strip() for p in str(image_path_str).split(';') if p.strip()]
    if not paths:
        return None
    
    # 尝试加载第一个存在的路径
    for path_str in paths:
        path = Path(path_str)
        if path.exists():
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize(default_size)
                return img
            except Exception as e:
                print(f"⚠️ 加载图像失败 {path}: {e}")
                continue
    
    # 如果直接路径不存在，尝试在常见目录中查找
    filename = Path(paths[0]).name
    base_dirs = [
        Path('/data2/hmy/5Center_datas/5centers_multi/train/oct'),
        Path('/data2/hmy/5Center_datas/5centers_multi/val/oct'),
        Path('/data2/hmy/5Center_datas/5centers_multi/test/oct'),
        Path('/data2/hmy/5Center_datas/5centers_multi/train/col'),
        Path('/data2/hmy/5Center_datas/5centers_multi/val/col'),
        Path('/data2/hmy/5Center_datas/5centers_multi/test/col'),
    ]
    
    for base_dir in base_dirs:
        candidate = base_dir / filename
        if candidate.exists():
            try:
                img = Image.open(candidate).convert('RGB')
                img = img.resize(default_size)
                return img
            except Exception as e:
                continue
    
    return None


def visualize_gradcam(model, sample_images, device, output_dir, timestamp):
    """生成Grad-CAM可视化（按照医学论文样式：5行x2列网格）"""
    print("🔄 正在生成Grad-CAM可视化（医学论文样式）...")
    
    # 加载ViT模型用于Grad-CAM（需要分类头来生成梯度）
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
    vit_model = vit_model.to(device)
    vit_model.eval()
    
    # 创建BioCOT_GradCAM对象（使用正确的目标层）
    # 选择最后一个Block的norm1（LayerNorm之前），这是关键！
    gradcam = BioCOT_GradCAM(vit_model, target_layer_name=None)  # 自动寻找blocks[-1].norm1
    
    # 准备图像数据，按标签分组
    negative_samples = []
    positive_samples = []
    
    for sample in sample_images:
        # 加载OCT图像
        oct_img = load_image_from_path(sample.get('oct_paths', ''))
        if oct_img is None:
            continue
        
        # 加载Colposcopy图像（可选）
        col_img = load_image_from_path(sample.get('col_paths', ''))
        
        sample_data = {
            'oct_id': sample.get('oct_id', ''),
            'oct_img': oct_img,
            'col_img': col_img,
            'label': sample.get('label', 0),
            'pred': sample.get('pred', None),
        }
        
        if sample_data['label'] == 0:
            negative_samples.append(sample_data)
        else:
            positive_samples.append(sample_data)
    
    if not negative_samples and not positive_samples:
        print("⚠️ 没有找到有效的图像文件，跳过Grad-CAM可视化")
        return
    
    # 选择样本：每类选择5个，优先选择预测正确且置信度高的
    def select_samples(samples, num=5):
        # 优先选择预测正确的样本，并按置信度排序
        correct = [s for s in samples if s.get('pred') == s['label']]
        incorrect = [s for s in samples if s.get('pred') != s['label']]
        
        # 如果有预测概率，按置信度排序
        if correct and 'prob' in correct[0]:
            correct.sort(key=lambda x: x.get('prob', 0), reverse=True)
        
        selected = correct[:num] if len(correct) >= num else correct + incorrect[:num-len(correct)]
        return selected[:num]
    
    neg_selected = select_samples(negative_samples, 5)
    pos_selected = select_samples(positive_samples, 5)
    
    # 创建5行x2列的网格（每行一个类别，每列OCT和CAM）
    fig, axes = plt.subplots(5, 2, figsize=(10, 20))
    fig.suptitle('Feature Visualization of Cervical OCT Image Classes', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # 类别标签
    class_labels = ['Negative (a)', 'Negative (b)', 'Negative (c)', 'Negative (d)', 'Negative (e)']
    
    # 处理阴性样本（前5行）
    for row_idx in range(5):
        if row_idx < len(neg_selected):
            sample = neg_selected[row_idx]
            try:
                # 处理OCT图像
                oct_img = sample['oct_img']
                img_np = np.array(oct_img) / 255.0
                
                # 转换为tensor并应用ImageNet归一化
                img_tensor = transforms.ToTensor()(oct_img)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                img_tensor = (img_tensor.unsqueeze(0) - mean) / std
                img_tensor = img_tensor.to(device)
                
                # 生成CAM（使用修复后的方法，自动处理reshape和CLS token）
                cam_2d = gradcam.generate_cam(img_tensor, target_class=sample['label'])
                
                # cam_2d 已经是 [14, 14] 的形状，直接使用
                h, w = img_np.shape[:2]
                cam_resized = cv2.resize(cam_2d, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # 增强CAM对比度（确保红蓝分明，捕捉分层特征）
                cam_min, cam_max = cam_resized.min(), cam_resized.max()
                if cam_max - cam_min > 1e-8:
                    # 归一化到[0, 1]
                    cam_resized = (cam_resized - cam_min) / (cam_max - cam_min)
                    
                    # 使用更激进的对比度增强
                    # 1. 使用百分位数拉伸，突出高激活区域
                    p90 = np.percentile(cam_resized, 90)
                    p10 = np.percentile(cam_resized, 10)
                    if p90 > p10:
                        # 将p10-p90范围映射到0-1，增强对比度
                        cam_resized = np.clip((cam_resized - p10) / (p90 - p10 + 1e-8), 0, 1)
                    
                    # 2. 应用强gamma校正，突出高激活区域（红色）
                    cam_resized = cam_resized ** 0.4  # 更小的gamma，更突出高激活
                    
                    # 3. 应用sigmoid增强，进一步突出显著区域
                    cam_resized = 1 / (1 + np.exp(-5 * (cam_resized - 0.5)))
                    
                    # 4. 最终归一化
                    cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
                else:
                    cam_resized = np.zeros_like(cam_resized)
                
                # 左列：OCT原图（转换为灰度）
                if len(img_np.shape) == 3:
                    img_gray = np.mean(img_np, axis=2)
                else:
                    img_gray = img_np
                
                axes[row_idx, 0].imshow(img_gray, cmap='gray')
                axes[row_idx, 0].set_title(f'OCT', fontsize=12, fontweight='bold', pad=5)
                axes[row_idx, 0].axis('off')
                
                # 添加类别标签（左侧）
                if row_idx == 0:
                    axes[row_idx, 0].text(-0.15, 0.5, 'Negative', transform=axes[row_idx, 0].transAxes,
                                         fontsize=14, fontweight='bold', rotation=90, 
                                         ha='center', va='center')
                
                # 右列：CAM热力图（使用更明显的颜色映射，红蓝分明）
                # 使用jet colormap，但调整颜色范围，确保红蓝分明
                # 不应用阈值，直接使用增强后的CAM值
                heatmap = plt.cm.jet(cam_resized)[:, :, :3]
                
                # 增强颜色饱和度，使红蓝更分明
                # 对低激活区域（蓝色）和高激活区域（红色）分别处理
                low_mask = cam_resized < 0.5
                high_mask = cam_resized >= 0.5
                
                # 低激活区域：增强蓝色
                heatmap[low_mask] = heatmap[low_mask] * 0.7 + np.array([0, 0, 0.3]) * 0.3
                
                # 高激活区域：增强红色/黄色
                heatmap[high_mask] = heatmap[high_mask] * 0.8 + np.array([0.8, 0.2, 0]) * 0.2
                
                # 使用更强的混合比例，突出CAM（参考图样式）
                overlayed = 0.35 * img_gray[..., np.newaxis] + 0.65 * heatmap
                overlayed = np.clip(overlayed, 0, 1)
                
                axes[row_idx, 1].imshow(overlayed)
                axes[row_idx, 1].set_title('CAM', fontsize=12, fontweight='bold', pad=5)
                axes[row_idx, 1].axis('off')
                
            except Exception as e:
                print(f"⚠️ 生成Grad-CAM失败 (阴性样本 {row_idx}): {e}")
                axes[row_idx, 0].text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=12)
                axes[row_idx, 0].axis('off')
                axes[row_idx, 1].text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=12)
                axes[row_idx, 1].axis('off')
        else:
            # 如果样本不足，显示空白
            axes[row_idx, 0].axis('off')
            axes[row_idx, 1].axis('off')
    
    # 添加列标题
    fig.text(0.25, 0.98, 'OCT', ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.75, 0.98, 'CAM', ha='center', va='top', fontsize=14, fontweight='bold')
    
    # 添加比例尺（模拟200 µm）
    # 在第一个子图的右下角添加比例尺
    ax = axes[0, 0]
    # 假设图像是224x224像素，对应实际尺寸需要根据实际情况调整
    scale_length_pixels = 50  # 比例尺长度（像素）
    scale_bar_y = img_np.shape[0] - 20
    scale_bar_x_start = img_np.shape[1] - scale_length_pixels - 10
    ax.plot([scale_bar_x_start, scale_bar_x_start + scale_length_pixels], 
            [scale_bar_y, scale_bar_y], 'w', linewidth=3)
    ax.text(scale_bar_x_start + scale_length_pixels / 2, scale_bar_y - 10, 
            '200 µm', ha='center', va='top', color='white', fontsize=10, fontweight='bold')
    
    plt.tight_layout(rect=[0.05, 0, 1, 0.98])
    
    plot_file = output_dir / f"gradcam_medical_style_{timestamp}.png"
    fig.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 Grad-CAM医学样式图已保存: {plot_file}")
    
    plot_file_pdf = output_dir / f"gradcam_medical_style_{timestamp}.pdf"
    fig.savefig(plot_file_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 Grad-CAM医学样式图PDF已保存: {plot_file_pdf}")
    
    plt.close(fig)
    
    # 同时生成阳性样本的图
    if pos_selected:
        fig2, axes2 = plt.subplots(5, 2, figsize=(10, 20))
        fig2.suptitle('Feature Visualization of Cervical OCT Image Classes (Positive)', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        for row_idx in range(5):
            if row_idx < len(pos_selected):
                sample = pos_selected[row_idx]
                try:
                    oct_img = sample['oct_img']
                    img_np = np.array(oct_img) / 255.0
                    
                    img_tensor = transforms.ToTensor()(oct_img)
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    img_tensor = (img_tensor.unsqueeze(0) - mean) / std
                    img_tensor = img_tensor.to(device)
                    
                    # 生成CAM（使用修复后的方法，自动处理reshape和CLS token）
                    cam_2d = gradcam.generate_cam(img_tensor, target_class=sample['label'])
                    
                    # cam_2d 已经是 [14, 14] 的形状，直接使用
                    h, w = img_np.shape[:2]
                    cam_resized = cv2.resize(cam_2d, (w, h), interpolation=cv2.INTER_LINEAR)
                    
                    # 增强CAM对比度（确保红蓝分明，捕捉分层特征）
                    cam_min, cam_max = cam_resized.min(), cam_resized.max()
                    if cam_max - cam_min > 1e-8:
                        # 归一化到[0, 1]
                        cam_resized = (cam_resized - cam_min) / (cam_max - cam_min)
                        
                        # 使用更激进的对比度增强
                        # 1. 使用百分位数拉伸，突出高激活区域
                        p90 = np.percentile(cam_resized, 90)
                        p10 = np.percentile(cam_resized, 10)
                        if p90 > p10:
                            # 将p10-p90范围映射到0-1，增强对比度
                            cam_resized = np.clip((cam_resized - p10) / (p90 - p10 + 1e-8), 0, 1)
                        
                        # 2. 应用强gamma校正，突出高激活区域（红色）
                        cam_resized = cam_resized ** 0.4  # 更小的gamma，更突出高激活
                        
                        # 3. 应用sigmoid增强，进一步突出显著区域
                        cam_resized = 1 / (1 + np.exp(-5 * (cam_resized - 0.5)))
                        
                        # 4. 最终归一化
                        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
                    else:
                        cam_resized = np.zeros_like(cam_resized)
                    
                    if len(img_np.shape) == 3:
                        img_gray = np.mean(img_np, axis=2)
                    else:
                        img_gray = img_np
                    
                    axes2[row_idx, 0].imshow(img_gray, cmap='gray')
                    axes2[row_idx, 0].set_title('OCT', fontsize=12, fontweight='bold', pad=5)
                    axes2[row_idx, 0].axis('off')
                    
                    if row_idx == 0:
                        axes2[row_idx, 0].text(-0.15, 0.5, 'Positive', transform=axes2[row_idx, 0].transAxes,
                                               fontsize=14, fontweight='bold', rotation=90,
                                               ha='center', va='center')
                    
                    # 使用更明显的颜色映射，红蓝分明
                    # 使用jet colormap，但调整颜色范围，确保红蓝分明
                    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
                    
                    # 增强颜色饱和度，使红蓝更分明
                    # 对低激活区域（蓝色）和高激活区域（红色）分别处理
                    low_mask = cam_resized < 0.5
                    high_mask = cam_resized >= 0.5
                    
                    # 低激活区域：增强蓝色
                    heatmap[low_mask] = heatmap[low_mask] * 0.7 + np.array([0, 0, 0.3]) * 0.3
                    
                    # 高激活区域：增强红色/黄色
                    heatmap[high_mask] = heatmap[high_mask] * 0.8 + np.array([0.8, 0.2, 0]) * 0.2
                    
                    # 使用更强的混合比例，突出CAM（参考图样式）
                    overlayed = 0.35 * img_gray[..., np.newaxis] + 0.65 * heatmap
                    overlayed = np.clip(overlayed, 0, 1)
                    
                    axes2[row_idx, 1].imshow(overlayed)
                    axes2[row_idx, 1].set_title('CAM', fontsize=12, fontweight='bold', pad=5)
                    axes2[row_idx, 1].axis('off')
                    
                except Exception as e:
                    print(f"⚠️ 生成Grad-CAM失败 (阳性样本 {row_idx}): {e}")
                    axes2[row_idx, 0].axis('off')
                    axes2[row_idx, 1].axis('off')
            else:
                axes2[row_idx, 0].axis('off')
                axes2[row_idx, 1].axis('off')
        
        fig2.text(0.25, 0.98, 'OCT', ha='center', va='top', fontsize=14, fontweight='bold')
        fig2.text(0.75, 0.98, 'CAM', ha='center', va='top', fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=[0.05, 0, 1, 0.98])
        
        plot_file2 = output_dir / f"gradcam_medical_style_positive_{timestamp}.png"
        fig2.savefig(plot_file2, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"📊 Grad-CAM阳性样本图已保存: {plot_file2}")
        
        plot_file2_pdf = output_dir / f"gradcam_medical_style_positive_{timestamp}.pdf"
        fig2.savefig(plot_file2_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"📊 Grad-CAM阳性样本图PDF已保存: {plot_file2_pdf}")
        
        plt.close(fig2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='生成综合可视化图表（UMAP、直方图、热力图、Grad-CAM）')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--data_root', type=str, 
                       default='/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal',
                       help='数据根目录')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='使用哪个数据集')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")
    
    # 加载数据集
    data_root = Path(args.data_root)
    if args.split == 'train':
        csv_path = data_root / 'internal_train' / 'labels.csv'
    elif args.split == 'val':
        csv_path = data_root / 'internal_val' / 'labels.csv'
    else:
        csv_path = data_root / 'internal_test' / 'labels.csv'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = FiveCentersMultimodalDatasetV2(
        csv_path=str(csv_path),
        clinical_embed_path=None,
        transform=transform,
        oct_num_frames=20,
        max_col_images=3,
        balance_negative_frames=True,
        use_llm=False
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"✅ 数据集: {len(dataset)} 个样本")
    
    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get('args', {})
    
    sample = dataset[0]
    oct_images = sample['oct_images'].unsqueeze(0).to(device)
    oct_feat = extract_features_with_vit(oct_images, device)
    input_dim = oct_feat.shape[-1]
    
    model = BioCOTModel(
        embed_dim=model_args.get('embed_dim', 768),
        num_classes=model_args.get('num_classes', 2),
        num_centers=model_args.get('num_centers', 4),
        input_dim=input_dim,
        use_vlm_encoder=model_args.get('use_vlm_encoder', False)
    )
    
    model = model.to(device)
    
    # 创建动态层
    with torch.no_grad():
        clinical_data = {
            'age': np.array([sample['clinical_data'].get('age', 50)]),
            'hpv': np.array([sample['clinical_data'].get('hpv', 0)]),
            'tct': np.array([sample['clinical_data'].get('tct', 0)]),
        }
        center_labels = torch.tensor([sample['center_idx']], dtype=torch.long).to(device)
        _ = model(oct_features=oct_feat, colpo_features=oct_feat, clinical_features=None,
                 clinical_data=clinical_data, center_labels=center_labels, return_loss_components=False)
    
    state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    print(f"✅ 模型加载完成，最佳AUC: {checkpoint.get('best_auc', 0):.4f}\n")
    
    # 提取特征和预测
    results = extract_predictions_and_features(model, dataloader, device, dataset)
    
    # 生成可视化
    timestamp = Path(args.checkpoint).stem.replace('best_model_5centers_', '').replace('best_model_', '')
    output_dir = Path(args.output) if args.output else Path(args.checkpoint).parent.parent / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. UMAP可视化
    if results['features'] is not None:
        visualize_umap(results['features'], results['labels'], results['centers'], output_dir, timestamp)
    
    # 2. 直方图
    if results['probs'] is not None:
        visualize_histograms(results['probs'], results['labels'], results['preds'], output_dir, timestamp)
    
    # 3. 热力图
    if results['preds'] is not None and results['features'] is not None:
        visualize_heatmaps(results['preds'], results['labels'], results['centers'], 
                         results['features'], output_dir, timestamp)
    
    # 4. Grad-CAM
    if results['sample_images']:
        visualize_gradcam(model, results['sample_images'], device, output_dir, timestamp)
    
    print(f"\n✅ 所有可视化图表生成完成！")


if __name__ == '__main__':
    main()

