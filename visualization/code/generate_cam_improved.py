#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2: 改进的CAM激活图生成器
关键改进：
1. 直接从输入特征计算梯度（更准确的空间映射）
2. 结合模型输出的attention maps（visual notes的attention）
3. 使用Grad-CAM++算法（更准确的权重计算）
4. 分别处理OCT和Colposcopy特征
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# 设置字体
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BioCOT_v3_2_Config
from models.bio_cot_v3_2 import create_bio_cot_v3_2
from data.dataset_v3_2 import FiveCentersMultimodalDatasetV3_2
from torch.utils.data import DataLoader

# 导入特征提取函数
local_training_path = Path(__file__).resolve().parent.parent.parent / 'training'
sys.path.insert(0, str(local_training_path))
from extract_vit_patches import extract_patch_features_with_vit

# 导入Nature配色
from nature_colors import NATURE_COLORS


class ImprovedGradCAM:
    """
    改进的Grad-CAM实现
    关键改进：
    1. 直接从输入特征计算梯度（f_oct, f_colpo）
    2. 结合attention maps进行加权
    3. 使用Grad-CAM++算法
    """
    
    def __init__(self, model):
        self.model = model
        self.gradients_oct = None
        self.gradients_colpo = None
        self.activations_oct = None
        self.activations_colpo = None
        
        # 注册hooks到输入特征
        self.hooks = []
    
    def _register_hooks(self, oct_features, colpo_features):
        """注册hooks到输入特征"""
        def make_hook_oct(tensor):
            def hook(grad):
                self.gradients_oct = grad.detach()
            return hook
        
        def make_hook_colpo(tensor):
            def hook(grad):
                self.gradients_colpo = grad.detach()
            return hook
        
        # 注册梯度hook
        if oct_features.requires_grad:
            oct_features.register_hook(make_hook_oct(oct_features))
        if colpo_features.requires_grad:
            colpo_features.register_hook(make_hook_colpo(colpo_features))
        
        # 保存激活
        self.activations_oct = oct_features.detach()
        self.activations_colpo = colpo_features.detach()
    
    def generate_cam_gradcam_plusplus(self, gradients, activations):
        """
        使用Grad-CAM++算法计算CAM
        
        Args:
            gradients: [B, N, D] 梯度
            activations: [B, N, D] 激活
        
        Returns:
            cam: [N] CAM热图
        """
        B, N, D = activations.shape
        
        # Grad-CAM++权重计算
        # alpha_k^c = sum_i sum_j (w_k^ij * ReLU(grad_k^ij))
        # 其中w_k^ij = exp(grad_k^ij) / sum_i sum_j exp(grad_k^ij)
        
        # 1. 计算每个patch的梯度重要性
        # 对每个特征维度求梯度的ReLU
        grad_relu = F.relu(gradients)  # [B, N, D]
        
        # 2. 计算权重（使用softmax归一化）
        # 对每个patch，计算所有特征维度的梯度重要性
        grad_importance = grad_relu.sum(dim=2, keepdim=True)  # [B, N, 1]
        
        # 3. 使用softmax归一化权重（Grad-CAM++的关键）
        # 这确保重要区域获得更高的权重
        weights = F.softmax(grad_importance.squeeze(-1), dim=1)  # [B, N]
        
        # 4. 计算加权激活
        # 对每个patch，计算激活的加权和
        activations_weighted = activations * weights.unsqueeze(-1)  # [B, N, D]
        
        # 5. 对特征维度求和，得到每个patch的重要性
        cam = activations_weighted.sum(dim=2)  # [B, N]
        if cam.size(0) == 1:
            cam = cam.squeeze(0)  # [N]
        
        # 6. ReLU + 归一化
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def generate_cam_with_attention(self, gradients, activations, attention_map):
        """
        结合attention map的CAM计算
        
        Args:
            gradients: [B, N, D] 梯度
            activations: [B, N, D] 激活
            attention_map: [B, N, 1] 或 [B, N] attention map
        
        Returns:
            cam: [N] CAM热图
        """
        B, N, D = activations.shape
        
        # 处理attention map维度
        if attention_map.dim() == 3:
            # [B, N, 1] -> [B, N]
            attention_map = attention_map.squeeze(-1)
        elif attention_map.dim() == 2:
            if attention_map.size(0) == 1:
                # [1, N] -> [N]
                attention_map = attention_map.squeeze(0)
            elif attention_map.size(0) == B:
                # [B, N] -> 取第一个batch
                attention_map = attention_map[0]
        
        # 确保attention_map是1D的
        if attention_map.dim() > 1:
            # 如果是2D矩阵，取对角线或平均值
            if attention_map.size(0) == attention_map.size(1):
                # 可能是attention矩阵，取对角线
                attention_map = attention_map.diag()
            else:
                # 取平均值
                attention_map = attention_map.mean(dim=0)
        
        # 确保维度匹配
        if attention_map.size(0) != N:
            # 如果维度不匹配，进行插值或截断
            if attention_map.size(0) > N:
                attention_map = attention_map[:N]
            else:
                # 填充
                padding = torch.zeros(N - attention_map.size(0), device=attention_map.device)
                attention_map = torch.cat([attention_map, padding])
        
        # 1. 计算Grad-CAM权重
        grad_relu = F.relu(gradients)
        grad_importance = grad_relu.sum(dim=2, keepdim=True)  # [B, N, 1]
        weights = F.softmax(grad_importance.squeeze(-1), dim=1)  # [B, N]
        
        # 2. 结合attention map
        # attention map已经指示了模型关注的位置
        # 将Grad-CAM权重与attention map相乘，突出两者都认为重要的区域
        attention_map_expanded = attention_map.unsqueeze(0).expand_as(weights)  # [B, N]
        combined_weights = weights * attention_map_expanded
        combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-8)  # 归一化
        
        # 3. 计算加权激活
        activations_weighted = activations * combined_weights.unsqueeze(-1)
        cam = activations_weighted.sum(dim=2)  # [B, N]
        if cam.size(0) == 1:
            cam = cam.squeeze(0)  # [N]
        
        # 4. ReLU + 归一化
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def generate_cam(self, oct_features, colpo_features, image_names, clinical_features, target_class=None):
        """
        生成CAM热图
        
        Args:
            oct_features: [B, N, D] OCT特征（需要requires_grad=True）
            colpo_features: [B, N, D] Colposcopy特征（需要requires_grad=True）
            image_names: 图像文件名列表
            clinical_features: [B, 7] 临床特征
            target_class: 目标类别
        
        Returns:
            oct_cam: [N] OCT的CAM热图
            colpo_cam: [N] Colposcopy的CAM热图
        """
        self.model.eval()
        
        # 确保特征需要梯度
        oct_features = oct_features.clone().detach().requires_grad_(True)
        colpo_features = colpo_features.clone().detach().requires_grad_(True)
        
        # 注册hooks
        self._register_hooks(oct_features, colpo_features)
        
        # 前向传播
        output = self.model(
            f_oct=oct_features,
            f_colpo=colpo_features,
            image_names=image_names,
            clinical_features=clinical_features
        )
        
        logits = output['logits']
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # 获取attention maps（如果可用）
        attn_oct = None
        attn_colpo = None
        if 'attn_maps' in output and output['attn_maps'] is not None:
            attn_maps = output['attn_maps']
            if len(attn_maps) >= 2:
                attn_oct = attn_maps[0]  # [B, N, 1]
                attn_colpo = attn_maps[1]  # [B, N, 1]
        
        # 反向传播
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        # 检查梯度是否捕获
        if self.gradients_oct is None or self.gradients_colpo is None:
            print("⚠️ 未捕获梯度，使用零CAM")
            N = oct_features.size(1)
            return np.zeros(N), np.zeros(N)
        
        # 生成CAM
        # 方案1：如果有attention map，结合使用
        if attn_oct is not None and attn_colpo is not None:
            oct_cam = self.generate_cam_with_attention(
                self.gradients_oct, 
                self.activations_oct,
                attn_oct[0]  # 取第一个batch
            )
            colpo_cam = self.generate_cam_with_attention(
                self.gradients_colpo,
                self.activations_colpo,
                attn_colpo[0]  # 取第一个batch
            )
        else:
            # 方案2：仅使用Grad-CAM++
            oct_cam = self.generate_cam_gradcam_plusplus(
                self.gradients_oct,
                self.activations_oct
            )
            colpo_cam = self.generate_cam_gradcam_plusplus(
                self.gradients_colpo,
                self.activations_colpo
            )
        
        return oct_cam, colpo_cam


def overlay_cam_on_image(image, cam, alpha=0.5, threshold=0.3):
    """
    将CAM热图叠加到图像上（改进版：只显示高激活区域）
    
    Args:
        image: [H, W, 3] RGB图像，范围[0, 255]
        cam: [H, W] 热图，范围[0, 1]
        alpha: 叠加透明度
        threshold: 只显示高于此阈值的区域
    
    Returns:
        result: [H, W, 3] 叠加结果
    """
    # 确保image是3通道
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # 应用阈值：只显示高激活区域
    cam_thresholded = np.where(cam_resized > threshold, cam_resized, 0)
    
    # 归一化到[0, 1]
    if cam_thresholded.max() > 0:
        cam_thresholded = cam_thresholded / cam_thresholded.max()
    
    # 应用颜色映射（使用Nature风格的低饱和度配色）
    heatmap = plt.cm.RdYlBu_r(cam_thresholded)[:, :, :3]  # 去掉alpha通道
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 叠加（只在高激活区域叠加）
    mask = cam_thresholded > 0
    result = image.copy().astype(float)
    result[mask] = (1 - alpha) * result[mask] + alpha * heatmap[mask]
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def tensor_to_image(tensor):
    """将Tensor转换为可显示的图像"""
    if tensor.dim() == 5:  # [B, F, C, H, W]
        tensor = tensor[0, 0]  # 取第一个batch，第一帧
    elif tensor.dim() == 4:
        tensor = tensor[0]  # [C, H, W]
    
    if tensor.shape[0] == 1:
        # 单通道
        img = tensor[0].cpu().numpy()
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        # 多通道
        img = tensor[:3].permute(1, 2, 0).cpu().numpy()
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def generate_cam_visualizations(
    model,
    dataloader,
    device,
    save_dir,
    num_samples=8
):
    """
    生成改进的CAM可视化
    """
    print("\n🎨 生成改进的CAM激活图可视化（使用Grad-CAM++和Attention）...")
    
    model.to(device)
    model.eval()
    
    # 创建GradCAM对象
    grad_cam = ImprovedGradCAM(model)
    
    # 收集样本
    print(f"   📊 收集 {num_samples} 个样本...")
    samples = []
    
    for batch in dataloader:
        if len(samples) >= num_samples:
            break
        
        oct_images = batch['oct_images'].to(device)
        colposcopy_images = batch['colposcopy_images'].to(device)
        clinical_features = batch['clinical_features'].to(device)
        labels = batch['label'].to(device)
        image_names = batch.get('image_names', [f'image_{i}' for i in range(oct_images.size(0))])
        
        batch_size = oct_images.size(0)
        
        # 提取特征
        B_oct = oct_images.shape[0]
        if len(oct_images.shape) == 5:  # [B, F, C, H, W]
            F_oct = oct_images.shape[1]
            oct_images_flat = oct_images.view(B_oct * F_oct, *oct_images.shape[2:])
            oct_features_patch = extract_patch_features_with_vit(oct_images_flat, device, batch_size=4)
            oct_features_patch = oct_features_patch.view(B_oct, F_oct, 196, 768).mean(dim=1)  # [B, 196, 768]
        else:
            oct_features_patch = extract_patch_features_with_vit(oct_images, device, batch_size=4)
        
        B_colpo = colposcopy_images.shape[0]
        if len(colposcopy_images.shape) == 5:  # [B, N, C, H, W]
            N_colpo = colposcopy_images.shape[1]
            colpo_images_flat = colposcopy_images.view(B_colpo * N_colpo, *colposcopy_images.shape[2:])
            colpo_features_patch = extract_patch_features_with_vit(colpo_images_flat, device, batch_size=4)
            colpo_features_patch = colpo_features_patch.view(B_colpo, N_colpo, 196, 768).mean(dim=1)  # [B, 196, 768]
        else:
            colpo_features_patch = extract_patch_features_with_vit(colposcopy_images, device, batch_size=4)
        
        for i in range(batch_size):
            if len(samples) >= num_samples:
                break
            
            samples.append({
                'oct_images': oct_images[i:i+1],
                'colpo_images': colposcopy_images[i:i+1],
                'oct_features': oct_features_patch[i:i+1],
                'colpo_features': colpo_features_patch[i:i+1],
                'clinical': clinical_features[i:i+1],
                'image_names': [image_names[i]],
                'label': labels[i].item()
            })
    
    print(f"   ✅ 实际收集 {len(samples)} 个样本")
    
    # 创建大图
    fig = plt.figure(figsize=(24, 5 * len(samples)))
    gs = GridSpec(len(samples), 6, figure=fig, hspace=0.4, wspace=0.3)
    
    for idx, sample in enumerate(tqdm(samples, desc="Generating CAM")):
        label_name = "Positive" if sample['label'] == 1 else "Negative"
        
        # 1. 原始OCT图像（取第一帧）
        oct_img = tensor_to_image(sample['oct_images'])
        
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(oct_img)
        ax1.set_title(f'Sample {idx+1}\nOCT (Frame 1)\nLabel: {label_name}',
                     fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        # 2. OCT CAM
        try:
            oct_cam, _ = grad_cam.generate_cam(
                sample['oct_features'],
                sample['colpo_features'],
                sample['image_names'],
                sample['clinical'],
                target_class=sample['label']
            )
            
            # Reshape到空间维度
            oct_cam_flat = oct_cam.flatten()
            n_patches = len(oct_cam_flat)
            side = int(np.sqrt(n_patches))
            if side * side == n_patches:
                oct_cam_2d = oct_cam_flat.reshape(side, side)
            else:
                # 如果不是完全平方数，取最接近的
                oct_cam_2d = oct_cam_flat[:side*side].reshape(side, side)
            
            oct_overlay = overlay_cam_on_image(oct_img, oct_cam_2d, alpha=0.6, threshold=0.2)
        except Exception as e:
            print(f"   ⚠️ Sample {idx} OCT CAM失败: {e}")
            import traceback
            traceback.print_exc()
            oct_overlay = oct_img
            oct_cam_2d = np.zeros((14, 14))
        
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.imshow(oct_overlay)
        ax2.set_title('OCT + CAM (Improved)', fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        # 3. OCT纯热图
        ax3 = fig.add_subplot(gs[idx, 2])
        im = ax3.imshow(oct_cam_2d, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax3.set_title('OCT CAM Heatmap', fontsize=11, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. 原始Colposcopy图像（取第一张）
        colpo_img = tensor_to_image(sample['colpo_images'])
        
        ax4 = fig.add_subplot(gs[idx, 3])
        ax4.imshow(colpo_img)
        ax4.set_title('Colposcopy (Image 1)', fontsize=11, fontweight='bold')
        ax4.axis('off')
        
        # 5. Colposcopy CAM
        try:
            _, colpo_cam = grad_cam.generate_cam(
                sample['oct_features'],
                sample['colpo_features'],
                sample['image_names'],
                sample['clinical'],
                target_class=sample['label']
            )
            
            # Reshape到空间维度
            colpo_cam_flat = colpo_cam.flatten()
            n_patches = len(colpo_cam_flat)
            side = int(np.sqrt(n_patches))
            if side * side == n_patches:
                colpo_cam_2d = colpo_cam_flat.reshape(side, side)
            else:
                # 如果不是完全平方数，取最接近的
                colpo_cam_2d = colpo_cam_flat[:side*side].reshape(side, side)
            
            colpo_overlay = overlay_cam_on_image(colpo_img, colpo_cam_2d, alpha=0.6, threshold=0.2)
        except Exception as e:
            print(f"   ⚠️ Sample {idx} Colpo CAM失败: {e}")
            colpo_overlay = colpo_img
            colpo_cam_2d = np.zeros((14, 14))
        
        ax5 = fig.add_subplot(gs[idx, 4])
        ax5.imshow(colpo_overlay)
        ax5.set_title('Colpo + CAM (Improved)', fontsize=11, fontweight='bold')
        ax5.axis('off')
        
        # 6. Colposcopy纯热图
        ax6 = fig.add_subplot(gs[idx, 5])
        im = ax6.imshow(colpo_cam_2d, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax6.set_title('Colpo CAM Heatmap', fontsize=11, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    
    plt.suptitle('Improved CAM Activation Analysis: Bio-COT 3.2 (Grad-CAM++ + Attention)',
                fontsize=18, fontweight='bold', y=0.995)
    
    # 保存
    save_path = Path(save_dir) / 'CAM_Activation_Analysis_Improved.pdf'
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    save_path_png = Path(save_dir) / 'CAM_Activation_Analysis_Improved.png'
    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    
    print(f"✅ Improved CAM activation visualization saved: {save_path}")
    
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("Bio-COT 3.2: Improved CAM Activation Visualization Generator")
    print("Key Improvements: Grad-CAM++ + Attention Maps + Direct Feature Gradients")
    print("=" * 80)
    
    # 配置
    config = BioCOT_v3_2_Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print("\n📦 加载模型...")
    model = create_bio_cot_v3_2(config)
    
    # 加载checkpoint
    checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pth'
    if checkpoint_path.exists():
        print(f"   📥 加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("   ✅ Checkpoint加载成功")
    else:
        print(f"   ⚠️ 未找到checkpoint: {checkpoint_path}")
        print("   ⚠️ 使用未训练的模型（CAM可能不准确）")
    
    model.to(device)
    model.eval()
    
    # 加载数据集
    print("\n📂 加载数据集...")
    dataset = FiveCentersMultimodalDatasetV3_2(
        csv_path=Path(config.data_root) / 'temp_val_labels.csv',
        data_root=config.data_root
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 生成CAM可视化
    save_dir = Path(__file__).parent.parent / 'figures'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    generate_cam_visualizations(model, dataloader, device, save_dir, num_samples=8)
    
    print("\n" + "=" * 80)
    print("✅ Improved CAM生成完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

