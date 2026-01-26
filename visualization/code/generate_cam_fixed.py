#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2: 改进的CAM激活图生成器
修复了模型调用和梯度提取问题
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
    改进的Grad-CAM实现，支持Bio-COT 3.2
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = []
        self.activations = []
        self.hooks = []
        
        # 注册hooks
        self.hooks.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self.hooks.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )
    
    def _save_activation(self, module, input, output):
        """保存前向传播的激活"""
        # 处理tuple输出（visual_notes_module返回(feat, attn_map)）
        if isinstance(output, tuple):
            output = output[0]  # 取第一个元素（特征）
        self.activations.append(output.detach())
    
    def _save_gradient(self, module, grad_input, grad_output):
        """保存反向传播的梯度"""
        if grad_output is not None and len(grad_output) > 0:
            grad = grad_output[0]
            # 处理tuple梯度
            if isinstance(grad, tuple):
                grad = grad[0]
            if grad is not None:
                self.gradients.append(grad.detach())
    
    def remove_hooks(self):
        """移除hooks"""
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, oct_features, colpo_features, image_names, clinical_features, target_class=None):
        """
        生成CAM热图
        
        Args:
            oct_features: [B, N, D] OCT特征
            colpo_features: [B, N, D] Colposcopy特征
            image_names: 图像文件名列表
            clinical_features: [B, 7] 临床特征
            target_class: 目标类别（None则使用预测类别）
        
        Returns:
            oct_cam: [H, W] OCT的CAM热图
            colpo_cam: [H, W] Colposcopy的CAM热图
        """
        self.model.eval()
        self.gradients = []
        self.activations = []
        
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
        
        # 反向传播
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        if not self.gradients or not self.activations:
            print("⚠️ 未捕获梯度或激活")
            return np.zeros((14, 14)), np.zeros((14, 14))
        
        # 使用最后一层
        gradients = self.gradients[-1]  # [B, N, D]
        activations = self.activations[-1]  # [B, N, D]
        
        # Grad-CAM权重计算
        # 对每个patch计算权重
        weights = gradients.mean(dim=2, keepdim=True)  # [B, N, 1]
        
        # 加权求和
        cam = (weights * activations).sum(dim=2).squeeze(0)  # [N]
        
        # ReLU + 归一化
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # 由于OCT和Colposcopy使用相同的特征提取路径，我们返回相同的CAM
        # 实际应用中，可以分别处理
        cam_np = cam.cpu().numpy()
        
        # Reshape到空间维度 (假设是14x14的patch grid)
        if len(cam_np) == 196:  # 14*14
            cam_2d = cam_np.reshape(14, 14)
        else:
            # 如果不是196，尝试找到最接近的平方数
            n_patches = len(cam_np)
            side = int(np.sqrt(n_patches))
            cam_2d = cam_np[:side*side].reshape(side, side)
        
        return cam_2d, cam_2d


def overlay_cam_on_image(image, cam, alpha=0.5):
    """
    将CAM热图叠加到图像上
    
    Args:
        image: [H, W, 3] RGB图像，范围[0, 255]
        cam: [H, W] 热图，范围[0, 1]
        alpha: 叠加透明度
    
    Returns:
        result: [H, W, 3] 叠加结果
    """
    # 确保image是3通道
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # 应用颜色映射（使用Nature风格的低饱和度配色）
    # 使用RdYlBu_r colormap（红-黄-蓝，反转）
    heatmap = plt.cm.RdYlBu_r(cam_resized)[:, :, :3]  # 去掉alpha通道
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 叠加
    result = (1 - alpha) * image.astype(float) + alpha * heatmap.astype(float)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def tensor_to_image(tensor):
    """
    将Tensor转换为可显示的图像
    """
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
    print("\n🎨 生成改进的CAM激活图可视化...")
    
    model.to(device)
    model.eval()
    
    # 选择目标层 - 由于输入已经是提取的特征，我们需要从特征处理层获取梯度
    try:
        # 优先使用visual_notes_module（特征处理的第一层）
        if hasattr(model, 'visual_notes_module') and model.visual_notes_module is not None:
            # visual_notes_module内部有多个层，我们选择最后一个处理层
            if hasattr(model.visual_notes_module, 'feat_refine'):
                target_layer = model.visual_notes_module.feat_refine[-1]  # 最后一层
                print(f"   ✅ 使用target_layer: visual_notes_module.feat_refine[-1]")
            elif hasattr(model.visual_notes_module, 'q_proj'):
                target_layer = model.visual_notes_module.q_proj
                print(f"   ✅ 使用target_layer: visual_notes_module.q_proj")
            else:
                target_layer = model.visual_notes_module
                print(f"   ✅ 使用target_layer: visual_notes_module")
        # 备选：使用dual_head
        elif hasattr(model, 'dual_head'):
            if isinstance(model.dual_head, nn.Sequential):
                target_layer = model.dual_head[-1]  # Sequential的最后一层
            else:
                target_layer = model.dual_head
            print(f"   ✅ 使用target_layer: dual_head")
        else:
            print("   ⚠️ 无法定位目标层，跳过CAM生成")
            return
    except Exception as e:
        print(f"   ⚠️ 无法定位目标层: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建GradCAM对象
    grad_cam = ImprovedGradCAM(model, target_layer)
    
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
            oct_overlay = overlay_cam_on_image(oct_img, oct_cam, alpha=0.5)
        except Exception as e:
            print(f"   ⚠️ Sample {idx} OCT CAM失败: {e}")
            oct_overlay = oct_img
            oct_cam = np.zeros((14, 14))
        
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.imshow(oct_overlay)
        ax2.set_title('OCT + CAM', fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        # 3. OCT纯热图
        ax3 = fig.add_subplot(gs[idx, 2])
        im = ax3.imshow(oct_cam, cmap='RdYlBu_r', vmin=0, vmax=1)
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
            colpo_overlay = overlay_cam_on_image(colpo_img, colpo_cam, alpha=0.5)
        except Exception as e:
            print(f"   ⚠️ Sample {idx} Colpo CAM失败: {e}")
            colpo_overlay = colpo_img
            colpo_cam = np.zeros((14, 14))
        
        ax5 = fig.add_subplot(gs[idx, 4])
        ax5.imshow(colpo_overlay)
        ax5.set_title('Colpo + CAM', fontsize=11, fontweight='bold')
        ax5.axis('off')
        
        # 6. Colposcopy纯热图
        ax6 = fig.add_subplot(gs[idx, 5])
        im = ax6.imshow(colpo_cam, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax6.set_title('Colpo CAM Heatmap', fontsize=11, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    
    plt.suptitle('CAM Activation Analysis: Bio-COT 3.2 Attention on OCT and Colposcopy',
                fontsize=18, fontweight='bold', y=0.995)
    
    # 保存
    save_path = Path(save_dir) / 'CAM_Activation_Analysis_Fixed.pdf'
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    save_path_png = Path(save_dir) / 'CAM_Activation_Analysis_Fixed.png'
    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    
    print(f"✅ CAM activation visualization saved: {save_path}")
    
    # 清理
    grad_cam.remove_hooks()
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("Bio-COT 3.2: Improved CAM Activation Visualization Generator")
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
    print("✅ CAM生成完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

