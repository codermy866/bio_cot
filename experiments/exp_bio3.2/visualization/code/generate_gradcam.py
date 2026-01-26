#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2: Grad-CAM Visualization Generator
生成Grad-CAM热图，需要加载训练好的模型
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

# 设置字体
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

# 导入模型和数据集
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BioCOT_v3_2_Config
from models.bio_cot_v3_2 import BioCOT_v3_2
from data.dataset_v3_2 import FiveCentersMultimodalDatasetV3_2
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class GradCAMpp:
    """
    Grad-CAM++实现（更精确的注意力可视化）
    论文: Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks
    """
    
    def __init__(self, model, target_layers):
        """
        Args:
            model: 训练好的模型
            target_layers: 目标层列表（通常是最后几层卷积）
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.activations = []
        self.hooks = []
        
        # 注册hook
        for layer in target_layers:
            self.hooks.append(
                layer.register_forward_hook(self._save_activation)
            )
            self.hooks.append(
                layer.register_full_backward_hook(self._save_gradient)
            )
    
    def _save_activation(self, module, input, output):
        self.activations.append(output.detach())
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())
    
    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, input_dict, target_class=None, layer_idx=-1):
        """
        生成Grad-CAM++热图
        
        Args:
            input_dict: 模型输入字典
            target_class: 目标类别
            layer_idx: 使用哪个层的激活（-1表示最后一层）
        
        Returns:
            cam: [H, W] 归一化的热图
        """
        self.model.eval()
        self.gradients = []
        self.activations = []
        
        # 前向传播
        output = self.model(**input_dict)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        # 获取梯度和激活
        if not self.gradients or not self.activations:
            print("⚠️ 未捕获到梯度或激活，检查target_layers")
            return np.zeros((14, 14))
        
        gradients = self.gradients[layer_idx]  # [1, C, H, W]
        activations = self.activations[layer_idx]  # [1, C, H, W]
        
        # Grad-CAM++权重计算
        # alpha = ReLU(gradient) / sum(ReLU(gradient))
        grad_2 = gradients ** 2
        grad_3 = grad_2 * gradients
        
        sum_activations = activations.sum(dim=[2, 3], keepdim=True)
        alpha = grad_2 / (2 * grad_2 + sum_activations * grad_3 + 1e-8)
        alpha = F.relu(gradients) * alpha
        alpha = alpha.sum(dim=1, keepdim=True)  # [1, 1, H, W]
        
        # 加权求和
        weights = alpha.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * activations).sum(dim=1).squeeze(0)  # [H, W]
        
        # ReLU + 归一化
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()


def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    将热图叠加到原图
    
    Args:
        image: [H, W, 3] 或 [H, W] 原图
        heatmap: [H, W] 热图
        alpha: 叠加透明度
        colormap: OpenCV颜色映射
    
    Returns:
        result: [H, W, 3] RGB叠加结果
    """
    # 确保image是3通道
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize heatmap
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # 应用颜色映射
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # 叠加
    result = (1 - alpha) * image.astype(float) + alpha * heatmap_colored.astype(float)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def tensor_to_image(tensor):
    """
    将Tensor转换为可显示的图像
    
    Args:
        tensor: [C, H, W] 或 [H, W]
    
    Returns:
        image: [H, W, 3] RGB图像，范围[0, 255]
    """
    if tensor.dim() == 3:
        # [C, H, W]
        if tensor.shape[0] == 1:
            # 单通道，转为灰度
            img = tensor[0].cpu().numpy()
        else:
            # 多通道，取前3个或转置
            img = tensor[:3].permute(1, 2, 0).cpu().numpy()
    else:
        # [H, W]
        img = tensor.cpu().numpy()
    
    # 归一化到[0, 255]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    img = img.astype(np.uint8)
    
    # 确保是3通道
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    return img


def generate_gradcam_visualizations(
    model, 
    dataloader, 
    device, 
    save_dir, 
    num_samples=8
):
    """
    生成Grad-CAM可视化
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        save_dir: 保存目录
        num_samples: 可视化样本数
    """
    print("\n🎨 生成Grad-CAM可视化...")
    
    model.to(device)
    model.eval()
    
    # 选择目标层
    # 对于ViT，选择最后几个Transformer block
    try:
        if hasattr(model, 'visual_encoder'):
            if hasattr(model.visual_encoder, 'blocks'):
                target_layers = [model.visual_encoder.blocks[-1]]
                print(f"   使用target_layer: visual_encoder.blocks[-1]")
            else:
                print("⚠️ 无法找到visual_encoder.blocks，使用默认层")
                target_layers = [model.visual_encoder]
        else:
            print("⚠️ 模型无visual_encoder属性")
            return
    except Exception as e:
        print(f"⚠️ 无法定位目标层: {e}")
        return
    
    # 创建Grad-CAM对象
    grad_cam = GradCAMpp(model, target_layers)
    
    # 收集样本
    print(f"   收集 {num_samples} 个样本...")
    samples = []
    
    for batch in dataloader:
        if len(samples) >= num_samples:
            break
        
        oct_images = batch['oct_images'].to(device)
        colpo_images = batch['colposcopy_images'].to(device)
        clinical = batch['clinical'].to(device)
        labels = batch['label'].to(device)
        
        batch_size = oct_images.size(0)
        
        for i in range(batch_size):
            if len(samples) >= num_samples:
                break
            
            samples.append({
                'oct': oct_images[i:i+1],
                'colpo': colpo_images[i:i+1],
                'clinical': clinical[i:i+1],
                'label': labels[i].item()
            })
    
    print(f"   实际收集 {len(samples)} 个样本")
    
    # 绘图
    fig = plt.figure(figsize=(20, 4 * len(samples)))
    gs = GridSpec(len(samples), 5, figure=fig, hspace=0.4, wspace=0.3)
    
    for idx, sample in enumerate(tqdm(samples, desc="Generating CAMs")):
        label_name = "Positive" if sample['label'] == 1 else "Negative"
        
        # 准备输入
        input_dict = {
            'oct_images': sample['oct'],
            'colposcopy_images': sample['colpo'],
            'clinical': sample['clinical']
        }
        
        # 1. 原始OCT图像（取第一帧）
        oct_img = tensor_to_image(sample['oct'][0, 0])  # [C, H, W] -> [H, W, 3]
        
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(oct_img)
        ax1.set_title(f'Sample {idx+1}\nOCT (Frame 1)\nLabel: {label_name}',
                     fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        # 2. OCT Grad-CAM
        try:
            oct_cam = grad_cam.generate_cam(input_dict, target_class=sample['label'])
            oct_overlay = overlay_heatmap(oct_img, oct_cam, alpha=0.5)
        except Exception as e:
            print(f"   ⚠️ Sample {idx} OCT CAM失败: {e}")
            oct_overlay = oct_img
        
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.imshow(oct_overlay)
        ax2.set_title('OCT Grad-CAM++', fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        # 3. 原始Colposcopy图像（取第一张）
        colpo_img = tensor_to_image(sample['colpo'][0, 0])  # [C, H, W] -> [H, W, 3]
        
        ax3 = fig.add_subplot(gs[idx, 2])
        ax3.imshow(colpo_img)
        ax3.set_title('Colposcopy (Image 1)', fontsize=11, fontweight='bold')
        ax3.axis('off')
        
        # 4. Colposcopy Grad-CAM
        try:
            colpo_cam = grad_cam.generate_cam(input_dict, target_class=sample['label'])
            colpo_overlay = overlay_heatmap(colpo_img, colpo_cam, alpha=0.5)
        except Exception as e:
            print(f"   ⚠️ Sample {idx} Colpo CAM失败: {e}")
            colpo_overlay = colpo_img
        
        ax4 = fig.add_subplot(gs[idx, 3])
        ax4.imshow(colpo_overlay)
        ax4.set_title('Colpo Grad-CAM++', fontsize=11, fontweight='bold')
        ax4.axis('off')
        
        # 5. 热图（单独显示）
        ax5 = fig.add_subplot(gs[idx, 4])
        im = ax5.imshow(oct_cam, cmap='jet')
        ax5.set_title('Attention Heatmap', fontsize=11, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    
    plt.suptitle('Grad-CAM++ Visualization: Bio-COT 3.2 Attention Analysis',
                fontsize=18, fontweight='bold', y=0.995)
    
    # 保存
    save_path = Path(save_dir) / 'GradCAM_Analysis.pdf'
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    save_path_png = Path(save_dir) / 'GradCAM_Analysis.png'
    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    
    print(f"✅ Grad-CAM visualization saved: {save_path}")
    
    # 清理
    grad_cam.remove_hooks()
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("Bio-COT 3.2: Grad-CAM Visualization Generator")
    print("=" * 80)
    
    # 设置路径
    exp_dir = Path(__file__).resolve().parents[2]
    vis_dir = exp_dir / 'visualization'
    figures_dir = vis_dir / 'figures'
    
    # 加载配置
    config = BioCOT_v3_2_Config()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 使用设备: {device}")
    
    # 加载模型
    print("\n📥 加载训练好的模型...")
    checkpoint_dir = exp_dir / 'checkpoints'
    checkpoint_files = list(checkpoint_dir.glob('best_model_*.pth'))
    
    if not checkpoint_files:
        print("❌ 未找到训练好的模型checkpoint")
        print(f"   请确保 {checkpoint_dir} 目录下有 best_model_*.pth 文件")
        return
    
    # 选择最新的checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    print(f"   加载checkpoint: {latest_checkpoint.name}")
    
    # 创建模型
    model = BioCOT_v3_2(
        embed_dim=config.embed_dim,
        num_classes=config.num_classes,
        num_centers=config.num_centers,
        vlm_json_path=str(config.vlm_json_path),
        text_model_name=config.text_model_name,
        use_visual_notes=config.use_visual_notes,
        use_adaptive_gating=config.use_adaptive_gating,
    )
    
    # 加载权重
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("✅ 模型加载成功")
    
    # 创建数据加载器
    print("\n📥 加载验证集数据...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_csv = Path(config.data_root) / 'val_labels.csv'
    val_dataset = FiveCentersMultimodalDatasetV3_2(
        csv_path=str(val_csv),
        data_root=str(config.data_root),
        transform=transform,
        oct_num_frames=config.oct_frames,
        max_col_images=config.colposcopy_images,
        balance_negative_frames=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"✅ 验证集样本数: {len(val_dataset)}")
    
    # 生成Grad-CAM
    generate_gradcam_visualizations(
        model=model,
        dataloader=val_loader,
        device=device,
        save_dir=figures_dir,
        num_samples=8
    )
    
    print("\n" + "=" * 80)
    print("✅ Grad-CAM可视化完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

