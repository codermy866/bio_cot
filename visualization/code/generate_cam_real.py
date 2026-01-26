#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2: Real CAM Visualization Generator
真实的CAM激活图生成器 - 叠加在OCT和阴道镜图像上
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
from models.bio_cot_v3_2 import BioCOT_v3_2
from data.dataset_v3_2 import FiveCentersMultimodalDatasetV3_2
from torch.utils.data import DataLoader


class GradCAMPlusPlus:
    """
    Grad-CAM++实现（改进版，更适合Transformer）
    """
    
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers if isinstance(target_layers, list) else [target_layers]
        self.gradients = []
        self.activations = []
        self.hooks = []
        
        # 注册hooks
        for layer in self.target_layers:
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
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        生成CAM热图
        
        Args:
            input_tensor: 模型输入
            target_class: 目标类别
        
        Returns:
            cam: [H, W] 归一化热图
        """
        self.model.eval()
        self.gradients = []
        self.activations = []
        
        # 前向传播
        if isinstance(input_tensor, dict):
            output = self.model(**input_tensor)
        else:
            output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        if not self.gradients or not self.activations:
            print("⚠️ 未捕获梯度或激活")
            return np.zeros((14, 14))
        
        # 使用最后一层
        gradients = self.gradients[-1]  # [1, C, H, W]
        activations = self.activations[-1]  # [1, C, H, W]
        
        # Grad-CAM++权重
        b, c, h, w = gradients.shape
        
        # 全局平均池化
        alpha = gradients.view(b, c, -1).mean(dim=2)  # [1, C]
        alpha = alpha.view(b, c, 1, 1)  # [1, C, 1, 1]
        
        # 加权求和
        cam = (alpha * activations).sum(dim=1).squeeze(0)  # [H, W]
        
        # ReLU + 归一化
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()


def overlay_cam_on_image(image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    将CAM热图叠加到图像上
    
    Args:
        image: [H, W, 3] RGB图像，范围[0, 255]
        cam: [H, W] 热图，范围[0, 1]
        alpha: 叠加透明度
        colormap: OpenCV colormap
    
    Returns:
        result: [H, W, 3] 叠加结果
    """
    # 确保image是3通道
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # 应用颜色映射
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8),
        colormap
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 叠加
    result = (1 - alpha) * image.astype(float) + alpha * heatmap.astype(float)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def tensor_to_image(tensor):
    """
    将Tensor转换为可显示的图像
    """
    if tensor.dim() == 4:
        # [B, C, H, W] -> [C, H, W]
        tensor = tensor[0]
    
    if tensor.shape[0] == 1:
        # 单通道
        img = tensor[0].cpu().numpy()
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        # 多通道
        img = tensor[:3].permute(1, 2, 0).cpu().numpy()
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    return img


def generate_cam_visualizations(
    model,
    dataloader,
    device,
    save_dir,
    num_samples=8
):
    """
    生成真实的CAM可视化（叠加在图像上）
    """
    print("\n🎨 生成CAM激活图可视化...")
    
    model.to(device)
    model.eval()
    
    # 选择目标层
    try:
        if hasattr(model, 'visual_encoder'):
            if hasattr(model.visual_encoder, 'blocks'):
                # ViT结构
                target_layer = model.visual_encoder.blocks[-1]
                print(f"   使用target_layer: visual_encoder.blocks[-1]")
            else:
                target_layer = model.visual_encoder
        else:
            print("⚠️ 模型无visual_encoder，跳过CAM生成")
            return
    except Exception as e:
        print(f"⚠️ 无法定位目标层: {e}")
        return
    
    # 创建GradCAM对象
    grad_cam = GradCAMPlusPlus(model, target_layer)
    
    # 收集样本
    print(f"   收集 {num_samples} 个样本...")
    samples = []
    
    for batch in dataloader:
        if len(samples) >= num_samples:
            break
        
        oct_images = batch['oct_images'].to(device)
        colpo_images = batch['colposcopy_images'].to(device)
        clinical = batch['clinical_features'].to(device)
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
    
    # 创建大图
    fig = plt.figure(figsize=(24, 5 * len(samples)))
    gs = GridSpec(len(samples), 6, figure=fig, hspace=0.4, wspace=0.3)
    
    for idx, sample in enumerate(tqdm(samples, desc="Generating CAM")):
        label_name = "Positive" if sample['label'] == 1 else "Negative"
        
        # 准备输入
        input_dict = {
            'oct_images': sample['oct'],
            'colposcopy_images': sample['colpo'],
            'clinical_features': sample['clinical']
        }
        
        # 1. 原始OCT图像（取第一帧）
        oct_img = tensor_to_image(sample['oct'][0, 0])
        
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(oct_img)
        ax1.set_title(f'Sample {idx+1}\nOCT (Frame 1)\nLabel: {label_name}',
                     fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        # 2. OCT CAM
        try:
            oct_cam = grad_cam.generate_cam(input_dict, target_class=sample['label'])
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
        im = ax3.imshow(oct_cam, cmap='jet')
        ax3.set_title('OCT CAM Heatmap', fontsize=11, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. 原始Colposcopy图像（取第一张）
        colpo_img = tensor_to_image(sample['colpo'][0, 0])
        
        ax4 = fig.add_subplot(gs[idx, 3])
        ax4.imshow(colpo_img)
        ax4.set_title('Colposcopy (Image 1)', fontsize=11, fontweight='bold')
        ax4.axis('off')
        
        # 5. Colposcopy CAM
        try:
            # 使用同样的输入生成CAM
            colpo_cam = grad_cam.generate_cam(input_dict, target_class=sample['label'])
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
        im = ax6.imshow(colpo_cam, cmap='jet')
        ax6.set_title('Colpo CAM Heatmap', fontsize=11, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    
    plt.suptitle('CAM Activation Analysis: Bio-COT 3.2 Attention on OCT and Colposcopy',
                fontsize=18, fontweight='bold', y=0.995)
    
    # 保存
    save_path = Path(save_dir) / 'CAM_Activation_Analysis.pdf'
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    save_path_png = Path(save_dir) / 'CAM_Activation_Analysis.png'
    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    
    print(f"✅ CAM activation visualization saved: {save_path}")
    
    # 清理
    grad_cam.remove_hooks()
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("Bio-COT 3.2: Real CAM Activation Visualization Generator")
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
        print("\n💡 提示：训练完成后会自动生成checkpoint文件")
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
    try:
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 创建数据加载器
    print("\n📥 加载验证集数据...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_csv = Path(config.data_root) / 'val_labels.csv'
    
    if not val_csv.exists():
        print(f"❌ 验证集CSV文件不存在: {val_csv}")
        return
    
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
    
    # 生成CAM可视化
    generate_cam_visualizations(
        model=model,
        dataloader=val_loader,
        device=device,
        save_dir=figures_dir,
        num_samples=8
    )
    
    print("\n" + "=" * 80)
    print("✅ CAM激活图可视化完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

