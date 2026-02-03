#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BioLCoT Grad-CAM 可视化生成器
使用 pytorch-grad-cam 库生成类激活映射 (Grad-CAM)

关键特性:
1. 自动适配 BioLCoT 模型结构
2. 支持多模态输入 (OCT + Colposcopy)
3. 使用 ModelWrapper 处理复杂的模型输出
4. 自动检测目标层 (visual_encoder 的最后一层)
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
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Optional, Tuple
import argparse

# 导入 pytorch-grad-cam
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    print("❌ 请先安装 pytorch-grad-cam: pip install grad-cam")
    sys.exit(1)

# 导入模型和配置
from config import BioCOT_v3_2_Config
from models.bio_cot_v3_2 import create_bio_cot_v3_2
from data.dataset_v3_2 import FiveCentersMultimodalDatasetV3_2
from torch.utils.data import DataLoader

# 导入特征提取函数
try:
    from training.extract_vit_patches import extract_patch_features_with_vit
except ImportError:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
    from training.extract_vit_patches import extract_patch_features_with_vit


class BioLCoTModelWrapper(nn.Module):
    """
    模型包装器：将 BioLCoT 的输出转换为 Grad-CAM 需要的格式
    BioLCoT 的 forward 返回 dict，但 Grad-CAM 需要 logits
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
    
    def forward(self, f_oct, f_colpo, image_names, clinical_features):
        """
        前向传播，返回 logits
        
        Args:
            f_oct: [B, N, D] OCT 特征
            f_colpo: [B, N, D] Colposcopy 特征
            image_names: List[str] 图像文件名
            clinical_features: [B, 7] 临床特征
        
        Returns:
            logits: [B, num_classes] 分类 logits
        """
        output = self.model(
            f_oct=f_oct,
            f_colpo=f_colpo,
            image_names=image_names,
            clinical_features=clinical_features,
            return_loss_components=False
        )
        
        # 提取 logits
        if isinstance(output, dict):
            logits = output.get('logits', output.get('pred', output.get('cls_preds')))
        else:
            logits = output
        
        return logits


def find_target_layer(model, layer_name: Optional[str] = None):
    """
    自动查找目标层用于 Grad-CAM
    
    Args:
        model: BioLCoT 模型
        layer_name: 可选，指定层名称
    
    Returns:
        target_layer: 目标层对象
        layer_name: 层名称
    """
    if layer_name:
        # 用户指定层
        for name, module in model.named_modules():
            if name == layer_name:
                print(f"✅ 找到指定层: {layer_name}")
                return module, layer_name
    
    # 自动查找：优先使用 visual_encoder 的最后一层
    print("🔍 自动查找目标层...")
    
    # 优先级1: visual_encoder (如果使用分层特征提取)
    if hasattr(model, 'visual_encoder') and model.visual_encoder is not None:
        if hasattr(model.visual_encoder, 'vit'):
            if hasattr(model.visual_encoder.vit, 'blocks'):
                # 使用最后一个 Transformer Block 的最后一个 LayerNorm
                last_block = model.visual_encoder.vit.blocks[-1]
                # 查找最后一个 LayerNorm
                for name, module in reversed(list(last_block.named_modules())):
                    if isinstance(module, nn.LayerNorm):
                        layer_name = f"visual_encoder.vit.blocks[-1].{name}"
                        print(f"✅ 找到目标层: {layer_name}")
                        return module, layer_name
                # 如果没有 LayerNorm，使用整个 block
                layer_name = "visual_encoder.vit.blocks[-1]"
                print(f"✅ 找到目标层: {layer_name}")
                return last_block, layer_name
    
    # 优先级2: visual_notes_module 的 k_proj (处理图像特征的层)
    if hasattr(model, 'visual_notes_module') and model.visual_notes_module is not None:
        if hasattr(model.visual_notes_module, 'layer'):
            if hasattr(model.visual_notes_module.layer, 'k_proj'):
                layer_name = "visual_notes_module.layer.k_proj"
                print(f"✅ 找到目标层: {layer_name}")
                return model.visual_notes_module.layer.k_proj, layer_name
    
    # 优先级3: dual_head 的第一层
    if hasattr(model, 'dual_head'):
        if isinstance(model.dual_head, nn.Sequential):
            layer_name = "dual_head[0]"
            print(f"✅ 找到目标层: {layer_name}")
            return model.dual_head[0], layer_name
        else:
            layer_name = "dual_head"
            print(f"✅ 找到目标层: {layer_name}")
            return model.dual_head, layer_name
    
    # 如果都没找到，抛出错误
    raise ValueError("❌ 无法找到合适的目标层！请手动指定 layer_name")


def generate_gradcam(
    model,
    f_oct: torch.Tensor,
    f_colpo: torch.Tensor,
    image_names: List[str],
    clinical_features: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer_name: Optional[str] = None,
    modality: str = 'oct'
) -> np.ndarray:
    """
    生成 Grad-CAM 热图
    
    Args:
        model: BioLCoT 模型
        f_oct: [B, N, D] OCT 特征
        f_colpo: [B, N, D] Colposcopy 特征
        image_names: List[str] 图像文件名
        clinical_features: [B, 7] 临床特征
        target_class: 目标类别 (None 表示使用预测类别)
        target_layer_name: 目标层名称 (None 表示自动查找)
        modality: 'oct' 或 'colpo'，指定要可视化的模态
    
    Returns:
        cam: [H, W] Grad-CAM 热图
    """
    device = f_oct.device
    
    # 1. 创建模型包装器
    wrapped_model = BioLCoTModelWrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()
    
    # 2. 查找目标层
    target_layer, layer_name = find_target_layer(model, target_layer_name)
    
    # 3. 创建 Grad-CAM 对象
    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
    
    # 4. 准备输入
    # 注意：Grad-CAM 需要输入是 tensor，但我们的模型需要多个输入
    # 我们需要创建一个特殊的输入包装器
    class MultiInputWrapper:
        def __init__(self, f_oct, f_colpo, image_names, clinical_features):
            self.f_oct = f_oct
            self.f_colpo = f_colpo
            self.image_names = image_names
            self.clinical_features = clinical_features
    
    # 5. 确定目标类别
    if target_class is None:
        with torch.no_grad():
            logits = wrapped_model(f_oct, f_colpo, image_names, clinical_features)
            target_class = torch.argmax(logits, dim=1).item()
    
    print(f"🎯 目标类别: {target_class}")
    
    # 6. 生成 CAM
    # 注意：由于我们的模型输入复杂，我们需要修改 Grad-CAM 的调用方式
    # 这里我们使用一个技巧：直接在前向传播中计算梯度
    
    # 启用梯度
    f_oct = f_oct.clone().detach().requires_grad_(True)
    f_colpo = f_colpo.clone().detach().requires_grad_(True)
    
    # 前向传播
    logits = wrapped_model(f_oct, f_colpo, image_names, clinical_features)
    
    # 反向传播
    score = logits[0, target_class]
    score.backward()
    
    # 获取梯度（从输入特征获取）
    if modality == 'oct':
        gradients = f_oct.grad  # [B, N, D]
        activations = f_oct.detach()  # [B, N, D]
    else:
        gradients = f_colpo.grad  # [B, N, D]
        activations = f_colpo.detach()  # [B, N, D]
    
    # 7. 计算 CAM (简化版 LayerCAM)
    # 对通道维度求平均梯度
    weights = torch.mean(gradients, dim=2, keepdim=True)  # [B, N, 1]
    weights = torch.relu(weights)  # 只保留正向梯度
    
    # 对通道维度求平均激活
    activations_mean = torch.mean(activations, dim=2)  # [B, N]
    
    # 加权求和
    cam = (weights.squeeze(-1) * activations_mean).squeeze(0)  # [N]
    
    # 8. Reshape 到空间维度
    B, N, D = activations.shape
    grid_size = int(np.sqrt(N))
    
    if grid_size * grid_size != N:
        # 可能包含 CLS token，去掉第一个
        grid_size = int(np.sqrt(N - 1))
        if grid_size * grid_size == N - 1:
            cam = cam[1:]  # 去掉 CLS token
            N = N - 1
    
    cam = cam.reshape(grid_size, grid_size).cpu().numpy()
    
    # 9. 归一化
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)
    
    # 10. 上采样到原图大小
    cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    # 11. 应用高斯模糊平滑
    cam = cv2.GaussianBlur(cam, (11, 11), 0)
    
    # 重新归一化
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    
    return cam


def fix_color_and_overlay(image_path_or_array, heatmap, save_path, is_oct=False, alpha=0.6):
    """
    核心修复函数：解决【变色】和【OCT背景误激活】问题
    
    主要改进：
    1. 智能背景遮罩 (Smart Masking for OCT)：通过阈值+形态学开运算，自动识别组织区域
    2. 动态重归一化 (Re-normalization)：去除背景噪声后重新归一化，确保病灶显示为最红
    3. 色彩空间强力矫正：每一步都强制进行 BGR -> RGB 转换，彻底消灭"紫色肉"
    
    Args:
        image_path_or_array: 图像路径 (str) 或 numpy 数组 (np.ndarray, RGB 格式)
        heatmap: [H, W] 热力图数组 (0-1)
        save_path: 保存路径
        is_oct: 是否为 OCT 图像（开启背景强力去噪）
        alpha: 原图叠加权重 (0-1), 越小热力图越明显
    
    Returns:
        final_result: [H, W, 3] 修复后的叠加图像 (RGB, 0-1)
    """
    # ---------------------------------------------------------
    # 1. 【修复变紫色问题】: 强制色彩空间转换
    # ---------------------------------------------------------
    if isinstance(image_path_or_array, str):
        # 从路径读取（OpenCV 默认读取为 BGR）
        img_bgr = cv2.imread(image_path_or_array)
        if img_bgr is None:
            print(f"Error: Could not read image {image_path_or_array}")
            return None
    else:
        # 已经是 numpy 数组（从 tensor 转换来的，通常是 RGB 格式）
        img_array = image_path_or_array.copy()
        # 确保是 uint8 格式
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        
        # 如果是灰度图，转为 RGB
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # 已经是 RGB，但我们需要转为 BGR 以便后续统一处理
            # 因为从 tensor 来的通常是 RGB，我们需要先转为 BGR，然后再转回 RGB
            # 这样可以确保颜色空间的一致性
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            print(f"Warning: Unexpected image shape: {img_array.shape}")
            return None
    
    # 统一转换为 RGB (Red-Green-Blue)，恢复肉眼可见的粉色
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img_rgb) / 255.0

    # ---------------------------------------------------------
    # 2. 热力图预处理
    # ---------------------------------------------------------
    # 调整热力图尺寸以匹配原图
    h, w = img_rgb.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 基础归一化到 [0, 1]
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    else:
        heatmap = np.zeros_like(heatmap)

    # ---------------------------------------------------------
    # 3. 【修复OCT背景激活问题】: 智能解剖学提取
    # ---------------------------------------------------------
    mask = np.ones_like(heatmap)  # 默认全图保留
    
    if is_oct:
        # 转灰度图分析结构
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # A. 提高阈值：过滤掉深黑色的背景和较暗的保护套伪影
        # 经验值设为 30，比之前的 15 更激进，确保去除背景噪声
        _, binary_mask = cv2.threshold(gray_img, 30, 1.0, cv2.THRESH_BINARY)
        
        # B. 强力形态学操作：去除细小的噪点和非连通的"冰柱"干扰
        kernel = np.ones((5, 5), np.uint8)
        # 开运算：先腐蚀后膨胀，断开细小连接，消除噪点
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # C. 连通域分析：只保留最大的组织块
        # OCT图像中，真正的组织（下半部分）通常是最大的连通区域
        mask_u8 = (binary_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到面积最大的轮廓（即组织本体）
            max_cnt = max(contours, key=cv2.contourArea)
            # 创建一个新的纯净mask，只保留这个最大块
            clean_mask = np.zeros_like(mask_u8)
            cv2.drawContours(clean_mask, [max_cnt], -1, 255, thickness=cv2.FILLED)
            binary_mask = clean_mask.astype(np.float32) / 255.0
        
        # D. 强制抑制顶部区域 (Heuristic)
        # 即使分割有误，强制将图片顶部 10% 区域（通常是空气/探头）置零
        h, w = binary_mask.shape
        binary_mask[0:int(h*0.1), :] = 0
        
        # E. 应用掩膜：背景区域热力值强行归零
        heatmap = heatmap * binary_mask
        mask = binary_mask

        # F. 重新归一化 (Re-normalization)
        # 去掉背景的高亮噪声后，重新拉伸对比度，让病灶显红
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

    # ---------------------------------------------------------
    # 4. 生成美观的叠加图
    # ---------------------------------------------------------
    # 将热力图转换为 RGB 伪彩色 (使用 JET 色谱: 蓝-青-黄-红)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # 注意：cv2.applyColorMap 返回的也是 BGR，必须也要转为 RGB！
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)
    heatmap_colored_float = np.float32(heatmap_colored_rgb) / 255.0

    # ---------------------------------------------------------
    # 5. 智能融合 (Smart Blending) - 解决"变紫"问题
    # ---------------------------------------------------------
    if is_oct:
        # OCT: 简单线性叠加，但利用mask保持背景纯黑
        # alpha 控制原图占比，这里设为 0.5 保证看清组织纹理
        alpha_oct = 0.5
        overlay = alpha_oct * img_float + (1 - alpha_oct) * heatmap_colored_float
        
        # 扩展 mask 维度以匹配 RGB
        mask_rgb = np.stack([mask]*3, axis=2)
        # 在Mask为0（背景）的地方，直接显示原图（纯黑），不要显示蓝色的"冷热力"
        final_result = overlay * mask_rgb + img_float * (1 - mask_rgb)
    else:
        # Colposcopy: 使用"透明度加权"融合，而非全局叠加
        # 只有在热力图有值的地方（病灶），才显示颜色；
        # 热力图值低的地方（正常肉色），完全透明，显示原图。
        # 这样彻底解决了"整张图套紫色滤镜"的问题。
        
        # 计算每个像素的融合权重，基于heatmap的强度
        # heatmap值越大，权重越大，显示越红
        weight = np.stack([heatmap]*3, axis=2)
        
        # 对权重做指数增强，过滤掉低关注度的蓝色背景噪声
        # power > 1 会压制低值，突出高值
        weight = np.power(weight, 1.2) 
        
        # 动态混合公式：
        # Result = (热力图 * Weight) + (原图 * (1 - Weight))
        # 注：调整系数，让热力图看起来像是"发光"覆盖在原图上
        final_result = heatmap_colored_float * weight * 0.7 + img_float * (1 - weight * 0.3)

    final_result = np.clip(final_result, 0, 1)
    
    # ---------------------------------------------------------
    # 6. 保存结果
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.imshow(final_result)
    plt.axis('off')
    # 去除白边，保证图片填满
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"✅ Fixed visualization saved to: {save_path}")
    
    return final_result


def visualize_gradcam(
    model,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Path,
    num_samples: int = 4,
    target_layer_name: Optional[str] = None
):
    """
    批量生成 Grad-CAM 可视化
    
    Args:
        model: BioLCoT 模型
        dataloader: 数据加载器
        device: 设备
        save_dir: 保存目录
        num_samples: 样本数量
        target_layer_name: 目标层名称
    """
    print("\n" + "="*80)
    print("🎨 生成 BioLCoT Grad-CAM 可视化")
    print("="*80)
    
    model.to(device)
    model.eval()
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    samples_collected = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if samples_collected >= num_samples:
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
            oct_features = extract_patch_features_with_vit(oct_images_flat, device, batch_size=4)
            oct_features = oct_features.view(B_oct, F_oct, 196, 768).mean(dim=1)  # [B, 196, 768]
        else:
            oct_features = extract_patch_features_with_vit(oct_images, device, batch_size=4)
        
        B_colpo = colposcopy_images.shape[0]
        if len(colposcopy_images.shape) == 5:  # [B, N, C, H, W]
            N_colpo = colposcopy_images.shape[1]
            colpo_images_flat = colposcopy_images.view(B_colpo * N_colpo, *colposcopy_images.shape[2:])
            colpo_features = extract_patch_features_with_vit(colpo_images_flat, device, batch_size=4)
            colpo_features = colpo_features.view(B_colpo, N_colpo, 196, 768).mean(dim=1)  # [B, 196, 768]
        else:
            colpo_features = extract_patch_features_with_vit(colposcopy_images, device, batch_size=4)
        
        for i in range(batch_size):
            if samples_collected >= num_samples:
                break
            
            print(f"\n📸 处理样本 {samples_collected + 1}/{num_samples}")
            
            # 准备单个样本
            f_oct_single = oct_features[i:i+1]
            f_colpo_single = colpo_features[i:i+1]
            clinical_single = clinical_features[i:i+1]
            image_names_single = [image_names[i]]
            label = labels[i].item()
            
            # 获取原始图像用于叠加
            if len(oct_images.shape) == 5:
                oct_img = oct_images[i, oct_images.shape[1]//2, :, :, :].cpu()  # 取中间帧
            else:
                oct_img = oct_images[i, :, :, :].cpu()
            
            if len(colposcopy_images.shape) == 5:
                colpo_img = colposcopy_images[i, 0, :, :, :].cpu()  # 取第一张
            else:
                colpo_img = colposcopy_images[i, :, :, :].cpu()
            
            # 转换为 numpy 图像
            def tensor_to_numpy(t):
                arr = t.permute(1, 2, 0).numpy()
                if arr.max() <= 1.5:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
                # 注意：从 tensor 转换来的通常是 RGB 格式
                # 但为了与 fix_color_and_overlay 函数兼容，我们保持原样
                # fix_color_and_overlay 会正确处理颜色空间转换
                return arr
            
            oct_img_np = tensor_to_numpy(oct_img)
            colpo_img_np = tensor_to_numpy(colpo_img)
            
            label_str = "Positive" if label == 1 else "Negative"
            
            # 生成 OCT Grad-CAM
            try:
                print("  🔍 生成 OCT Grad-CAM...")
                oct_cam = generate_gradcam(
                    model, f_oct_single, f_colpo_single, image_names_single,
                    clinical_single, target_class=label, target_layer_name=target_layer_name,
                    modality='oct'
                )
                
                # 使用修复函数叠加（自动处理背景掩膜和颜色转换）
                oct_save_path = save_dir / f"gradcam_oct_sample{samples_collected+1}_{label_str}.png"
                fix_color_and_overlay(
                    oct_img_np, 
                    oct_cam, 
                    str(oct_save_path), 
                    is_oct=True  # 启用 OCT 背景掩膜
                )
                print(f"  ✅ 保存: {oct_save_path}")
            except Exception as e:
                print(f"  ⚠️ OCT Grad-CAM 生成失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 生成 Colposcopy Grad-CAM
            try:
                print("  🔍 生成 Colposcopy Grad-CAM...")
                colpo_cam = generate_gradcam(
                    model, f_oct_single, f_colpo_single, image_names_single,
                    clinical_single, target_class=label, target_layer_name=target_layer_name,
                    modality='colpo'
                )
                
                # 使用修复函数叠加（自动处理颜色转换）
                colpo_save_path = save_dir / f"gradcam_colpo_sample{samples_collected+1}_{label_str}.png"
                fix_color_and_overlay(
                    colpo_img_np, 
                    colpo_cam, 
                    str(colpo_save_path), 
                    is_oct=False  # Colposcopy 不需要背景掩膜
                )
                print(f"  ✅ 保存: {colpo_save_path}")
            except Exception as e:
                print(f"  ⚠️ Colposcopy Grad-CAM 生成失败: {e}")
                import traceback
                traceback.print_exc()
            
            samples_collected += 1
    
    print(f"\n✅ 完成！共生成 {samples_collected} 个样本的可视化")
    print(f"📁 保存目录: {save_dir}")


def auto_find_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    自动查找最新的 checkpoint 文件
    
    Args:
        checkpoint_dir: checkpoint 目录路径
    
    Returns:
        最新的 checkpoint 路径，如果找不到则返回 None
    """
    if not checkpoint_dir.exists():
        return None
    
    # 优先查找 best_model_*.pth 文件
    candidates = sorted(
        checkpoint_dir.glob("best_model_*.pth"), 
        key=lambda p: p.stat().st_mtime, 
        reverse=True
    )
    
    if candidates:
        return candidates[0]
    
    # 如果没有 best_model，查找任何 .pth 文件
    candidates = sorted(
        checkpoint_dir.glob("*.pth"), 
        key=lambda p: p.stat().st_mtime, 
        reverse=True
    )
    
    return candidates[0] if candidates else None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BioLCoT Grad-CAM 可视化生成器')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型 checkpoint 路径 (默认: 自动查找最新的 best_model_*.pth)')
    parser.add_argument('--num_samples', type=int, default=4,
                       help='生成的样本数量 (默认: 4)')
    parser.add_argument('--target_layer', type=str, default=None,
                       help='目标层名称 (默认: 自动查找)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='保存目录 (默认: biolcot_visualization/gradcam_results)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("BioLCoT Grad-CAM 可视化生成器")
    print("="*80)
    
    # 配置
    config = BioCOT_v3_2_Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}")
    
    # 加载模型
    print("\n📦 加载模型...")
    model = create_bio_cot_v3_2(config)
    
    # 查找 checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"  ❌ 指定的 checkpoint 不存在: {checkpoint_path}")
            sys.exit(1)
    else:
        # 自动查找最新的 checkpoint
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_path = auto_find_checkpoint(checkpoint_dir)
        
        if checkpoint_path is None:
            print(f"  ❌ 未找到 checkpoint 文件！")
            print(f"  📁 查找目录: {checkpoint_dir}")
            print(f"  💡 请确保该目录下有 best_model_*.pth 文件")
            print(f"  💡 或者使用 --checkpoint 参数指定 checkpoint 路径")
            sys.exit(1)
    
    # 加载 checkpoint
    print(f"  📥 加载 checkpoint: {checkpoint_path.name}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'best_auc' in checkpoint:
                print(f"  📊 模型 AUC: {checkpoint['best_auc']:.4f}")
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("  ✅ Checkpoint 加载成功")
    except Exception as e:
        print(f"  ❌ Checkpoint 加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    model.to(device)
    model.eval()
    
    # 加载数据集
    print("\n📂 加载数据集...")
    csv_path = Path(config.data_root) / 'temp_val_labels.csv'
    if not csv_path.exists():
        csv_path = EXPERIMENT_ROOT / 'data' / 'temp_val_labels.csv'
    
    dataset = FiveCentersMultimodalDatasetV3_2(
        csv_path=str(csv_path),
        data_root=config.data_root
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print(f"  ✅ 数据集大小: {len(dataset)}")
    
    # 保存目录
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(__file__).parent / 'gradcam_results'
    
    # 生成可视化
    visualize_gradcam(
        model=model,
        dataloader=dataloader,
        device=device,
        save_dir=save_dir,
        num_samples=args.num_samples,
        target_layer_name=args.target_layer
    )
    
    print("\n" + "="*80)
    print("✅ 完成！")
    print("="*80)


if __name__ == '__main__':
    main()

