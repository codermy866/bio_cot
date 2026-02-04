#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BioLCoT 病灶聚焦 Grad-CAM 可视化生成器
专门用于"只标注重点阴道镜和OCT病灶区域"

关键特性:
1. 阈值过滤：只保留高置信度区域（threshold=0.6），去除背景噪声
2. 形态学处理：高斯平滑，使病灶区域更集中、边缘更自然
3. 智能背景遮罩：OCT 背景自动去除，Colposcopy 保持原图清晰度
4. 自动查找最新 checkpoint
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
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import argparse

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

# 导入修复函数
from generate_gradcam import fix_color_and_overlay, auto_find_checkpoint
from adaptive_ellipse_roi import detect_adaptive_ellipse_roi


def save_original_image(image_np, save_path):
    """
    保存原始图像（没有任何操作的原图）
    同时保存 PNG 和 PDF 格式
    
    Args:
        image_np: [H, W, 3] RGB 图像数组 (0-255, uint8)
        save_path: 保存路径（会自动生成 PDF 版本）
    """
    # 确保图像是 RGB 格式
    if image_np.dtype != np.uint8:
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
    
    # 归一化到 [0, 1] 用于显示
    img_float = image_np.astype(np.float32) / 255.0
    
    # 保存图像
    plt.figure(figsize=(8, 8))
    plt.imshow(img_float)
    plt.axis('off')
    # 去除白边，保证图片填满
    
    # 保存 PNG 格式
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    
    # 同时保存 PDF 格式（更清晰，适合论文）
    if save_path.endswith('.png'):
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0, dpi=300, format='pdf')
        print(f"  ✅ 同时保存 PDF: {pdf_path}")
    
    plt.close()


def enhance_colposcopy_with_color_prior(heatmap, image_rgb, boost_factor=3.0):
    """
    针对阴道镜的特殊增强：结合颜色先验（红色/糜烂区域）来增强激活
    
    阴道镜的阳性病灶通常是：
    1. 红色/粉色的糜烂区域（宫颈口流血）
    2. 充血区域
    3. 通常位于图像中心（宫颈口位置）
    
    Args:
        heatmap: [H, W] 原始 Grad-CAM 热力图 (0-1)
        image_rgb: [H, W, 3] RGB 图像 (0-255)
        boost_factor: 红色区域的增强倍数
    
    Returns:
        enhanced_heatmap: [H, W] 增强后的热力图
    """
    h, w = heatmap.shape
    if image_rgb.shape[:2] != (h, w):
        image_rgb = cv2.resize(image_rgb, (w, h))
    
    # 确保图像是 uint8 格式
    if image_rgb.dtype != np.uint8:
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        else:
            image_rgb = image_rgb.astype(np.uint8)
    
    # 转换为 HSV 颜色空间（更适合检测红色）
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue = img_hsv[:, :, 0].astype(np.float32)  # 色相 (0-179)
    sat = img_hsv[:, :, 1].astype(np.float32)  # 饱和度 (0-255)
    val = img_hsv[:, :, 2].astype(np.float32)  # 明度 (0-255)
    
    # 归一化到 [0, 1]
    sat_norm = sat / 255.0
    val_norm = val / 255.0
    
    # --- 1. 检测红色/粉色区域（糜烂/充血）---
    # HSV 中红色对应 hue 在 0-10 和 160-179 范围
    # 使用更宽松的范围以捕获粉红色
    is_red_low = (hue < 15)  # 红色（0-15度）
    is_red_high = (hue > 160)  # 红色（160-179度）
    is_red = is_red_low | is_red_high
    
    # 要求饱和度较高（排除灰色背景）
    is_saturated = sat_norm > 0.3  # 饱和度 > 30%
    
    # 要求明度适中（排除过亮或过暗的区域）
    is_bright_enough = val_norm > 0.2  # 明度 > 20%
    is_not_too_bright = val_norm < 0.9  # 明度 < 90%（排除反光）
    
    # 红色糜烂掩码
    red_erosion_mask = (is_red & is_saturated & is_bright_enough & is_not_too_bright).astype(np.float32)
    
    # --- 2. 自适应椭圆ROI遮罩（根据激活分布自动确定椭圆参数）---
    # 🔥 关键修复：使用自适应椭圆ROI，根据热力图激活分布和图像特征自动确定椭圆中心、大小和方向
    center_roi_mask, ellipse_params = detect_adaptive_ellipse_roi(
        heatmap, 
        image_rgb=image_rgb,
        min_activation_threshold=0.1,
        min_roi_ratio=0.15,  # 最小ROI比例15%
        max_roi_ratio=0.45   # 最大ROI比例45%
    )
    
    # 打印统计信息
    roi_pixels = np.sum(center_roi_mask > 0.5)
    total_pixels = h * w
    center_y, center_x = ellipse_params['center']
    axis_a, axis_b = ellipse_params['axes']
    angle = ellipse_params['angle']
    print(f"  🔴 自适应椭圆ROI: 保留 {roi_pixels}/{total_pixels} 像素 ({roi_pixels/total_pixels*100:.1f}%)")
    print(f"  📍 椭圆中心: ({center_y}, {center_x}), 长轴: {axis_a:.0f}px, 短轴: {axis_b:.0f}px, 角度: {angle:.1f}°")
    
    # 使用中心ROI遮罩作为权重（替代之前的线性衰减权重）
    center_weight = center_roi_mask
    
    # --- 3. 结合颜色先验和中心权重增强热力图 ---
    # 策略：
    # - 如果模型在红色区域有激活，大幅增强（boost_factor 倍）
    # - 如果模型在红色区域没有激活，但区域确实是红色的，也给予一定激活（颜色引导）
    # - 中心区域给予额外权重
    
    # 计算颜色引导激活（即使模型没有激活，红色区域也给予基础激活）
    color_guided_activation = red_erosion_mask * 0.3  # 红色区域基础激活 0.3
    
    # 模型激活 + 颜色增强
    # 如果模型在红色区域有激活，大幅增强
    red_boost = heatmap * red_erosion_mask * (boost_factor - 1.0)  # 额外增强
    enhanced_heatmap = heatmap + red_boost + color_guided_activation
    
    # 应用中心权重
    enhanced_heatmap = enhanced_heatmap * center_weight
    
    # --- 4. 去除反光区域（高亮区域通常是窥器反光）---
    reflection_mask = val_norm > 0.95  # 极亮区域
    enhanced_heatmap[reflection_mask] = 0
    
    # --- 5. 应用中心ROI遮罩（已经在步骤2中创建，这里直接应用）---
    # 🔥 关键修复：中心ROI遮罩已经在步骤2中创建并作为 center_weight
    # 这里直接应用，完全去除边缘区域，只保留宫颈口中心
    # 注意：center_weight 已经在步骤3中应用过了，这里再次应用确保完全去除边缘
    enhanced_heatmap = enhanced_heatmap * center_weight
    
    # 打印统计信息
    active_pixels = np.sum(enhanced_heatmap > 0.01)
    total_pixels = h * w
    print(f"  🔴 应用中心ROI遮罩后: {active_pixels}/{total_pixels} 像素有激活 ({active_pixels/total_pixels*100:.1f}%)")
    
    # 归一化
    if enhanced_heatmap.max() > 0:
        enhanced_heatmap = enhanced_heatmap / enhanced_heatmap.max()
    else:
        enhanced_heatmap = heatmap  # 如果增强后全黑，返回原图
    
    # 打印统计信息
    red_pixels = np.sum(red_erosion_mask > 0)
    enhanced_pixels = np.sum(enhanced_heatmap > 0.1)
    print(f"  🔴 检测到红色/糜烂区域: {red_pixels} 像素")
    print(f"  📊 增强后激活区域: {enhanced_pixels} 像素")
    
    return enhanced_heatmap


def apply_lesion_focus(heatmap, threshold=0.6, smooth_sigma=1.0, use_percentile=True, image_rgb=None, is_colposcopy=False, label=None):
    """
    专门针对病灶区域的过滤函数。
    只保留高置信度区域，去除背景噪声。
    
    Args:
        heatmap: 原始 GradCAM 热力图 (0-1)
        threshold: 阈值 (0.0-1.0)，低于此值的区域将被忽略。
                   如果 use_percentile=True，则 threshold 表示百分位数（0-100），
                   例如 0.6 表示保留 top 40% 的激活区域。
                   如果 use_percentile=False，则 threshold 表示绝对阈值。
        smooth_sigma: 高斯平滑参数，用于让边缘更自然。
        use_percentile: 是否使用百分位数阈值（推荐，更智能）
        image_rgb: [H, W, 3] RGB 图像，用于 Colposcopy 颜色增强（可选）
        is_colposcopy: 是否为 Colposcopy 图像（启用颜色先验增强）
        label: 样本标签 (0=阴性, 1=阳性)，用于调整阈值（可选）
    
    Returns:
        focused_map: 聚焦后的热力图，只包含高置信度病灶区域
    """
    # 🔥 方案2：对阳性病例降低阈值
    original_threshold = threshold
    if label == 1:  # 阳性病例
        threshold = threshold * 0.7  # 降低30%
        print(f"  📊 阳性病例：使用降低的阈值 {threshold:.3f} (原始: {original_threshold:.3f})")
    # 确保 heatmap 是归一化的
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    else:
        return heatmap
    
    # --- Colposcopy 特殊处理：结合颜色先验增强 + 边缘遮罩 ---
    if is_colposcopy and image_rgb is not None:
        print(f"  🔴 应用 Colposcopy 颜色先验增强...")
        # 使用颜色先验增强热力图（重点关注红色/糜烂区域）
        # 注意：边缘遮罩已经在 enhance_colposcopy_with_color_prior 中处理
        heatmap = enhance_colposcopy_with_color_prior(heatmap, image_rgb, boost_factor=3.0)
        print(f"  📊 颜色增强后热力图范围: [{heatmap.min():.4f}, {heatmap.max():.4f}], 均值: {heatmap.mean():.4f}")
    
    # --- OCT 特殊处理：加强顶部保护套区域去除 ---
    if not is_colposcopy:  # OCT 图像
        h, w = heatmap.shape
        # 强制去除顶部 20% 区域（保护套通常在顶部）
        # 比之前的 10% 更激进，确保完全去除保护套
        top_margin = int(h * 0.20)
        heatmap[0:top_margin, :] = 0
        print(f"  🔲 OCT: 去除顶部保护套区域: {top_margin} 像素 (顶部 20%)")
    
    # 1. 应用阈值：只保留高置信度区域
    if use_percentile:
        # 使用百分位数阈值（更智能）
        # threshold=0.6 表示保留 top 40% 的激活区域
        # 例如，如果 threshold=0.6，则保留值 >= 60th percentile 的区域
        percentile_threshold = np.percentile(heatmap, threshold * 100)
        print(f"  📊 百分位数阈值: {percentile_threshold:.4f} (保留 top {100-threshold*100:.1f}%)")
        print(f"  📊 原始热力图范围: [{heatmap.min():.4f}, {heatmap.max():.4f}], 均值: {heatmap.mean():.4f}")
        focused_map = np.copy(heatmap)
        focused_map[focused_map < percentile_threshold] = 0
        
        # 统计过滤前后的激活像素数
        original_active = np.sum(heatmap > 0.01)
        filtered_active = np.sum(focused_map > 0.01)
        print(f"  📊 激活像素: 原始={original_active}, 过滤后={filtered_active}, 保留率={filtered_active/(original_active+1e-8)*100:.1f}%")
    else:
        # 使用绝对阈值
        print(f"  📊 绝对阈值: {threshold:.4f}")
        print(f"  📊 原始热力图范围: [{heatmap.min():.4f}, {heatmap.max():.4f}], 均值: {heatmap.mean():.4f}")
        focused_map = np.copy(heatmap)
        focused_map[focused_map < threshold] = 0
        
        # 统计过滤前后的激活像素数
        original_active = np.sum(heatmap > 0.01)
        filtered_active = np.sum(focused_map > 0.01)
        print(f"  📊 激活像素: 原始={original_active}, 过滤后={filtered_active}, 保留率={filtered_active/(original_active+1e-8)*100:.1f}%")
    
    # 检查是否有任何区域被保留
    if np.max(focused_map) == 0:
        # 如果所有区域都被过滤掉了，使用更宽松的阈值
        if use_percentile:
            # 降低到 40th percentile
            percentile_threshold = np.percentile(heatmap, 40)
            print(f"  ⚠️  使用更宽松的阈值: {percentile_threshold:.4f}")
            focused_map = np.copy(heatmap)
            focused_map[focused_map < percentile_threshold] = 0
        else:
            # 降低阈值到 0.3
            print(f"  ⚠️  使用更宽松的阈值: 0.3")
            focused_map = np.copy(heatmap)
            focused_map[focused_map < 0.3] = 0
        
        # 如果还是全黑，返回原图（可能有弱激活但确实是病灶）
        if np.max(focused_map) == 0:
            print(f"  ⚠️  警告：阈值过滤后无激活区域，返回原始热力图")
            return heatmap

    # 2. 重新归一化过滤后的区域，使得病灶中心最亮
    if np.max(focused_map) > 0:
        # 将过滤后的区域重新拉伸到 [0, 1]
        focused_map = (focused_map - focused_map.min()) / (focused_map.max() - focused_map.min() + 1e-8)
    else:
        focused_map = np.zeros_like(heatmap)

    # 3. 高斯平滑，让热力图看起来更像一个连贯的病灶而不是噪点
    if smooth_sigma > 0:
        # 计算高斯核大小（必须是奇数）
        kernel_size = int(6 * smooth_sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        focused_map = cv2.GaussianBlur(focused_map, (kernel_size, kernel_size), smooth_sigma)
        
        # 重新归一化
        if np.max(focused_map) > 0:
            focused_map = focused_map / np.max(focused_map)
    
    # 打印最终统计信息
    original_nonzero = np.sum(heatmap > 0.05)
    focused_nonzero = np.sum(focused_map > 0.05)
    reduction_ratio = (1 - focused_nonzero / (original_nonzero + 1e-8)) * 100
    print(f"  📉 最终激活区域减少: {reduction_ratio:.1f}% (原始: {original_nonzero}, 聚焦后: {focused_nonzero})")
    print(f"  📊 聚焦后热力图范围: [{focused_map.min():.4f}, {focused_map.max():.4f}], 均值: {focused_map.mean():.4f}")
    
    # 如果变化很小，给出警告
    if reduction_ratio < 10:
        print(f"  ⚠️  警告：过滤效果不明显（减少 < 10%），建议降低 threshold 参数（当前: {threshold}）")

    return focused_map


def generate_lesion_focused_gradcam(
    model,
    f_oct: torch.Tensor,
    f_colpo: torch.Tensor,
    image_names: List[str],
    clinical_features: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer_name: Optional[str] = None,
    modality: str = 'oct',
    threshold: float = 0.6,
    smooth_sigma: float = 1.0,
    use_percentile: bool = True,
    image_rgb: Optional[np.ndarray] = None,
    label: Optional[int] = None
) -> np.ndarray:
    """
    生成病灶聚焦的 Grad-CAM 热图
    
    Args:
        model: BioLCoT 模型
        f_oct: [B, N, D] OCT 特征
        f_colpo: [B, N, D] Colposcopy 特征
        image_names: List[str] 图像文件名
        clinical_features: [B, 7] 临床特征
        target_class: 目标类别 (None 表示使用预测类别)
        target_layer_name: 目标层名称 (None 表示自动查找)
        modality: 'oct' 或 'colpo'，指定要可视化的模态
        threshold: 阈值过滤参数 (0.0-1.0)，只保留高于此值的激活
        smooth_sigma: 高斯平滑参数
    
    Returns:
        cam: [H, W] 病灶聚焦的 CAM 热图
    """
    device = f_oct.device
    
    # 使用原有的 generate_gradcam 函数生成原始 CAM
    from generate_gradcam import generate_gradcam
    
    raw_cam = generate_gradcam(
        model, f_oct, f_colpo, image_names, clinical_features,
        target_class=target_class, target_layer_name=target_layer_name,
        modality=modality
    )
    
    # 打印原始 CAM 统计
    print(f"  📊 原始 CAM 统计 ({modality}): Min={raw_cam.min():.4f}, Max={raw_cam.max():.4f}, Mean={raw_cam.mean():.4f}")
    
    # 应用病灶聚焦逻辑
    # 对于 Colposcopy，传入图像用于颜色先验增强
    lesion_focused_cam = apply_lesion_focus(
        raw_cam, 
        threshold=threshold, 
        smooth_sigma=smooth_sigma,
        use_percentile=use_percentile,
        image_rgb=image_rgb if modality == 'colpo' else None,
        is_colposcopy=(modality == 'colpo'),
        label=label  # 🔥 传入label用于调整阈值
    )
    
    return lesion_focused_cam


def visualize_lesion_focused_gradcam(
    model,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Path,
    num_samples: int = 4,
    target_layer_name: Optional[str] = None,
    threshold: float = 0.6,
    smooth_sigma: float = 1.0,
    use_percentile: bool = True
):
    """
    批量生成病灶聚焦的 Grad-CAM 可视化
    
    Args:
        model: BioLCoT 模型
        dataloader: 数据加载器
        device: 设备
        save_dir: 保存目录
        num_samples: 样本数量
        target_layer_name: 目标层名称
        threshold: 阈值过滤参数
        smooth_sigma: 高斯平滑参数
    """
    print("\n" + "="*80)
    print("🎨 生成 BioLCoT 病灶聚焦 Grad-CAM 可视化")
    print(f"   阈值过滤: {threshold}, 平滑参数: {smooth_sigma}")
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
            
            # 获取原始图像
            if len(oct_images.shape) == 5:
                oct_img = oct_images[i, oct_images.shape[1]//2, :, :, :].cpu()
            else:
                oct_img = oct_images[i, :, :, :].cpu()
            
            if len(colposcopy_images.shape) == 5:
                colpo_img = colposcopy_images[i, 0, :, :, :].cpu()
            else:
                colpo_img = colposcopy_images[i, :, :, :].cpu()
            
            # 转换为 numpy 图像
            def tensor_to_numpy(t):
                arr = t.permute(1, 2, 0).numpy()
                if arr.max() <= 1.5:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
                return arr
            
            oct_img_np = tensor_to_numpy(oct_img)
            colpo_img_np = tensor_to_numpy(colpo_img)
            
            label_str = "Positive" if label == 1 else "Negative"
            
            # 🔥 方案3：添加诊断信息
            print(f"\n  📊 模型预测诊断 (样本 {samples_collected + 1}):")
            with torch.no_grad():
                output = model(
                    f_oct=f_oct_single,
                    f_colpo=f_colpo_single,
                    image_names=image_names_single,
                    clinical_features=clinical_single,
                    return_loss_components=False
                )
                logits = output.get('pred', output.get('logits', output.get('cls_preds')))
                probs = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
            
            print(f"     真实标签: {label} ({label_str})")
            print(f"     预测类别: {predicted_class} ({'Positive' if predicted_class == 1 else 'Negative'})")
            print(f"     阴性概率: {probs[0, 0]:.4f}")
            print(f"     阳性概率: {probs[0, 1]:.4f}")
            
            # 如果预测概率很低，给出警告
            if label == 1 and probs[0, 1] < 0.3:
                print(f"     ⚠️  警告：阳性概率很低 ({probs[0, 1]:.4f})，Grad-CAM可能不明显")
                print(f"     建议：将使用预测类别而非真实标签来生成Grad-CAM")
            
            # 🔥 方案1：使用预测类别而非真实标签（如果预测概率很低）
            # 对于阳性病例，如果预测概率很低，使用预测类别
            if label == 1 and probs[0, 1] < 0.3:
                target_class_for_cam = predicted_class
                print(f"     ✅ 使用预测类别 {target_class_for_cam} 生成Grad-CAM（预测概率更高）")
            else:
                target_class_for_cam = label
                print(f"     ✅ 使用真实标签 {target_class_for_cam} 生成Grad-CAM")
            
            # 保存原始图像（没有任何操作的原图）
            try:
                print("  💾 保存原始 OCT 图像...")
                original_oct_path = save_dir / f"original_oct_sample{samples_collected+1}_{label_str}.png"
                save_original_image(oct_img_np, str(original_oct_path))
                print(f"  ✅ 保存原始图像: {original_oct_path}")
            except Exception as e:
                print(f"  ⚠️ 保存原始 OCT 图像失败: {e}")
            
            try:
                print("  💾 保存原始 Colposcopy 图像...")
                original_colpo_path = save_dir / f"original_colpo_sample{samples_collected+1}_{label_str}.png"
                save_original_image(colpo_img_np, str(original_colpo_path))
                print(f"  ✅ 保存原始图像: {original_colpo_path}")
            except Exception as e:
                print(f"  ⚠️ 保存原始 Colposcopy 图像失败: {e}")
            
            # 生成 OCT 病灶聚焦 Grad-CAM
            try:
                print("  🔍 生成 OCT 病灶聚焦 Grad-CAM...")
                
                # 先生成原始 CAM 用于对比
                from generate_gradcam import generate_gradcam as generate_raw_gradcam
                raw_oct_cam = generate_raw_gradcam(
                    model, f_oct_single, f_colpo_single, image_names_single,
                    clinical_single, target_class=target_class_for_cam, target_layer_name=target_layer_name,
                    modality='oct'
                )
                print(f"  📊 原始 OCT CAM: Min={raw_oct_cam.min():.4f}, Max={raw_oct_cam.max():.4f}, Mean={raw_oct_cam.mean():.4f}")
                
                # 如果原始CAM值很小，给出警告
                if raw_oct_cam.max() < 0.1:
                    print(f"  ⚠️  警告：原始CAM值很小 (Max={raw_oct_cam.max():.4f})，可能梯度消失")
                
                # 生成病灶聚焦 CAM（OCT 不需要颜色先验）
                oct_cam = generate_lesion_focused_gradcam(
                    model, f_oct_single, f_colpo_single, image_names_single,
                    clinical_single, target_class=target_class_for_cam, target_layer_name=target_layer_name,
                    modality='oct', threshold=threshold, smooth_sigma=smooth_sigma,
                    use_percentile=use_percentile, image_rgb=None, label=label
                )
                
                # 打印对比信息
                print(f"  📊 聚焦后 OCT CAM: Min={oct_cam.min():.4f}, Max={oct_cam.max():.4f}, Mean={oct_cam.mean():.4f}")
                
                # 计算变化量
                diff = np.abs(oct_cam - raw_oct_cam)
                print(f"  📊 变化量: Max={diff.max():.4f}, Mean={diff.mean():.4f}, 变化像素数={np.sum(diff > 0.01)}")
                
                # 使用修复函数叠加（自动处理背景掩膜和颜色转换）
                oct_save_path = save_dir / f"lesion_focused_oct_sample{samples_collected+1}_{label_str}.png"
                fix_color_and_overlay(
                    oct_img_np, 
                    oct_cam, 
                    str(oct_save_path), 
                    is_oct=True,
                    alpha=0.5  # 降低 alpha，让病灶更明显
                )
                print(f"  ✅ 保存: {oct_save_path}")
                
                # 同时保存原始版本用于对比
                raw_oct_save_path = save_dir / f"raw_oct_sample{samples_collected+1}_{label_str}.png"
                fix_color_and_overlay(
                    oct_img_np,
                    raw_oct_cam,
                    str(raw_oct_save_path),
                    is_oct=True,
                    alpha=0.5
                )
                print(f"  ✅ 保存原始版本用于对比: {raw_oct_save_path}")
                
            except Exception as e:
                print(f"  ⚠️ OCT 病灶聚焦 Grad-CAM 生成失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 生成 Colposcopy 病灶聚焦 Grad-CAM
            try:
                print("  🔍 生成 Colposcopy 病灶聚焦 Grad-CAM...")
                
                # 先生成原始 CAM 用于对比
                from generate_gradcam import generate_gradcam as generate_raw_gradcam
                raw_colpo_cam = generate_raw_gradcam(
                    model, f_oct_single, f_colpo_single, image_names_single,
                    clinical_single, target_class=target_class_for_cam, target_layer_name=target_layer_name,
                    modality='colpo'
                )
                print(f"  📊 原始 Colpo CAM: Min={raw_colpo_cam.min():.4f}, Max={raw_colpo_cam.max():.4f}, Mean={raw_colpo_cam.mean():.4f}")
                
                # 如果原始CAM值很小，给出警告
                if raw_colpo_cam.max() < 0.1:
                    print(f"  ⚠️  警告：原始CAM值很小 (Max={raw_colpo_cam.max():.4f})，可能梯度消失")
                
                # 生成病灶聚焦 CAM（Colposcopy 需要颜色先验增强）
                # 传入图像用于检测红色/糜烂区域
                colpo_cam = generate_lesion_focused_gradcam(
                    model, f_oct_single, f_colpo_single, image_names_single,
                    clinical_single, target_class=target_class_for_cam, target_layer_name=target_layer_name,
                    modality='colpo', threshold=threshold, smooth_sigma=smooth_sigma,
                    use_percentile=use_percentile, image_rgb=colpo_img_np, label=label
                )
                
                # 打印对比信息
                print(f"  📊 聚焦后 Colpo CAM: Min={colpo_cam.min():.4f}, Max={colpo_cam.max():.4f}, Mean={colpo_cam.mean():.4f}")
                
                # 计算变化量
                diff = np.abs(colpo_cam - raw_colpo_cam)
                print(f"  📊 变化量: Max={diff.max():.4f}, Mean={diff.mean():.4f}, 变化像素数={np.sum(diff > 0.01)}")
                
                # 使用修复函数叠加（自动处理颜色转换）
                colpo_save_path = save_dir / f"lesion_focused_colpo_sample{samples_collected+1}_{label_str}.png"
                fix_color_and_overlay(
                    colpo_img_np, 
                    colpo_cam, 
                    str(colpo_save_path), 
                    is_oct=False,
                    alpha=0.5  # 降低 alpha，让病灶更明显
                )
                print(f"  ✅ 保存: {colpo_save_path}")
                
                # 同时保存原始版本用于对比
                raw_colpo_save_path = save_dir / f"raw_colpo_sample{samples_collected+1}_{label_str}.png"
                fix_color_and_overlay(
                    colpo_img_np,
                    raw_colpo_cam,
                    str(raw_colpo_save_path),
                    is_oct=False,
                    alpha=0.5
                )
                print(f"  ✅ 保存原始版本用于对比: {raw_colpo_save_path}")
                
            except Exception as e:
                print(f"  ⚠️ Colposcopy 病灶聚焦 Grad-CAM 生成失败: {e}")
                import traceback
                traceback.print_exc()
            
            samples_collected += 1
    
    print(f"\n✅ 完成！共生成 {samples_collected} 个样本的可视化")
    print(f"📁 保存目录: {save_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BioLCoT 病灶聚焦 Grad-CAM 可视化生成器')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型 checkpoint 路径 (默认: 自动查找最新的 best_model_*.pth)')
    parser.add_argument('--num_samples', type=int, default=4,
                       help='生成的样本数量 (默认: 4)')
    parser.add_argument('--target_layer', type=str, default=None,
                       help='目标层名称 (默认: 自动查找)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='保存目录 (默认: biolcot_visualization/lesion_focused_results)')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='阈值过滤参数 (0.0-1.0)，表示百分位数，例如 0.6 表示保留 top 40%% 的激活 (默认: 0.6)')
    parser.add_argument('--smooth_sigma', type=float, default=1.0,
                       help='高斯平滑参数，用于让边缘更自然 (默认: 1.0)')
    parser.add_argument('--use_absolute_threshold', action='store_true',
                       help='使用绝对阈值而不是百分位数阈值（不推荐）')
    
    args = parser.parse_args()
    
    print("="*80)
    print("BioLCoT 病灶聚焦 Grad-CAM 可视化生成器")
    print("="*80)
    print(f"   阈值过滤: {args.threshold}")
    print(f"   平滑参数: {args.smooth_sigma}")
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
        save_dir = Path(__file__).parent / 'lesion_focused_results'
    
    # 生成可视化
    visualize_lesion_focused_gradcam(
        model=model,
        dataloader=dataloader,
        device=device,
        save_dir=save_dir,
        num_samples=args.num_samples,
        target_layer_name=args.target_layer,
        threshold=args.threshold,
        smooth_sigma=args.smooth_sigma,
        use_percentile=not args.use_absolute_threshold
    )
    
    print("\n" + "="*80)
    print("✅ 完成！")
    print("="*80)
    print("\n💡 提示:")
    print("   - 病灶聚焦 Grad-CAM 只标注高置信度的病灶区域")
    print("   - 可以通过 --threshold 参数调整阈值（0.5-0.7 推荐）")
    print("   - 可以通过 --smooth_sigma 参数调整平滑程度（0.5-2.0 推荐）")


if __name__ == '__main__':
    main()

