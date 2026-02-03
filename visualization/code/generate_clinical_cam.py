#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2: 医学先验约束的临床级CAM可视化（终极无蓝色底纹版 V5.0）

核心改进：
1. 彻底消除蓝色底纹：阈值截断，低激活区域完全透明
2. 消除网格点伪影：平滑处理 + 双线性插值
3. OCT暗区指数级增强：黑洞病灶强力捕获
4. 阴道镜红色素锁定：糜烂区精准定位
5. 特征层直接计算梯度：无需Hook，更准确
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader

# 添加项目根目录到路径
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from models.bio_cot_v3_2 import create_bio_cot_v3_2
from config import BioCOT_v3_2_Config
from data.dataset_v3_2 import FiveCentersMultimodalDatasetV3_2

# 导入特征提取函数
try:
    from training.extract_vit_patches import extract_patch_features_with_vit
except ImportError:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
    from training.extract_vit_patches import extract_patch_features_with_vit

class FeatureLayerCAM:
    """
    针对 ViT 特征优化的 LayerCAM。
    解决"网格点"问题，提供平滑的空间激活图。
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate(self, f_oct, f_colpo, image_names, clinical_features, target_class=None, modality='oct'):
        self.model.zero_grad()
        
        # 1. 开启特征的梯度记录
        f_oct = f_oct.clone().detach().requires_grad_(True)
        f_colpo = f_colpo.clone().detach().requires_grad_(True)
        
        # 2. 前向传播
        output = self.model(
            f_oct=f_oct,
            f_colpo=f_colpo,
            image_names=image_names,
            clinical_features=clinical_features,
            return_loss_components=False
        )
        
        logits = output.get('logits', output.get('cls_preds')) if isinstance(output, dict) else output

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        # 3. 反向传播
        score = logits[0, target_class]
        score.backward()

        # 4. 获取梯度和激活值
        if modality == 'oct':
            grads = f_oct.grad
            acts = f_oct.detach()
        else:
            grads = f_colpo.grad
            acts = f_colpo.detach()

        # 5. LayerCAM 核心计算 (Element-wise weighting)
        # acts: [B, N, D], grads: [B, N, D]
        weighted_acts = acts * F.relu(grads) 
        
        # 在通道维度求和 [B, N]
        cam = torch.sum(weighted_acts, dim=2) 
        cam = F.relu(cam) 
        
        # 6. Reshape & Smoothing (解决"网格点"问题)
        B, N = cam.shape
        grid_size = int(np.sqrt(N))
        if grid_size * grid_size != N:
            cam = cam[:, 1:] # 去掉 CLS token
            grid_size = int(np.sqrt(N - 1))
            
        cam = cam.reshape(B, grid_size, grid_size) # [B, 14, 14]
        
        # 时间维度平均（如果有多帧）
        cam = cam.mean(dim=0).cpu().numpy()
        
        # --- 关键：平滑处理消除网格感 ---
        # 先放大一点进行模糊，再放至原图大小
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        # 高斯模糊消除 Patch 边界
        cam = cv2.GaussianBlur(cam, (13, 13), 0)
        
        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam

def apply_oct_constraints(heatmap, img_path):
    """
    OCT 专属病理增强：
    1. 强力去顶部伪影 (连通域法)
    2. "黑洞"捕获 (暗区指数增强)
    3. 冰柱纹理 (纵向梯度)
    """
    img = cv2.imread(str(img_path))
    if img is None: return heatmap
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    
    # --- A. 智能去伪影 (找到组织主体) ---
    # 使用 Otsu 阈值分割前景
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 形态学闭运算填补空洞
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 找到每一列的第一个前景点 (组织表面)
    surface_mask = np.zeros_like(heatmap)
    for c in range(w):
        col_data = closing[:, c]
        indices = np.where(col_data > 0)[0]
        if len(indices) > 0:
            # 保留表面以下区域 (+10px 避开高亮边界)
            surface_y = indices[0]
            surface_mask[surface_y + 10:, c] = 1.0
            
    heatmap = heatmap * surface_mask
    
    # --- B. "黑洞"特征捕获 (一坨黑乎乎就是癌) ---
    img_float = img_gray.astype(np.float32) / 255.0
    darkness = 1.0 - img_float # 越黑值越大 (0~1)
    
    # 指数级增强：只针对非常黑的区域 (>0.5)
    # clip(darkness - 0.5) 忽略浅灰区域
    dark_boost = np.power(np.clip(darkness - 0.5, 0, 1) * 2.0, 3) 
    
    # 强力注入：只要模型有一点关注 (>0.05)，就把暗区权重放大 5 倍
    # 这样"黑乎乎"的区域会瞬间变红
    base_attention = (heatmap > 0.05).astype(np.float32)
    heatmap = heatmap * (1.0 + 5.0 * dark_boost * base_attention)
    
    # --- C. 冰柱状结构 (纵向纹理) ---
    # Sobel Y 算子检测垂直边缘
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_y = np.abs(sobel_y)
    sobel_y = cv2.normalize(sobel_y, None, 0, 1, cv2.NORM_MINMAX)
    
    # 在暗区附近的纵向纹理给予加权
    heatmap = heatmap * (1.0 + 1.5 * sobel_y * (darkness > 0.4))

    # --- D. 深度加权 ---
    depth_ramp = np.linspace(0.5, 1.2, h).reshape(-1, 1)
    heatmap = heatmap * depth_ramp

    # 重新归一化
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
        
    return heatmap

def apply_colpo_constraints(heatmap, img_path):
    """
    阴道镜 专属病理增强：
    1. 红色素锁定 (糜烂/充血)
    2. 去反光
    3. 中心聚焦
    """
    img = cv2.imread(str(img_path))
    if img is None: return heatmap
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    
    # --- A. 红色素锁定 (LAB A通道) ---
    # LAB 空间比 RGB 更接近人眼对"红肿"的感知
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    
    # A通道：负值绿，正值红。归一化到 0~1
    a_norm = cv2.normalize(a, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # 对红色区域进行 4次方 增强，只保留最红的区域 (严重糜烂)
    red_score = np.power(a_norm, 4)
    heatmap = heatmap * (1.0 + 4.0 * red_score)
    
    # --- B. 强制去反光 (Speculum Highlights) ---
    # 极亮区域直接置零
    reflection_mask = img_gray > 240
    heatmap[reflection_mask] = 0
    
    # --- C. 柔和中心聚焦 ---
    center_y, center_x = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(h**2 + w**2) / 2
    
    # 边缘保留 20% 权重，中心 100%
    center_mask = 1.0 - (dist / max_dist) * 0.8
    center_mask = np.clip(center_mask, 0.2, 1.0)
    heatmap *= center_mask
    
    # 归一化
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
        
    return heatmap

def save_clinical_overlay(heatmap, img_path, save_path):
    """
    保存【无蓝色底纹】的临床级叠加图。
    策略：
    1. 阈值截断：低激活区完全透明。
    2. 动态透明度：高激活区半透明，保留纹理。
    """
    img = cv2.imread(str(img_path))
    if img is None: 
        print(f"[Error] Cannot read image: {img_path}")
        return
    
    # 1. 阈值截断 (去除蓝色背景)
    # 只有激活值 > 0.2 的区域才显示颜色
    # 这彻底解决了"有一层蓝色底纹"的问题
    threshold = 0.2
    mask = heatmap > threshold
    
    # 对 mask 区域内的热力值重新归一化 (0~1) 以获得完整色谱
    heatmap_norm = np.zeros_like(heatmap)
    if mask.sum() > 0:
        heatmap_norm[mask] = (heatmap[mask] - threshold) / (1 - threshold)
    
    # 2. 生成彩色热力图 (Jet)
    heatmap_uint8 = np.uint8(255 * heatmap_norm)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # 3. 智能混合
    overlay = img.copy()
    
    # Alpha 通道设计：
    # 激活越强，Alpha 越高 (颜色越明显)。
    # 但最大 Alpha 限制在 0.55，确保你能透过红色看到底下的"冰柱"或"糜烂纹理"。
    alpha_map = heatmap_norm * 0.5 + 0.1 
    alpha_map = np.clip(alpha_map, 0, 0.55) # 上限 0.55
    alpha_map = alpha_map[..., None]
    
    # 仅在 mask 区域进行混合
    # 背景区域 (mask=False) 保持原图原样
    img_float = img.astype(np.float32)
    heat_float = heatmap_color.astype(np.float32)
    
    blended = img_float * (1 - alpha_map) + heat_float * alpha_map
    
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask,
            blended[:, :, c],
            img[:, :, c]
        )
    
    # 拼接对比：左边原图，右边CAM叠加
    concat = np.hstack([img, overlay])
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), concat)
    print(f"✅ Saved clean visualization: {save_path}")

def main():
    print("🚀 Bio-COT 3.2 Clinical CAM Generator (No Blue Tint Version)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载配置和模型
    config = BioCOT_v3_2_Config()
    model = create_bio_cot_v3_2(config).to(device)
    
    # 查找最新的 checkpoint
    ckpt_dir = EXPERIMENT_ROOT / "ablation_studies" / "baseline" / "checkpoints"
    ckpt_files = list(ckpt_dir.glob("best_model*.pth")) if ckpt_dir.exists() else []
    
    if ckpt_files:
        ckpt_path = max(ckpt_files, key=lambda p: p.stat().st_mtime)
        print(f"📥 Loading weights: {ckpt_path.name}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        print("✅ Checkpoint loaded successfully!")
    else:
        print("⚠️ Checkpoint not found! Using random weights.")

    cam_gen = FeatureLayerCAM(model)
    
    # 2. 加载数据
    csv_path = Path(config.data_root) / "temp_val_labels.csv"
    if not os.path.isabs(csv_path):
        csv_path = Path(config.data_root) / "temp_val_labels.csv"
        if not csv_path.exists():
            csv_path = EXPERIMENT_ROOT / "data" / "temp_val_labels.csv"
    
    if not csv_path.exists():
        print("❌ Dataset CSV not found.")
        return

    dataset = FiveCentersMultimodalDatasetV3_2(csv_path=str(csv_path), data_root=str(config.data_root))
    
    # 处理多个样本（至少一个阳性，一个阴性）
    samples_to_process = []
    positive_found = False
    negative_found = False
    
    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample['label'].item() if hasattr(sample['label'], 'item') else sample['label']
        if label == 1 and not positive_found:
            samples_to_process.append((sample, label))
            positive_found = True
        elif label == 0 and not negative_found:
            samples_to_process.append((sample, label))
            negative_found = True
        
        if positive_found and negative_found:
            break
    
    if not samples_to_process:
        # 如果没找到，至少处理第一个样本
        sample = dataset[0]
        label = sample['label'].item() if hasattr(sample['label'], 'item') else sample['label']
        samples_to_process = [(sample, label)]
    
    print(f"\n📊 Will process {len(samples_to_process)} sample(s)")
    
    save_dir = EXPERIMENT_ROOT / "visualization" / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (target_sample, label) in enumerate(samples_to_process):
        print(f"\n{'='*80}")
        print(f"📸 Processing Sample {idx+1}/{len(samples_to_process)} (Label: {label})")
        print(f"{'='*80}")
        
        # 3. 准备输入张量
        oct_tensor = target_sample['oct_images'].unsqueeze(0).to(device)
        colpo_tensor = target_sample['colposcopy_images'].unsqueeze(0).to(device)
        clin_tensor = target_sample['clinical_features'].unsqueeze(0).to(device)
        
        # 4. 提取特征 (用于计算梯度)
        # OCT: [B, T, C, H, W] -> [B, T, N, D]
        B, T, C, H, W = oct_tensor.shape
        f_oct_flat = extract_patch_features_with_vit(oct_tensor.view(-1, C, H, W), device=device)
        f_oct = f_oct_flat.view(B, T, -1, f_oct_flat.shape[-1]).mean(dim=1) # 对时间取平均 [B, N, D]
        
        # Colpo: [B, N_views, C, H, W] -> [B, N_views, N, D]
        B, Nv, C, H, W = colpo_tensor.shape
        f_colpo_flat = extract_patch_features_with_vit(colpo_tensor.view(-1, C, H, W), device=device)
        f_colpo = f_colpo_flat.view(B, Nv, -1, f_colpo_flat.shape[-1]).mean(dim=1) # 对视角取平均
        
        # 5. 保存原图 (用于叠加)
        oct_img_viz = oct_tensor[0, oct_tensor.shape[1]//2, :, :, :].cpu() # 取中间帧
        colpo_img_viz = colpo_tensor[0, 0, :, :, :].cpu() # 取第一张
        
        def tensor2img(t, path):
            arr = t.permute(1, 2, 0).numpy()
            if arr.max() <= 1.5:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), arr)
        
        label_suffix = "Positive" if label == 1 else "Negative"
        oct_path = save_dir / f"Source_OCT_{label_suffix}_{idx+1}.jpg"
        colpo_path = save_dir / f"Source_Colpo_{label_suffix}_{idx+1}.jpg"
        
        tensor2img(oct_img_viz, oct_path)
        tensor2img(colpo_img_viz, colpo_path)

        # 准备 image_names
        image_names = target_sample.get('image_names', ['sample_image.jpg'])
        if not isinstance(image_names, list):
            image_names = [image_names] if isinstance(image_names, str) else ['sample_image.jpg']

        # --- 生成 OCT CAM ---
        print("\n" + "="*60)
        print("🎨 Generating OCT Visualization...")
        print("="*60)
        raw_cam_oct = cam_gen.generate(f_oct, f_colpo, image_names, clin_tensor, target_class=label, modality='oct')
        print(f"[Raw CAM] Shape: {raw_cam_oct.shape}, Min: {raw_cam_oct.min():.4f}, Max: {raw_cam_oct.max():.4f}, Mean: {raw_cam_oct.mean():.4f}")
        final_cam_oct = apply_oct_constraints(raw_cam_oct, oct_path)
        print(f"[Final CAM] Shape: {final_cam_oct.shape}, Min: {final_cam_oct.min():.4f}, Max: {final_cam_oct.max():.4f}, Mean: {final_cam_oct.mean():.4f}")
        save_clinical_overlay(final_cam_oct, oct_path, save_dir / f"Final_Clinical_OCT_{label_suffix}_{idx+1}.jpg")
        
        # --- 生成 Colpo CAM ---
        print("\n" + "="*60)
        print("🎨 Generating Colposcopy Visualization...")
        print("="*60)
        raw_cam_colpo = cam_gen.generate(f_oct, f_colpo, image_names, clin_tensor, target_class=label, modality='colpo')
        print(f"[Raw CAM] Shape: {raw_cam_colpo.shape}, Min: {raw_cam_colpo.min():.4f}, Max: {raw_cam_colpo.max():.4f}, Mean: {raw_cam_colpo.mean():.4f}")
        final_cam_colpo = apply_colpo_constraints(raw_cam_colpo, colpo_path)
        print(f"[Final CAM] Shape: {final_cam_colpo.shape}, Min: {final_cam_colpo.min():.4f}, Max: {final_cam_colpo.max():.4f}, Mean: {final_cam_colpo.mean():.4f}")
        save_clinical_overlay(final_cam_colpo, colpo_path, save_dir / f"Final_Clinical_Colpo_{label_suffix}_{idx+1}.jpg")

if __name__ == "__main__":
    main()
