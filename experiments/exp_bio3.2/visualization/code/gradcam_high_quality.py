#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2: High-Quality Grad-CAM Visualization (GitHub pytorch-grad-cam style)

Features:
1. Uses aug_smooth=True and eigen_smooth=True for cleaner CAMs (as per GitHub docs)
2. Enhanced contrast and thresholding for visible activation regions
3. OCT: Shows ALL slices matching the number of OCT frames
4. Colposcopy: 3×4 figure (3 phases × 4 columns: Pos Orig|CAM, Neg Orig|CAM)

Based on: https://github.com/jacobgil/pytorch-grad-cam
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import cv2

from config import BioCOT_v3_2_Config
from models.bio_cot_v3_2 import create_bio_cot_v3_2
from data.dataset_v3_2 import FiveCentersMultimodalDatasetV3_2

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_transform_vit_robust(tensor):
    """
    鲁棒的 ViT 特征重塑函数（针对 Bio-COT 3.2 的 HierarchicalViT）。
    自动计算 Patch 网格大小，正确处理 CLS token。
    
    Args:
        tensor: [Batch, N_tokens, Dim] 例如 [1, 197, 768] (196 patches + 1 CLS)
    
    Returns:
        [Batch, Dim, H, W] 例如 [1, 768, 14, 14]
    """
    # 1. 去掉 CLS token (第一个 token)
    # Bio-COT 3.2 使用标准 ViT，第一个 token 是 CLS
    has_cls_token = True
    if has_cls_token:
        result = tensor[:, 1:, :]  # [B, N_patches, Dim]
    else:
        result = tensor
    
    # 2. 动态计算网格大小
    n_patches = result.size(1)
    grid_size = int(np.sqrt(n_patches))
    
    # 验证是否为完全平方数
    if grid_size * grid_size != n_patches:
        raise ValueError(
            f"Patch count {n_patches} is not a perfect square! "
            f"Expected square number (e.g., 196=14x14), got {n_patches}. "
            f"Please check your ViT configuration."
        )
    
    # 3. 维度变换: [B, N, C] -> [B, C, N] -> [B, C, H, W]
    result = result.transpose(1, 2)  # [B, C, N]
    result = result.reshape(tensor.size(0), tensor.size(2), grid_size, grid_size)  # [B, C, H, W]
    
    return result


class ColpoOnlyWrapper(nn.Module):
    """Wrapper: single colposcopy image -> logits [B,2]."""
    def __init__(self, base_model: nn.Module, device: torch.device):
        super().__init__()
        self.model = base_model
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        f_oct = x.to(self.device)
        f_colpo = x.to(self.device)
        image_names = [f"colpo_{i}.jpg" for i in range(B)]
        clinical_features = torch.zeros(B, 7, device=self.device)
        out = self.model(
            f_oct=f_oct, f_colpo=f_colpo, image_names=image_names,
            clinical_info=None, center_labels=None, clinical_features=clinical_features,
            return_loss_components=False, current_beta=0.1,
        )
        return out["logits"]


class OctOnlyWrapper(nn.Module):
    """Wrapper: single OCT slice -> logits [B,2]."""
    def __init__(self, base_model: nn.Module, device: torch.device):
        super().__init__()
        self.model = base_model
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        f_oct = x.to(self.device)
        f_colpo = x.to(self.device)
        image_names = [f"oct_{i}.png" for i in range(B)]
        clinical_features = torch.zeros(B, 7, device=self.device)
        out = self.model(
            f_oct=f_oct, f_colpo=f_colpo, image_names=image_names,
            clinical_info=None, center_labels=None, clinical_features=clinical_features,
            return_loss_components=False, current_beta=0.1,
        )
        return out["logits"]


def denorm_to_rgb(t: torch.Tensor) -> np.ndarray:
    """[3,H,W] -> float32 [H,W,3] in [0,1]."""
    arr = t.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return arr.astype(np.float32)


def enhance_cam(grayscale_cam: np.ndarray, threshold_percentile: float = 20.0) -> np.ndarray:
    """
    Enhance CAM contrast: threshold low values, normalize, apply gamma.
    This makes activation regions much more visible.
    """
    # Threshold: only keep top (100-threshold_percentile)% values
    threshold = np.percentile(grayscale_cam, threshold_percentile)
    cam_thresh = np.where(grayscale_cam >= threshold, grayscale_cam, 0)
    
    # Normalize
    if cam_thresh.max() > 0:
        cam_thresh = (cam_thresh - cam_thresh.min()) / (cam_thresh.max() + 1e-8)
    
    # Gamma correction to emphasize high activations
    cam_thresh = np.power(cam_thresh, 0.4)  # Lower gamma = more contrast
    
    return cam_thresh


def show_cam_on_image_enhanced(img: np.ndarray,
                              mask: np.ndarray,
                              use_rgb: bool = True,
                              colormap: int = cv2.COLORMAP_JET,
                              image_weight: float = 0.4) -> np.ndarray:
    """
    增强版热力图叠加函数：强制红色高亮病灶区域。
    
    关键修复：
    1. 强制归一化到 [0, 1]，确保最大值是 1.0 (最红)
    2. 阈值过滤低激活区域，让红色更集中
    3. 调整叠加权重，让热力图更鲜艳
    
    Args:
        img: 原图 [H, W, 3] in [0, 1]
        mask: CAM激活图 [H, W] in [0, 1]
        use_rgb: 是否使用RGB格式
        colormap: OpenCV颜色映射（JET=红色高激活）
        image_weight: 原图权重（越小，热力图越明显）
    
    Returns:
        叠加后的可视化图像 [H, W, 3] in [0, 1]
    """
    # 1. 强制归一化到 [0, 1]，确保最大值是 1.0 (最红)
    # 这是解决"颜色不够红"的关键！
    if mask.max() > mask.min():
        mask_normalized = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    else:
        mask_normalized = mask
    
    # 2. 阈值过滤：去掉低激活区域（< 0.2），让红色更集中
    # 这能让病灶区域更显眼，不会被背景噪声干扰
    mask_thresh = np.where(mask_normalized >= 0.2, mask_normalized, 0)
    
    # 3. 应用颜色映射（JET：蓝色=低激活，红色=高激活）
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_thresh), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    heatmap = np.float32(heatmap) / 255.0
    
    # 4. 叠加：在病灶区域（高激活）让热力图更不透明
    # image_weight=0.4 意味着：热力图占60%，原图占40%
    # 这样红色会更鲜艳，不会被原图的暗红色背景"吃掉"
    cam = (1 - image_weight) * heatmap + image_weight * img
    
    # 5. 再次归一化确保输出在 [0, 1]
    cam = np.clip(cam, 0, 1)
    
    return cam


def reshape_transform_vit_robust(tensor: torch.Tensor) -> torch.Tensor:
    """
    鲁棒的 ViT 特征重塑函数（针对 Bio-COT 3.2 的 HierarchicalViT）。
    自动计算 Patch 网格大小，正确处理 CLS token。
    
    Args:
        tensor: [Batch, N_tokens, Dim] 例如 [1, 197, 768] (196 patches + 1 CLS)
    
    Returns:
        [Batch, Dim, H, W] 例如 [1, 768, 14, 14]
    """
    # 1. 去掉 CLS token (第一个 token)
    # Bio-COT 3.2 使用标准 ViT，第一个 token 是 CLS
    has_cls_token = True
    if has_cls_token:
        result = tensor[:, 1:, :]  # [B, N_patches, Dim]
    else:
        result = tensor
    
    # 2. 动态计算网格大小
    n_patches = result.size(1)
    grid_size = int(np.sqrt(n_patches))
    
    # 验证是否为完全平方数
    if grid_size * grid_size != n_patches:
        raise ValueError(
            f"Patch count {n_patches} is not a perfect square! "
            f"Expected square number (e.g., 196=14x14), got {n_patches}. "
            f"Please check your ViT configuration."
        )
    
    # 3. 维度变换: [B, N, C] -> [B, C, N] -> [B, C, H, W]
    result = result.transpose(1, 2)  # [B, C, N]
    result = result.reshape(tensor.size(0), tensor.size(2), grid_size, grid_size)  # [B, C, H, W]
    
    return result

def find_cases(model, wrapper_class, dataset, device):
    """Find one positive and one negative case (prefer high confidence, but fallback to any)."""
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    wrapper = wrapper_class(model, device).to(device).eval()
    
    pos_case = None
    neg_case = None
    pos_fallback = None
    neg_fallback = None
    
    for batch in loader:
        label = int(batch["label"].item())
        colpos = batch["colposcopy_images"][0]
        img_t = colpos[0]
        inp = img_t.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = wrapper(inp)
            prob_pos = torch.softmax(logits, 1)[0, 1].item()
            pred = int(torch.argmax(logits, 1)[0].item())
        
        correct = (pred == label)
        
        # Fallback: any case with matching label
        if label == 1 and pos_fallback is None:
            pos_fallback = batch
        if label == 0 and neg_fallback is None:
            neg_fallback = batch
        
        # Preferred: correct prediction with reasonable confidence
        if label == 1 and correct and prob_pos >= 0.5 and pos_case is None:
            pos_case = batch
        if label == 0 and correct and prob_pos <= 0.5 and neg_case is None:
            neg_case = batch
    
    if pos_case is None:
        pos_case = pos_fallback
    if neg_case is None:
        neg_case = neg_fallback
    
    return pos_case, neg_case


def build_colpo_3x4_figure(model, device, pos_sample, neg_sample, save_path):
    """Build 3×4 colposcopy figure: 3 phases × (Pos Orig|CAM, Neg Orig|CAM)."""
    wrapper = ColpoOnlyWrapper(model, device).eval()
    target_layer = model.visual_encoder.vit.blocks[-1].norm1
    # 关键修复：使用鲁棒的 reshape_transform 处理 ViT 的序列特征
    cam = EigenCAM(
        model=wrapper,
        target_layers=[target_layer],
        reshape_transform=reshape_transform_vit_robust  # ViT 维度修复
    )
    
    phases = ["Raw", "Acetic", "Iodine"]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), dpi=300)
    
    def process_phase(sample, col_offset, tag):
        colpos = sample["colposcopy_images"][0]
        label = int(sample["label"].item())
        
        for i in range(3):
            img_t = colpos[i]
            inp = img_t.unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(label)]
            
            with torch.no_grad():
                logits = wrapper(inp)
                prob_pos = torch.softmax(logits, 1)[0, 1].item()
            
            # Generate CAM with smoothing
            # 对阴道镜，我们更希望 CAM 聚焦在最强的病灶区域，而不是大面积铺开。
            # 因此保留 aug_smooth 去噪，但关闭 eigen_smooth，避免过度平滑。
            grayscale_cam = cam(
                input_tensor=inp,
                targets=targets,
                aug_smooth=True,
                eigen_smooth=False
            )[0]
            
            # 强制拉伸对比度：确保高激活区域显示为深红色
            # 这是解决"颜色不够红"的关键步骤！
            if grayscale_cam.max() > grayscale_cam.min():
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
            
            # 增强对比度：阈值过滤 + Gamma校正
            grayscale_cam = enhance_cam(grayscale_cam, threshold_percentile=30.0)
            
            # 再次归一化确保在 [0, 1] 范围
            if grayscale_cam.max() > 0:
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() + 1e-8)
            
            rgb = denorm_to_rgb(img_t)
            # 使用增强版叠加函数：强制红色高亮病灶区域
            overlay = show_cam_on_image_enhanced(rgb, grayscale_cam, use_rgb=True, image_weight=0.4)
            
            axes[i, col_offset].imshow(rgb)
            axes[i, col_offset].set_title(f"{phases[i]} {tag}\nOrig", fontsize=10, fontweight='bold')
            axes[i, col_offset + 1].imshow(overlay)
            axes[i, col_offset + 1].set_title(f"{phases[i]} {tag}\nCAM (p={prob_pos:.2f})", fontsize=10, fontweight='bold')
            axes[i, col_offset].axis("off")
            axes[i, col_offset + 1].axis("off")
    
    process_phase(pos_sample, 0, "Pos")
    process_phase(neg_sample, 2, "Neg")
    
    plt.suptitle("Colposcopy Grad-CAM: Positive vs Negative Cases", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def build_oct_all_slices_figure(model, device, pos_sample, neg_sample, save_dir):
    """
    Build OCT figure showing ALL slices (matching OCT frame count).
    Creates a grid: 2 rows (Pos/Neg) × N columns (one per OCT slice).
    """
    wrapper = OctOnlyWrapper(model, device).eval()
    target_layer = model.visual_encoder.vit.blocks[-1].norm1
    cam = EigenCAM(model=wrapper, target_layers=[target_layer], reshape_transform=reshape_transform_vit)
    
    def process_case(sample, tag):
        oct_images = sample["oct_images"][0]  # [F,C,H,W]
        F = oct_images.size(0)
        label = int(sample["label"].item())
        
        # Create figure: 2 rows (Orig|CAM) × F columns (one per slice)
        fig, axes = plt.subplots(2, F, figsize=(F * 2, 4), dpi=300)
        if F == 1:
            axes = axes.reshape(2, 1)
        
        for f in range(F):
            img_t = oct_images[f]
            inp = img_t.unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(label)]
            
            with torch.no_grad():
                logits = wrapper(inp)
                prob_pos = torch.softmax(logits, 1)[0, 1].item()
            
            # Generate CAM with smoothing
            grayscale_cam = cam(
                input_tensor=inp,
                targets=targets,
                aug_smooth=True,
                eigen_smooth=True
            )[0]
            
            # Enhance contrast
            grayscale_cam = enhance_cam(grayscale_cam, threshold_percentile=15.0)
            
            rgb = denorm_to_rgb(img_t)
            overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True, image_weight=0.5)
            
            axes[0, f].imshow(rgb)
            axes[0, f].set_title(f"Slice {f+1}\nOrig", fontsize=8)
            axes[1, f].imshow(overlay)
            axes[1, f].set_title(f"Slice {f+1}\nCAM", fontsize=8)
            axes[0, f].axis("off")
            axes[1, f].axis("off")
        
        plt.suptitle(f"OCT {tag} Case: All {F} Slices (Orig | CAM)", fontsize=12, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        save_path = save_dir / f"OCT_{tag}_all_{F}_slices.png"
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return save_path
    
    save_dir.mkdir(parents=True, exist_ok=True)
    pos_path = process_case(pos_sample, "Pos")
    neg_path = process_case(neg_sample, "Neg")
    
    return pos_path, neg_path


def main():
    print("=" * 80)
    print("Bio-COT 3.2: High-Quality Grad-CAM Visualization")
    print("Using aug_smooth + eigen_smooth for cleaner activation maps")
    print("=" * 80)
    
    config = BioCOT_v3_2_Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("📦 Loading model...")
    model = create_bio_cot_v3_2(config).to(device).eval()
    
    # Load checkpoint
    ckpt_dir = Path(config.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = Path(__file__).resolve().parents[2] / ckpt_dir
    ckpts = sorted(ckpt_dir.glob("best_model*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        print(f"📥 Loading: {ckpts[0]}")
        state = torch.load(ckpts[0], map_location=device, weights_only=False)
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Checkpoint loaded.")
    else:
        print(f"⚠️ No checkpoint in {ckpt_dir}, using random weights.")
    
    # Load dataset
    print("📂 Loading dataset...")
    csv_path = Path(config.data_root) / "temp_val_labels.csv"
    dataset = FiveCentersMultimodalDatasetV3_2(csv_path=str(csv_path), data_root=str(config.data_root))
    
    # Find cases
    print("🔍 Finding positive & negative cases...")
    pos_sample, neg_sample = find_cases(model, ColpoOnlyWrapper, dataset, device)
    if pos_sample is None or neg_sample is None:
        print("❌ Could not find both cases. Exiting.")
        return
    
    figures_dir = Path(__file__).parent.parent / "figures"
    
    # Generate colposcopy 3×4 figure
    print("🎨 Generating colposcopy 3×4 figure (with aug_smooth + eigen_smooth)...")
    colpo_path = figures_dir / "Colpo_GradCAM_HighQuality_3x4.png"
    build_colpo_3x4_figure(model, device, pos_sample, neg_sample, colpo_path)
    print(f"✅ Saved: {colpo_path}")
    
    # ⚠️ OCT 全帧大图在当前环境下尺寸过大，容易触发 Matplotlib 限制。
    # 你已经可以使用 `gradcam_oct_all_slices.py` 生成逐帧 CAM PNG，
    # 更适合作为素材在 PPT / Illustrator 里手动排版成总图，这里不再自动拼接巨幅图像。
    print("\n" + "=" * 80)
    print("✅ High-quality colposcopy Grad-CAM figure generated successfully!")
    print("💡 For OCT per-slice CAMs, use: gradcam_oct_all_slices.py (已经成功生成逐帧 PNG)")
    print("=" * 80)


if __name__ == "__main__":
    main()

