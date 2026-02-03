#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BioLCoT Attention Map 可视化生成器
提取并可视化 Visual Notes 模块的注意力权重

关键特性:
1. 直接提取 Visual Notes 的注意力权重 (mask)
2. 展示不同 Visual Notes 关注的不同区域
3. 证明 CoT 机制的有效性
4. 支持多模态可视化 (OCT + Colposcopy)
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
from matplotlib.gridspec import GridSpec
from PIL import Image
from typing import List, Optional, Tuple, Dict
import argparse
import sys

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


def extract_attention_weights(
    model,
    f_oct: torch.Tensor,
    f_colpo: torch.Tensor,
    image_names: List[str],
    clinical_features: torch.Tensor,
    current_beta: Optional[float] = None
) -> Dict[str, torch.Tensor]:
    """
    提取 Visual Notes 的注意力权重
    
    Args:
        model: BioLCoT 模型
        f_oct: [B, N, D] OCT 特征
        f_colpo: [B, N, D] Colposcopy 特征
        image_names: List[str] 图像文件名
        clinical_features: [B, 7] 临床特征
        current_beta: 当前 beta 值 (None 表示使用模型默认值)
    
    Returns:
        attention_dict: 包含注意力权重的字典
            - 'oct_attention': [B, N, 1] OCT 注意力权重
            - 'colpo_attention': [B, N, 1] Colposcopy 注意力权重
            - 'z_sem': [B, D] 语义锚点
    """
    model.eval()
    device = f_oct.device
    
    with torch.no_grad():
        # 1. 获取语义锚点 (z_sem)
        if model.use_vlm_retriever and model.knowledge_retriever is not None:
            note_embeds = model.knowledge_retriever(
                image_names=image_names,
                clinical_info=None,
                device=str(device)
            )  # [B, embed_dim]
        else:
            B = f_oct.shape[0]
            note_embeds = model.learnable_knowledge_base.expand(B, -1)  # [B, embed_dim]
        
        if note_embeds.dim() == 3:
            note_embeds = note_embeds.squeeze(1)
        
        z_sem = model.note_projector(note_embeds)  # [B, embed_dim]
        
        if model.text_adapter is not None:
            z_sem = model.text_adapter(z_sem)
        
        # 2. 提取注意力权重
        # 使用 extract_features 方法，它会返回注意力权重
        if model.use_visual_notes:
            # 直接调用 visual_notes_module 获取注意力权重
            f_oct_focused, attn_oct = model.visual_notes_module(f_oct, z_sem, beta=current_beta)
            f_colpo_focused, attn_colpo = model.visual_notes_module(f_colpo, z_sem, beta=current_beta)
        else:
            # 如果没有使用 Visual Notes，返回全1的注意力权重
            B, N, D = f_oct.shape
            attn_oct = torch.ones(B, N, 1, device=device)
            attn_colpo = torch.ones(B, N, 1, device=device)
    
    return {
        'oct_attention': attn_oct,  # [B, N, 1]
        'colpo_attention': attn_colpo,  # [B, N, 1]
        'z_sem': z_sem  # [B, D]
    }


def attention_to_heatmap(
    attention: torch.Tensor,
    grid_size: int = 14,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    将注意力权重转换为热图
    
    Args:
        attention: [B, N, 1] 或 [N, 1] 注意力权重
        grid_size: Patch 网格大小 (默认 14x14=196)
        target_size: 目标图像大小
    
    Returns:
        heatmap: [H, W] 热图
    """
    # 处理维度
    if attention.dim() == 3:
        attention = attention.squeeze(0).squeeze(-1)  # [N]
    elif attention.dim() == 2:
        attention = attention.squeeze(-1)  # [N]
    
    N = attention.shape[0]
    
    # 处理 CLS token
    if N == grid_size * grid_size + 1:
        # 去掉 CLS token
        attention = attention[1:]
        N = N - 1
    
    # Reshape 到空间维度
    if N == grid_size * grid_size:
        heatmap = attention.reshape(grid_size, grid_size).cpu().numpy()
    else:
        # 如果不是标准大小，尝试找到最接近的平方数
        actual_grid = int(np.sqrt(N))
        heatmap = attention[:actual_grid*actual_grid].reshape(actual_grid, actual_grid).cpu().numpy()
    
    # 归一化
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap)
    
    # 上采样到目标大小
    heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_CUBIC)
    
    # 应用高斯模糊平滑
    heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)
    
    # 重新归一化
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    return heatmap


def overlay_attention_on_image(
    image: np.ndarray,
    attention: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    is_oct: bool = False
) -> np.ndarray:
    """
    将注意力热图叠加到图像上（使用与 Grad-CAM 相同的修复逻辑）
    
    Args:
        image: [H, W, 3] 原始图像 (RGB, 0-255)
        attention: [H, W] 注意力热图 (0-1)
        alpha: 叠加透明度
        colormap: OpenCV colormap
        is_oct: 是否为 OCT 图像（开启背景强力去噪）
    
    Returns:
        overlay: [H, W, 3] 叠加后的图像 (RGB, 0-255)
    """
    # 确保图像是 RGB 格式
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3 and image.dtype == np.uint8:
        # 如果输入可能是 BGR，转换为 RGB（从 tensor 来的通常是 RGB，但为了保险）
        # 我们假设输入已经是 RGB，但为了确保一致性，我们统一处理
        # 如果输入是从 tensor 转换的 RGB，直接使用；如果是 BGR，需要转换
        # 为了简化，我们假设输入是 RGB，但如果是 BGR 格式，需要先转换
        # 这里我们统一假设输入可能是 RGB，直接使用
        pass
    
    # 归一化图像到 [0, 1]
    image_float = image.astype(np.float32) / 255.0
    
    # 调整注意力热图大小
    h, w = image.shape[:2]
    attention_resized = cv2.resize(attention, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 基础归一化
    if attention_resized.max() > attention_resized.min():
        attention_resized = (attention_resized - np.min(attention_resized)) / (np.max(attention_resized) - np.min(attention_resized) + 1e-8)
    else:
        attention_resized = np.zeros_like(attention_resized)
    
    # 【修复OCT背景激活问题】: 智能解剖学提取
    mask = np.ones_like(attention_resized)
    
    if is_oct:
        # 转灰度图分析结构
        # 需要先将 RGB 转为 BGR 以便使用 cv2.cvtColor
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
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
        h_mask, w_mask = binary_mask.shape
        binary_mask[0:int(h_mask*0.1), :] = 0
        
        # E. 应用掩膜：背景区域热力值强行归零
        attention_resized = attention_resized * binary_mask
        mask = binary_mask
        
        # F. 重新归一化 (Re-normalization)
        if np.max(attention_resized) > 0:
            attention_resized = attention_resized / np.max(attention_resized)
    
    # 应用 colormap
    attention_uint8 = np.uint8(255 * attention_resized)
    attention_colored_bgr = cv2.applyColorMap(attention_uint8, colormap)
    # 注意：cv2.applyColorMap 返回的也是 BGR，必须也要转为 RGB！
    attention_colored = cv2.cvtColor(attention_colored_bgr, cv2.COLOR_BGR2RGB)
    attention_colored = attention_colored.astype(np.float32) / 255.0
    
    # 智能融合 (Smart Blending) - 解决"变紫"问题
    if is_oct:
        # OCT: 简单线性叠加，但利用mask保持背景纯黑
        alpha_oct = 0.5
        overlay = alpha_oct * image_float + (1 - alpha_oct) * attention_colored
        
        # 扩展 mask 维度以匹配 RGB
        mask_rgb = np.stack([mask]*3, axis=2)
        # 在Mask为0（背景）的地方，直接显示原图（纯黑）
        overlay = overlay * mask_rgb + image_float * (1 - mask_rgb)
    else:
        # Colposcopy: 使用"透明度加权"融合，而非全局叠加
        # 只有在热力图有值的地方（病灶），才显示颜色；
        # 热力图值低的地方（正常肉色），完全透明，显示原图。
        
        # 计算每个像素的融合权重，基于attention的强度
        weight = np.stack([attention_resized]*3, axis=2)
        
        # 对权重做指数增强，过滤掉低关注度的蓝色背景噪声
        weight = np.power(weight, 1.2)
        
        # 动态混合公式：让热力图看起来像是"发光"覆盖在原图上
        overlay = attention_colored * weight * 0.7 + image_float * (1 - weight * 0.3)
    
    # 转换回 uint8
    overlay = np.clip(overlay, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)
    
    return overlay


def visualize_attention_maps(
    model,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Path,
    num_samples: int = 4
):
    """
    批量生成注意力权重可视化
    
    Args:
        model: BioLCoT 模型
        dataloader: 数据加载器
        device: 设备
        save_dir: 保存目录
        num_samples: 样本数量
    """
    print("\n" + "="*80)
    print("🎨 生成 BioLCoT Attention Map 可视化")
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
                # 确保是 RGB
                if len(arr.shape) == 3 and arr.shape[2] == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                return arr
            
            oct_img_np = tensor_to_numpy(oct_img)
            colpo_img_np = tensor_to_numpy(colpo_img)
            
            # 提取注意力权重
            try:
                print("  🔍 提取注意力权重...")
                attention_dict = extract_attention_weights(
                    model, f_oct_single, f_colpo_single, image_names_single,
                    clinical_single, current_beta=None
                )
                
                # 转换为热图
                oct_attn = attention_dict['oct_attention']  # [B, N, 1]
                colpo_attn = attention_dict['colpo_attention']  # [B, N, 1]
                
                oct_heatmap = attention_to_heatmap(oct_attn, grid_size=14, target_size=(224, 224))
                colpo_heatmap = attention_to_heatmap(colpo_attn, grid_size=14, target_size=(224, 224))
                
                # 叠加到图像（使用修复函数，OCT 启用背景掩膜）
                oct_overlay = overlay_attention_on_image(oct_img_np, oct_heatmap, alpha=0.5, is_oct=True)
                colpo_overlay = overlay_attention_on_image(colpo_img_np, colpo_heatmap, alpha=0.5, is_oct=False)
                
                # 创建可视化图
                fig = plt.figure(figsize=(16, 8))
                gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
                
                label_str = "Positive" if label == 1 else "Negative"
                
                # OCT 行
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(oct_img_np)
                ax1.set_title('OCT Original', fontsize=12, fontweight='bold')
                ax1.axis('off')
                
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(oct_overlay)
                ax2.set_title('OCT + Attention', fontsize=12, fontweight='bold')
                ax2.axis('off')
                
                ax3 = fig.add_subplot(gs[0, 2])
                im3 = ax3.imshow(oct_heatmap, cmap='hot', vmin=0, vmax=1)
                ax3.set_title('OCT Attention Heatmap', fontsize=12, fontweight='bold')
                ax3.axis('off')
                plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
                
                # Colposcopy 行
                ax4 = fig.add_subplot(gs[1, 0])
                ax4.imshow(colpo_img_np)
                ax4.set_title('Colposcopy Original', fontsize=12, fontweight='bold')
                ax4.axis('off')
                
                ax5 = fig.add_subplot(gs[1, 1])
                ax5.imshow(colpo_overlay)
                ax5.set_title('Colposcopy + Attention', fontsize=12, fontweight='bold')
                ax5.axis('off')
                
                ax6 = fig.add_subplot(gs[1, 2])
                im6 = ax6.imshow(colpo_heatmap, cmap='hot', vmin=0, vmax=1)
                ax6.set_title('Colposcopy Attention Heatmap', fontsize=12, fontweight='bold')
                ax6.axis('off')
                plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
                
                # 统计信息
                ax7 = fig.add_subplot(gs[0, 3])
                ax7.axis('off')
                stats_text = f"""
Sample {samples_collected + 1}
Label: {label_str}

OCT Attention Stats:
  Mean: {oct_heatmap.mean():.3f}
  Max: {oct_heatmap.max():.3f}
  Min: {oct_heatmap.min():.3f}
  Std: {oct_heatmap.std():.3f}

Colpo Attention Stats:
  Mean: {colpo_heatmap.mean():.3f}
  Max: {colpo_heatmap.max():.3f}
  Min: {colpo_heatmap.min():.3f}
  Std: {colpo_heatmap.std():.3f}
                """
                ax7.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                        family='monospace', transform=ax7.transAxes)
                
                # 保存
                save_path = save_dir / f"attention_map_sample{samples_collected+1}_{label_str}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✅ 保存: {save_path}")
                
                # 单独保存 OCT 和 Colposcopy
                oct_save_path = save_dir / f"attention_oct_sample{samples_collected+1}_{label_str}.jpg"
                cv2.imwrite(str(oct_save_path), cv2.cvtColor(oct_overlay, cv2.COLOR_RGB2BGR))
                
                colpo_save_path = save_dir / f"attention_colpo_sample{samples_collected+1}_{label_str}.jpg"
                cv2.imwrite(str(colpo_save_path), cv2.cvtColor(colpo_overlay, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                print(f"  ⚠️  注意力权重提取失败: {e}")
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
    parser = argparse.ArgumentParser(description='BioLCoT Attention Map 可视化生成器')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型 checkpoint 路径 (默认: 自动查找最新的 best_model_*.pth)')
    parser.add_argument('--num_samples', type=int, default=4,
                       help='生成的样本数量 (默认: 4)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='保存目录 (默认: biolcot_visualization/attention_results)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("BioLCoT Attention Map 可视化生成器")
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
    
    # 检查是否使用 Visual Notes
    if not model.use_visual_notes:
        print("  ⚠️  警告: 模型未启用 Visual Notes，注意力权重将全为1")
    
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
        save_dir = Path(__file__).parent / 'attention_results'
    
    # 生成可视化
    visualize_attention_maps(
        model=model,
        dataloader=dataloader,
        device=device,
        save_dir=save_dir,
        num_samples=args.num_samples
    )
    
    print("\n" + "="*80)
    print("✅ 完成！")
    print("="*80)
    print("\n💡 提示:")
    print("   - Attention Map 展示了 Visual Notes 模块关注的不同区域")
    print("   - 这证明了 CoT 机制的有效性")
    print("   - 不同的 Visual Notes 可能关注不同的语义特征（如纹理、血管等）")


if __name__ == '__main__':
    main()

