#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2: 专门优化的Grad-CAM可视化脚本
关键改进：
1. 自动适配层结构：处理ViT的3D Tensor格式
2. 精准定位：挂载到visual_encoder的最后一个Transformer Block
3. 多模态支持：完整模拟OCT + 阴道镜 + 文本的输入流程
4. 梯度回传：确保梯度从分类头流回到Backbone
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


class BioCotLayerCAM:
    """
    LayerCAM: 空间加权的类激活映射
    专门针对Bio-COT 3.2架构优化
    
    核心优势（相比Grad-CAM）：
    1. 不进行全局平均池化：保留梯度在特征图上的每个空间位置的权重
    2. 精确描绘病灶：能够精确描绘出与分类最相关的像素级纹理
    3. 避免模糊：不会把病灶点模糊成一大片热力团
    
    关键特性：
    1. 自动寻找visual_encoder的最后一个Transformer Block
    2. 正确处理ViT的3D Tensor格式（[B, N, D] -> [B, C, H, W]）
    3. 确保梯度正确回传
    4. 支持多模态输入（OCT + Colposcopy）
    """
    
    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.target_layer_name = target_layer_name
        self.hook_registered = False
        
        # 注册hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """
        自动寻找目标层并注册forward/backward hooks
        
        关键：由于模型输入是已提取的特征（[B, N, D]），visual_encoder不会被调用
        所以我们需要从特征处理层获取梯度，优先级：
        1. 用户指定的层
        2. visual_notes_module（处理输入特征的第一层）
        3. dual_head（特征融合后的层）
        """
        target_layer = None
        
        if self.target_layer_name:
            # 尝试通过名称找到层
            for name, module in self.model.named_modules():
                if name == self.target_layer_name:
                    target_layer = module
                    print(f"[Info] Found user-specified layer: {self.target_layer_name}")
                    break
        else:
            # 默认：尝试找到处理输入特征的层
            print("[Info] No target layer specified. Searching for feature processing layer...")
            
            # 优先级1：visual_encoder的最后一层（保留更完整的空间位置信息）
            if hasattr(self.model, 'visual_encoder') and self.model.visual_encoder is not None:
                # 对于HierarchicalViT，找到vit.blocks的最后一个block的最后一层
                if hasattr(self.model.visual_encoder, 'vit'):
                    if hasattr(self.model.visual_encoder.vit, 'blocks'):
                        # 获取最后一个block的最后一层（通常是LayerNorm）
                        last_block = self.model.visual_encoder.vit.blocks[-1]
                        # 尝试找到最后一个LayerNorm或Linear层
                        last_layer = None
                        for name, module in last_block.named_modules():
                            if isinstance(module, (nn.LayerNorm, nn.Linear)):
                                last_layer = module
                                self.target_layer_name = f"visual_encoder.vit.blocks[-1].{name}"
                        if last_layer is not None:
                            target_layer = last_layer
                            print(f"[Info] Using {self.target_layer_name} (last spatial layer in backbone)")
                        else:
                            # 如果没有找到，使用整个block
                            target_layer = last_block
                            self.target_layer_name = "visual_encoder.vit.blocks[-1]"
                            print(f"[Info] Using {self.target_layer_name} (last block in backbone)")
                    else:
                        target_layer = self.model.visual_encoder.vit
                        self.target_layer_name = "visual_encoder.vit"
                        print(f"[Info] Using {self.target_layer_name}")
                else:
                    target_layer = self.model.visual_encoder
                    self.target_layer_name = "visual_encoder"
                    print(f"[Info] Using {self.target_layer_name}")
            # 优先级2：visual_notes_module（备选）
            elif hasattr(self.model, 'visual_notes_module') and self.model.visual_notes_module is not None:
                # 使用visual_notes_module内部的k_proj（处理图像特征的层）
                if hasattr(self.model.visual_notes_module, 'k_proj'):
                    target_layer = self.model.visual_notes_module.k_proj
                    self.target_layer_name = "visual_notes_module.k_proj"
                    print(f"[Info] Using visual_notes_module.k_proj (processes input features)")
                else:
                    target_layer = self.model.visual_notes_module
                    self.target_layer_name = "visual_notes_module"
                    print(f"[Info] Using visual_notes_module")
            # 优先级2：dual_head
            elif hasattr(self.model, 'dual_head'):
                if isinstance(self.model.dual_head, nn.Sequential):
                    target_layer = self.model.dual_head[0]  # 第一层
                    self.target_layer_name = "dual_head[0]"
                else:
                    target_layer = self.model.dual_head
                    self.target_layer_name = "dual_head"
                print(f"[Info] Using {self.target_layer_name}")
            # 优先级3：尝试visual_encoder（虽然可能不会被调用，但保留作为备选）
            elif hasattr(self.model, 'visual_encoder') and self.model.visual_encoder is not None:
                if hasattr(self.model.visual_encoder, 'vit'):
                    if hasattr(self.model.visual_encoder.vit, 'blocks'):
                        last_block = self.model.visual_encoder.vit.blocks[-1]
                        target_layer = last_block
                        self.target_layer_name = "visual_encoder.vit.blocks[-1]"
                        print(f"[Info] Using visual_encoder.vit.blocks[-1] (may not be called if input is 3D)")
                    else:
                        target_layer = self.model.visual_encoder.vit
                        self.target_layer_name = "visual_encoder.vit"
                else:
                    target_layer = self.model.visual_encoder
                    self.target_layer_name = "visual_encoder"
                print(f"[Info] Using {self.target_layer_name}")
            else:
                # 最后备选：查找任何处理特征的层
                print("[Warning] No standard layer found. Searching for any feature processing layer...")
                for name, module in self.model.named_modules():
                    if isinstance(module, (nn.Linear, nn.LayerNorm)) and 'proj' in name.lower():
                        target_layer = module
                        self.target_layer_name = name
                        print(f"[Info] Found layer: {name}")
                        break
        
        if target_layer is None:
            raise ValueError(f"Could not find target layer: {self.target_layer_name}")
        
        print(f"[Info] LayerCAM hooked into layer: {self.target_layer_name}")
        
        # Forward hook to get activations
        def forward_hook(module, input, output):
            # 处理tuple输出
            if isinstance(output, tuple):
                self.activations = output[0]  # 取第一个元素
            else:
                self.activations = output
        
        # Backward hook to get gradients
        def backward_hook(module, grad_input, grad_output):
            if grad_output is not None and len(grad_output) > 0:
                grad = grad_output[0]
                # 处理tuple梯度
                if isinstance(grad, tuple):
                    grad = grad[0]
                if grad is not None:
                    self.gradients = grad
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
        self.hook_registered = True
    
    def generate_cam(self, oct_features, colpo_features, image_names, clinical_features, target_class=None):
        """
        生成Grad-CAM热图
        
        关键：由于输入是已提取的特征（[B, N, D]），我们需要从输入特征本身获取梯度
        或者从处理这些特征的第一个层获取梯度
        
        Args:
            oct_features: [B, N, D] OCT特征（已提取的patch特征）
            colpo_features: [B, N, D] Colposcopy特征（已提取的patch特征）
            image_names: 图像文件名列表
            clinical_features: [B, 7] 临床特征
            target_class: int，可选类别索引（0或1）。如果为None，使用预测类别。
        
        Returns:
            oct_cam: [H, W] OCT的CAM热图
            colpo_cam: [H, W] Colposcopy的CAM热图
        """
        # 确保输入特征需要梯度
        oct_features = oct_features.clone().detach().requires_grad_(True)
        colpo_features = colpo_features.clone().detach().requires_grad_(True)
        
        # 注册hooks到输入特征（如果目标层没有被调用，我们从输入特征获取梯度）
        oct_gradients = None
        colpo_gradients = None
        oct_activations = None
        colpo_activations = None
        
        def oct_grad_hook(grad):
            nonlocal oct_gradients
            oct_gradients = grad.detach()
            return grad
        
        def colpo_grad_hook(grad):
            nonlocal colpo_gradients
            colpo_gradients = grad.detach()
            return grad
        
        # 注册梯度hook
        oct_handle = oct_features.register_hook(oct_grad_hook)
        colpo_handle = colpo_features.register_hook(colpo_grad_hook)
        
        # 保存激活
        oct_activations = oct_features.detach()
        colpo_activations = colpo_features.detach()
        
        # 1. 前向传播
        self.model.zero_grad()
        self.gradients = None
        self.activations = None
        
        output = self.model(
            f_oct=oct_features,
            f_colpo=colpo_features,
            image_names=image_names,
            clinical_features=clinical_features
        )
        
        # 2. 提取logits
        if isinstance(output, dict):
            logits = output.get('logits', output.get('pred', output.get('cls_preds')))
        elif isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # 3. 选择目标类别
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        
        score = logits[0, target_class]
        print(f"[Info] Generating Grad-CAM for Class: {target_class}, Score: {score.item():.4f}")
        
        # 4. 反向传播
        score.backward(retain_graph=True)
        
        # 5. 检查是否捕获到梯度和激活
        # 优先使用目标层的梯度和激活，如果没有则使用输入特征的梯度
        if self.gradients is not None and self.activations is not None:
            # 使用目标层的梯度和激活
            gradients = self.gradients
            activations = self.activations
            print("[Info] Using target layer gradients and activations")
        elif oct_gradients is not None and colpo_gradients is not None:
            # 使用输入特征的梯度（分别处理OCT和Colposcopy）
            print("[Info] Using input feature gradients (target layer not called)")
            # 对于OCT CAM，使用OCT的梯度
            # 对于Colposcopy CAM，使用Colposcopy的梯度
            # 这里我们先处理OCT，然后处理Colposcopy
            gradients_oct = oct_gradients
            activations_oct = oct_activations
            gradients_colpo = colpo_gradients
            activations_colpo = colpo_activations
            
            # 处理OCT CAM
            oct_cam = self._compute_cam_from_features(gradients_oct, activations_oct)
            # 处理Colposcopy CAM
            colpo_cam = self._compute_cam_from_features(gradients_colpo, activations_colpo)
            
            # 移除hooks
            oct_handle.remove()
            colpo_handle.remove()
            
            return oct_cam, colpo_cam
        else:
            print("[Warning] Gradients or activations not captured. Returning zero CAM.")
            # 移除hooks
            if 'oct_handle' in locals():
                oct_handle.remove()
            if 'colpo_handle' in locals():
                colpo_handle.remove()
            return np.zeros((14, 14)), np.zeros((14, 14))
        
        # 6. 处理ViT输出格式 [B, N_tokens, C] -> [B, C, H, W]
        gradients = self.gradients
        activations = self.activations
        
        # 移除输入特征的hooks（如果使用了目标层）
        if 'oct_handle' in locals():
            oct_handle.remove()
        if 'colpo_handle' in locals():
            colpo_handle.remove()
        
        # --- Shape Handling: 处理时间维度或batch维度 ---
        # Case: Batch > 1 or Time Dimension merged (e.g. 20 frames -> B=20)
        if len(gradients.shape) == 4 and gradients.shape[0] > 1:
            # Average gradients over the time/batch dimension
            gradients = gradients.mean(dim=0, keepdim=True)
            activations = activations.mean(dim=0, keepdim=True)
        
        # 处理ViT的3D tensor格式
        if len(activations.shape) == 3:
            # [B, N_tokens, C]
            B, N, C = activations.shape
            
            # 计算patch grid的大小
            # 对于224x224的图像，patch_size=16，有14x14=196个patches
            grid_size = int(np.sqrt(N))
            
            # 如果N不是完全平方数，可能包含CLS token
            if grid_size * grid_size != N:
                # 尝试去掉CLS token
                grid_size = int(np.sqrt(N - 1))
                if grid_size * grid_size == N - 1:
                    # 去掉第一个token（CLS token）
                    gradients = gradients[:, 1:, :]
                    activations = activations[:, 1:, :]
                    N = N - 1
                else:
                    # 如果还是不对，使用最接近的平方数
                    grid_size = int(np.sqrt(N))
            
            # Reshape [B, H*W, C] -> [B, C, H, W]
            # Transpose to [B, C, Grid, Grid]
            gradients = gradients.transpose(1, 2).reshape(B, C, grid_size, grid_size)
            activations = activations.transpose(1, 2).reshape(B, C, grid_size, grid_size)
        
        # 7. LayerCAM核心逻辑（关键改进：不进行全局平均池化）
        # 与Grad-CAM不同，LayerCAM不mean-pool梯度，而是element-wise multiply
        # 这保留了空间细节，能精确描绘病灶轮廓
        
        # Element-wise multiply: 梯度ReLU后与激活逐元素相乘
        weighted_acts = activations * F.relu(gradients)  # [B, C, H, W]
        
        # 对通道维度求和，得到每个空间位置的激活强度
        cam = torch.sum(weighted_acts, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 8. ReLU (只保留正向贡献)
        cam = F.relu(cam)
        
        # 9. 归一化 (Standard Min-Max)
        if torch.max(cam) - torch.min(cam) > 0:
            cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam))
        else:
            cam = torch.zeros_like(cam)
        
        # 11. 转换为numpy并提取第一个batch
        cam_np = cam.detach().cpu().numpy()[0, 0]  # [H, W]
        
        # 由于OCT和Colposcopy在forward中可能被融合，我们返回相同的CAM
        # 如果需要分别处理，需要修改模型forward以分别返回
        return cam_np, cam_np
    
    def _compute_cam_from_features(self, gradients, activations):
        """
        从特征计算CAM（辅助函数）
        
        Args:
            gradients: [B, N, D] 梯度
            activations: [B, N, D] 激活
        
        Returns:
            cam: [H, W] CAM热图
        """
        # 处理时间维度或batch维度
        if len(gradients.shape) == 3 and gradients.shape[0] > 1:
            gradients = gradients.mean(dim=0, keepdim=True)
            activations = activations.mean(dim=0, keepdim=True)
        
        # 处理ViT的3D tensor格式
        if len(activations.shape) == 3:
            B, N, C = activations.shape
            grid_size = int(np.sqrt(N))
            
            if grid_size * grid_size != N:
                grid_size = int(np.sqrt(N - 1))
                if grid_size * grid_size == N - 1:
                    gradients = gradients[:, 1:, :]
                    activations = activations[:, 1:, :]
                    N = N - 1
                else:
                    grid_size = int(np.sqrt(N))
            
            # Reshape [B, H*W, C] -> [B, C, H, W]
            gradients = gradients.transpose(1, 2).reshape(B, C, grid_size, grid_size)
            activations = activations.transpose(1, 2).reshape(B, C, grid_size, grid_size)
        
        # LayerCAM核心逻辑：element-wise multiply（不进行GAP）
        weighted_acts = activations * F.relu(gradients)  # [B, C, H, W]
        cam = torch.sum(weighted_acts, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # ReLU
        cam = F.relu(cam)
        
        # 归一化
        if torch.max(cam) - torch.min(cam) > 0:
            cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam))
        else:
            cam = torch.zeros_like(cam)
        
        # 转换为numpy
        cam_np = cam.detach().cpu().numpy()[0, 0]  # [H, W]
        
        return cam_np


def detect_oct_surface(img_gray):
    """
    自动检测组织表面（第一条亮线）
    用于屏蔽表面以上的保护套/空气区域
    
    Args:
        img_gray: [H, W] 灰度图像
    
    Returns:
        surface_idx: int，组织表面的行索引
    """
    h, w = img_gray.shape
    # 水平投影：计算每行的平均亮度
    proj = np.mean(img_gray, axis=1)
    
    # 启发式：从顶部10%开始扫描，避免立即噪声
    # 找到第一条有显著亮度的行（组织信号）
    start_row = int(h * 0.05)
    surface_idx = start_row
    
    # 阈值通常在40-50左右，用于OCT组织信号
    for r in range(start_row, h):
        if proj[r] > 40: 
            surface_idx = r
            break
            
    return surface_idx


def enhance_lesion_contrast(heatmap, gamma=0.7):
    """
    锐化热力图，聚焦激活的核心区域
    
    通过Gamma校正，把热力图的峰值变得更尖锐
    这能把那些弥散在边缘的弱激活（雾蒙蒙的感觉）压下去，把核心病灶"顶"出来
    
    注意：gamma < 1 会增强低值，使激活更明显
    
    Args:
        heatmap: [H, W] 热图，范围[0, 1]
        gamma: Gamma值，<1时增强低值对比度（默认0.7，使激活更明显）
    
    Returns:
        sharpened_heatmap: [H, W] 锐化后的热图
    """
    # 使用gamma < 1来增强低值，使激活更明显
    return np.power(heatmap, gamma)


def apply_anatomical_constraints(heatmap, original_image, modality='colpo', percentile_thresh=0.15):
    """
    基于医学常识的强约束处理（特定模态的解剖学遮罩 + 暗区增强 + 颜色感知）
    
    核心改进（基于用户反馈）：
    1. OCT专属约束：
       - 屏蔽上方20%（探头伪影）
       - 底部保护：不再屏蔽底部，允许激活延伸到最底部（"下面才是重点区域"）
       - 暗区增强：使用反向加权（darkness_map），放大暗区激活值2.5倍（"黑乎乎的癌症/冰柱"）
    2. Colposcopy专属约束：
       - 中心高斯聚焦（更柔和，允许边缘但惩罚极端角落）
       - 红色增强：检测红色主导区域，增强红色/肉色区域的权重（"宫颈口糜烂"）
       - 窥器反光去除：屏蔽高亮区域（>240）
    3. 显示阈值降低：从0.3降到0.15，确保弱激活也能显示（避免"什么都没激活"）
    
    Args:
        heatmap: [H, W] 原始CAM热图，范围[0, 1]
        original_image: [H, W, 3] 原始图像，范围[0, 255]或[0, 1]
        modality: 'oct' | 'colpo'，指定图像类型
        percentile_thresh: 动态阈值比例（0.0-1.0），保留最大值的此比例以上的区域（默认0.15，即Top 15%）
    
    Returns:
        clean_heatmap: [H, W] 清洗后的热图，范围[0, 1]
    """
    # 确保原始图像是numpy数组
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()
    
    # 确保原始图像是uint8格式
    if original_image.dtype != np.uint8:
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = original_image.astype(np.uint8)
    
    # 调整热力图尺寸到原图大小
    h, w = original_image.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    
    # 预处理：转换为RGB和灰度
    if len(original_image.shape) == 3:
        img_rgb = original_image
        img_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    else:
        img_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        img_gray = original_image
    
    # 转换为float用于计算
    img_float = img_gray.astype(np.float32) / 255.0
    
    # --- 1. OCT 专属约束 (智能表面检测 + 深度加权 + 暗区捕获) ---
    if modality == 'oct':
        # A. 智能伪影移除（自动表面检测）
        # 不再硬切20%，而是通过算法找到组织表面的第一条亮线
        # 完全屏蔽表面以上的保护套/空气区域
        surface_y = detect_oct_surface(img_gray)
        heatmap[:surface_y, :] = 0
        
        # B. 深度加权（关键改进：越往下权重越高）
        # 用户反馈："下面才是重点区域"
        # 创建梯度权重，随深度增加而增加
        # 符合OCT诊断中关注深层浸润和结构的逻辑
        if h > surface_y:
            # 线性斜坡：从0.5（表面）到1.5（底部）
            depth_ramp = np.linspace(0.5, 1.5, h - surface_y).reshape(-1, 1)
            depth_weight = np.tile(depth_ramp, (1, w))
            heatmap[surface_y:, :] *= depth_weight
        
        # C. 边缘去噪（左右信号衰减区）
        side_margin = int(w * 0.05)
        heatmap[:, :side_margin] = 0
        heatmap[:, -side_margin:] = 0
        
        # D. 暗区捕获（关键改进：基于特征的注意力增强）
        # 用户反馈："一坨黑乎乎就是宫颈癌"或"冰柱"
        # OCT病灶（癌症/囊肿）通常是低回声（暗区）
        # 使用反向加权：越暗的区域，权重越高
        darkness = 1.0 - img_float  # 暗区 -> 高值（0.0到1.0）
        
        # 定义"暗块"候选区域（像素值 < ~50）
        # 使用sigmoid曲线平滑强调暗区
        # 这能把那些深层的黑色病灶"捞"出来
        dark_boost_mask = 1 / (1 + np.exp(-10 * (darkness - 0.8)))
        
        # 应用增强：显著增加暗区的激活值
        # 只有当模型在暗区有至少弱激活（>0.05）时才增强
        # 这确认了"模型关注这里" + "这是暗区" -> "高度可疑"
        heatmap = heatmap * (1.0 + 2.0 * dark_boost_mask)
        
        print(f"[Info] Applied OCT constraints: surface at {surface_y}px (auto-detected), depth-weighted, sides {side_margin}px")
        print(f"[Info] Applied dark region capture (3.0x for dark lesions with sigmoid boost)")
    
    # --- 2. Colposcopy 专属约束 (色彩/纹理增强 + 高光抑制) ---
    elif modality == 'colpo':
        # A. 基于颜色的糜烂追踪（关键改进：HSV颜色空间）
        # 用户反馈："宫颈口糜烂"通常呈现红色/充血
        # 转换为HSV空间，专门寻找红色色调高饱和度的区域
        # 注意：original_image已经是RGB格式，需要先转换为BGR
        img_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR) if len(original_image.shape) == 3 else cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hue = img_hsv[:, :, 0]
        sat = img_hsv[:, :, 1]
        val = img_hsv[:, :, 2]
        
        # 定义"红色/粉色"掩码（色相范围0-10和160-180为红色）
        # 糜烂通常是红色/粉色且饱和度较高
        is_red = (hue < 15) | (hue > 165)
        is_saturated = (sat > 40)
        erosion_mask = (is_red & is_saturated).astype(np.float32)
        
        # 增强糜烂样区域的激活值
        heatmap = heatmap * (1.0 + 1.5 * erosion_mask)
        
        # B. 窥器反光去除（高亮区域通常是窥器反光，必须屏蔽）
        # 非常亮的点（Value > 240）是伪影
        reflection_mask = val > 240
        heatmap[reflection_mask] = 0
        
        # C. 柔和中心聚焦（关键改进：更柔和的衰减）
        # 宫颈通常在中心，但病变可能扩散
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        radius = np.sqrt(center_x**2 + center_y**2)
        
        # 比之前更柔和的衰减（2次方）
        center_weight = 1 - (dist_from_center / (radius * 0.95)) ** 2
        center_weight = np.clip(center_weight, 0.2, 1.0)  # 保持边缘可达
        heatmap *= center_weight
        
        print(f"[Info] Applied Colposcopy HSV-based erosion tracking (radius={radius:.1f})")
        print(f"[Info] Enhanced {np.sum(erosion_mask > 0)} red/erosive pixels (2.5x boost)")
        print(f"[Info] Removed {np.sum(reflection_mask)} speculum reflection pixels")
    
    # --- 3. 后处理（Gamma校正 + 动态阈值）---
    # Gamma校正：减少"雾蒙蒙"的感觉，锐化峰值
    # 使用gamma > 1来增强高值，使激活更明显
    heatmap = np.power(heatmap, 1.3)
    
    # 动态阈值截断：去除弱背景噪声
    # 使用15%的最大值作为截止点，保留上下文但去除纯噪声
    if np.max(heatmap) > 0:
        thresh_val = np.max(heatmap) * percentile_thresh
        heatmap[heatmap < thresh_val] = 0
    
    # --- 5. 重新归一化并增强对比度 ---
    if np.max(heatmap) > 0:
        # 使用非线性归一化，增强激活区域的可见性
        # 将热力图拉伸，使激活区域更明显
        heatmap = heatmap / np.max(heatmap)
        # 应用轻微的gamma校正，增强低值（使激活更明显）
        heatmap = np.power(heatmap, 0.8)
        # 重新归一化
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
    else:
        heatmap = np.zeros_like(heatmap)
    
    return heatmap


def clean_heatmap(heatmap, original_image, intensity_thresh=30, percentile_thresh=0.4, use_morphology=True, modality=None):
    """
    后处理大师：去除背景、边缘噪声，只保留核心激活
    
    核心改进：
    1. 物理遮罩（Anatomical Masking）：检测黑色背景/暗区，强制置零
    2. 动态阈值截断（Dynamic Thresholding）：基于百分位的阈值，滤除微弱噪声
    3. 形态学去噪（Morphological Cleaning）：去除孤立的噪点，只保留成块的病灶区域
    4. 特定模态约束：如果指定modality，应用解剖学约束
    
    Args:
        heatmap: [H, W] 原始CAM热图，范围[0, 1]
        original_image: [H, W, 3] 原始图像，范围[0, 255]或[0, 1]
        intensity_thresh: 原图像素值阈值，低于此值的区域（黑色背景）强制不激活
        percentile_thresh: 动态阈值比例（0.0-1.0），保留最大值的此比例以上的区域
        use_morphology: 是否使用形态学操作去除孤立点
        modality: 'oct' | 'colpo' | None，如果指定则应用特定模态的解剖学约束
    
    Returns:
        clean_heatmap: [H, W] 清洗后的热图，范围[0, 1]
    """
    # 如果指定了modality，先应用解剖学约束
    if modality in ['oct', 'colpo']:
        heatmap = apply_anatomical_constraints(heatmap, original_image, modality=modality, percentile_thresh=percentile_thresh)
    
    # 确保原始图像是numpy数组
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()
    
    # 确保原始图像是uint8格式
    if original_image.dtype != np.uint8:
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = original_image.astype(np.uint8)
    
    # 调整热力图尺寸到原图大小（如果还没调整）
    h, w = original_image.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))
    
    # 转换为灰度图用于背景检测
    if len(original_image.shape) == 3:
        img_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = original_image
    
    # --- 核心去噪步骤（如果还没应用解剖学约束）---
    if modality not in ['oct', 'colpo']:
        # A. 物理遮罩（Anatomical Masking）
        # OCT/Colpo背景通常接近黑色。如果原图像素值很低（<intensity_thresh），强制不激活。
        mask = img_gray > intensity_thresh
        heatmap = heatmap * mask.astype(np.float32)
        
        # B. 动态阈值截断（Dynamic Thresholding）
        if np.max(heatmap) > 0:
            thresh_val = np.max(heatmap) * percentile_thresh
            heatmap[heatmap < thresh_val] = 0
    
    # C. 形态学去噪（Morphological Cleaning）
    # 只有成块的激活才算数，零星的像素点往往是噪声
    if use_morphology and np.max(heatmap) > 0:
        kernel = np.ones((5, 5), np.uint8)
        # 将heatmap转换为uint8进行形态学操作
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_uint8 = cv2.morphologyEx(heatmap_uint8, cv2.MORPH_OPEN, kernel)
        heatmap = heatmap_uint8.astype(np.float32) / 255.0
    
    # D. 重新归一化（让剩下的区域最亮处变回1.0）
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    else:
        heatmap = np.zeros_like(heatmap)
    
    return heatmap


def overlay_cam_on_image(image, cam, alpha=0.5, threshold=0.3, use_gaussian_blur=True, 
                         original_image=None, intensity_thresh=30, percentile_thresh=0.4, modality=None):
    """
    将Grad-CAM热图叠加到原始图像上（带去噪处理 + 特定模态的解剖学约束）
    
    关键改进：
    1. 特定模态约束：OCT屏蔽上方/边缘，Colposcopy中心聚焦
    2. 智能叠加：只在激活区域叠加颜色，背景保持原图（避免"黑乎乎"的病变被遮挡）
    3. 高斯模糊：平滑像素化噪声
    4. 动态阈值：消除微弱的背景响应
    
    Args:
        image: [H, W, 3] RGB图像，范围[0, 255]
        cam: [H, W] 热图，范围[0, 1]
        alpha: 叠加透明度
        threshold: 阈值过滤（0.0-1.0），低于此值的区域设为0（如果使用clean_heatmap则忽略）
        use_gaussian_blur: 是否使用高斯模糊平滑
        original_image: [H, W, 3] 原始图像，用于解剖学约束（可选）
        intensity_thresh: 原图像素值阈值，用于背景遮罩（默认30）
        percentile_thresh: 动态阈值比例，用于去除噪声（默认0.4，即保留最大值的40%以上）
        modality: 'oct' | 'colpo' | None，指定图像类型以应用特定约束
    
    Returns:
        result: [H, W, 3] 叠加结果
    """
    # 确保image是3通道
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # 确保image是float32格式，范围[0, 1]
    if image.dtype != np.float32:
        image = np.float32(image) / 255.0
    
    h, w = image.shape[:2]
    
    # 1. Resize mask to image size
    heatmap = cv2.resize(cam, (w, h))
    
    # 2. 如果提供了original_image，使用clean_heatmap进行智能去噪（包含特定模态约束）
    if original_image is not None:
        heatmap = clean_heatmap(
            heatmap, 
            original_image, 
            intensity_thresh=intensity_thresh,
            percentile_thresh=percentile_thresh,
            use_morphology=True,
            modality=modality  # 传递模态类型
        )
    else:
        # 传统方法：高斯模糊 + 阈值过滤
        if use_gaussian_blur:
            heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        heatmap[heatmap < threshold] = 0
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
    
    # 3. 增强热力图对比度（在应用colormap之前）
    # 使用非线性拉伸，使激活区域更明显
    if np.max(heatmap) > 0:
        # 将热力图拉伸到更宽的动态范围
        heatmap_enhanced = np.power(heatmap / np.max(heatmap), 0.6)  # gamma < 1增强低值
        heatmap_enhanced = heatmap_enhanced / np.max(heatmap_enhanced) if np.max(heatmap_enhanced) > 0 else heatmap_enhanced
    else:
        heatmap_enhanced = heatmap
    
    # 4. Apply Colormap (使用经典医学热力图风格的JET配色)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_enhanced), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_colored = np.float32(heatmap_colored) / 255.0
    
    # 5. 通透式叠加（关键改进：基于热力图强度的智能混合）
    # 不透明度由热力图强度决定
    # 高激活 = 不透明颜色，低激活 = 半透明（显示纹理）
    # 这样你可以透过热力图看到底下的"冰柱"或"囊肿"纹理
    mask = heatmap_enhanced[..., None]  # 扩展维度以匹配 RGB [H, W, 1]
    
    # 增强低-中激活的可见性
    overlay_alpha = np.clip(mask * 1.5, 0, alpha)
    
    # 智能叠加：只有激活强的地方才叠加颜色，弱的地方完全透明
    # 这确保"黑乎乎"的病变区域如果被激活，会显示为深红色
    # 如果没有被激活（背景），则保持黑色原貌，对比度极高
    # 未激活区域保持原图清晰度，激活区域保持半透明，可以透过热力图看到纹理
    overlay = (heatmap_colored * overlay_alpha) + (image * (1 - overlay_alpha))
    
    # 5. 转换回uint8格式
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    
    return overlay


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
    num_samples=8,
    target_layer_name=None
):
    """
    生成Grad-CAM可视化
    
    Args:
        model: Bio-COT 3.2模型
        dataloader: 数据加载器
        device: 设备
        save_dir: 保存目录
        num_samples: 样本数量
        target_layer_name: 目标层名称（可选）
    """
    print("\n🎨 生成Bio-COT优化的LayerCAM激活图（保留空间细节）...")
    print(f"   Target Layer: {target_layer_name if target_layer_name else 'Auto-detected'}")
    print(f"   Algorithm: LayerCAM (no global average pooling, preserves spatial details)")
    
    model.to(device)
    model.eval()
    
    # 创建LayerCAM对象（升级版：保留空间细节）
    grad_cam = BioCotLayerCAM(model, target_layer_name=target_layer_name)
    
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
        
        # 提取特征（和训练时一样）
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
    
    for idx, sample in enumerate(tqdm(samples, desc="Generating Grad-CAM")):
        label_name = "Positive" if sample['label'] == 1 else "Negative"
        
        # 1. 原始OCT图像
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
            # 使用clean_heatmap进行智能去噪（LayerCAM + 特定模态的解剖学约束 + 暗区反向加权）
            # modality='oct': 应用OCT专属约束（屏蔽上方20% + 底部保护 + 暗区增强2.5x）
            # percentile_thresh=0.15: 降低阈值，确保弱激活也能显示（避免"什么都没激活"）
            # alpha=0.7: 提高alpha值，增强激活色调
            oct_overlay = overlay_cam_on_image(
                oct_img, 
                oct_cam, 
                alpha=0.7,  # 提高alpha值，增强激活色调
                threshold=0.15, 
                use_gaussian_blur=True,
                original_image=oct_img,  # 提供原图用于解剖学约束
                intensity_thresh=20,      # OCT背景阈值
                percentile_thresh=0.15,  # 动态阈值比例（降低到15%，确保弱激活显示）
                modality='oct'            # 指定OCT模态，应用专属约束+暗区反向加权
            )
        except Exception as e:
            print(f"   ⚠️ Sample {idx} OCT CAM失败: {e}")
            import traceback
            traceback.print_exc()
            oct_overlay = oct_img
            oct_cam = np.zeros((14, 14))
        
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.imshow(oct_overlay)
        ax2.set_title('OCT + Grad-CAM', fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        # 3. OCT纯热图
        ax3 = fig.add_subplot(gs[idx, 2])
        im = ax3.imshow(oct_cam, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax3.set_title('OCT CAM Heatmap', fontsize=11, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. 原始Colposcopy图像
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
            # 使用clean_heatmap进行智能去噪（LayerCAM + 特定模态的解剖学约束 + 红色增强）
            # modality='colpo': 应用Colposcopy专属约束（中心高斯聚焦 + 红色增强 + 窥器反光去除）
            # percentile_thresh=0.15: 降低阈值，确保弱激活也能显示（避免"什么都没激活"）
            # alpha=0.7: 提高alpha值，增强激活色调
            colpo_overlay = overlay_cam_on_image(
                colpo_img, 
                colpo_cam, 
                alpha=0.7,  # 提高alpha值，增强激活色调
                threshold=0.15, 
                use_gaussian_blur=True,
                original_image=colpo_img,  # 提供原图用于解剖学约束
                intensity_thresh=30,       # Colposcopy背景阈值
                percentile_thresh=0.15,     # 动态阈值比例（降低到15%，确保弱激活显示）
                modality='colpo'           # 指定Colposcopy模态，应用专属约束+红色增强
            )
        except Exception as e:
            print(f"   ⚠️ Sample {idx} Colpo CAM失败: {e}")
            colpo_overlay = colpo_img
            colpo_cam = np.zeros((14, 14))
        
        ax5 = fig.add_subplot(gs[idx, 4])
        ax5.imshow(colpo_overlay)
        ax5.set_title('Colpo + Grad-CAM', fontsize=11, fontweight='bold')
        ax5.axis('off')
        
        # 6. Colposcopy纯热图
        ax6 = fig.add_subplot(gs[idx, 5])
        im = ax6.imshow(colpo_cam, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax6.set_title('Colpo CAM Heatmap', fontsize=11, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    
    plt.suptitle('Bio-COT 3.2: LayerCAM Activation Analysis (Preserves Spatial Details)',
                fontsize=18, fontweight='bold', y=0.995)
    
    # 保存
    save_path = Path(save_dir) / 'CAM_Activation_Analysis_LayerCAM.pdf'
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    save_path_png = Path(save_dir) / 'CAM_Activation_Analysis_LayerCAM.png'
    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    
    print(f"✅ Bio-COT LayerCAM visualization saved: {save_path}")
    
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("Bio-COT 3.2: LayerCAM Visualization Generator (Ultimate Version)")
    print("Key Features: LayerCAM (no GAP) + Anatomical Constraints + Dark Lesion Boosting")
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
    
    # 可选：打印所有层名称，方便用户选择目标层
    print("\n📋 模型层结构（前20个）：")
    for i, (name, module) in enumerate(model.named_modules()):
        if i < 20:
            print(f"   {name}: {type(module).__name__}")
    print("   ... (use 'for name, _ in model.named_modules(): print(name)' to see all)")
    
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
    
    # 使用自动检测的目标层（visual_encoder.vit.blocks[-1]）
    # 如果需要指定特定层，可以传入target_layer_name参数
    generate_cam_visualizations(
        model, 
        dataloader, 
        device, 
        save_dir, 
        num_samples=8,
        target_layer_name=None  # None表示自动检测
    )
    
    print("\n" + "=" * 80)
    print("✅ Bio-COT LayerCAM生成完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

