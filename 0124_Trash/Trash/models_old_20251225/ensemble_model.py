#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成学习模型
融合多个模型的预测结果以提升性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class EnsembleModel(nn.Module):
    """
    集成多个模型的预测结果
    支持加权平均和可学习的权重
    """
    
    def __init__(
        self,
        model_paths: Dict[str, str],
        model_configs: Dict[str, dict],
        ensemble_method: str = 'weighted_average',
        learnable_weights: bool = False,
        device: str = 'cuda'
    ):
        """
        Args:
            model_paths: 模型路径字典，如 {'swint': 'path/to/swint/best_model.pth', ...}
            model_configs: 模型配置字典
            ensemble_method: 集成方法 ('weighted_average', 'learnable', 'voting')
            learnable_weights: 是否使用可学习的权重
            device: 设备
        """
        super().__init__()
        self.model_paths = model_paths
        self.model_configs = model_configs
        self.ensemble_method = ensemble_method
        self.device = device
        
        # 加载各个模型
        self.models = nn.ModuleDict()
        self._load_models()
        
        # 集成权重
        if ensemble_method == 'learnable' or learnable_weights:
            num_models = len(self.models)
            self.weights = nn.Parameter(torch.ones(num_models) / num_models)
        elif ensemble_method == 'weighted_average':
            # 基于性能的固定权重（可以根据验证集性能调整）
            self.weights = {
                'swint': 0.4,  # Swin-T性能最好
                'cnn': 0.3,
                'vmamba': 0.3
            }
        else:
            self.weights = None
    
    def _load_models(self):
        """加载所有模型"""
        print("🔄 加载集成模型...")
        
        for model_name, model_path in self.model_paths.items():
            if not os.path.exists(model_path):
                print(f"⚠️  模型文件不存在: {model_path}，跳过")
                continue
            
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 根据模型类型创建模型
                if model_name.lower() in ['swint', 'swin']:
                    from models.SwinT.swin_multimodal_model import SwinTMultimodalTransformer
                    config = self.model_configs.get(model_name, {})
                    model = SwinTMultimodalTransformer(
                        num_classes=2,
                        embed_dim=config.get('embed_dim', 768),
                        num_heads=config.get('num_heads', 12),
                        dropout=config.get('dropout', 0.2),
                        clinical_dim=config.get('clinical_dim', 7),
                        oct_num_frames=config.get('oct_num_frames', 48),
                        col_num_frames=config.get('col_num_frames', 3),
                        swin_name=config.get('swin_name', 'swin_tiny_patch4_window7_224'),
                        pretrained=False,
                        input_size=config.get('input_size', 224),
                        use_frame_attention=config.get('use_frame_attention', False),
                    ).to(self.device)
                
                elif model_name.lower() == 'cnn':
                    from models.cnn_multimodal_model import CNNMultimodalTransformer
                    config = self.model_configs.get(model_name, {})
                    model = CNNMultimodalTransformer(
                        num_classes=2,
                        embed_dim=config.get('embed_dim', 1280),
                        num_heads=config.get('num_heads', 20),
                        dropout=config.get('dropout', 0.3),
                        clinical_dim=config.get('clinical_dim', 7),
                        oct_num_frames=config.get('oct_num_frames', 120),
                        col_num_frames=config.get('col_num_frames', 3)
                    ).to(self.device)
                
                elif model_name.lower() == 'vmamba':
                    from models.vmamba_multimodal_model import VMambaMultimodalTransformer
                    config = self.model_configs.get(model_name, {})
                    model = VMambaMultimodalTransformer(
                        num_classes=2,
                        embed_dim=config.get('embed_dim', 1024),
                        num_heads=config.get('num_heads', 16),
                        dropout=config.get('dropout', 0.1),
                        clinical_dim=config.get('clinical_dim', 7),
                        oct_num_frames=config.get('oct_num_frames', 120),
                        col_num_frames=config.get('col_num_frames', 3),
                        img_size=config.get('img_size', 224),
                        patch_size=config.get('patch_size', 16),
                        depth=config.get('depth', 10),
                        d_state=config.get('d_state', 16)
                    ).to(self.device)
                
                else:
                    print(f"⚠️  未知的模型类型: {model_name}，跳过")
                    continue
                
                # 加载权重
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # 尝试加载权重
                try:
                    model.load_state_dict(state_dict, strict=False)
                    print(f"✅ {model_name} 模型加载成功")
                except Exception as e:
                    print(f"⚠️  {model_name} 模型权重加载失败: {e}")
                    # 尝试非严格加载
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in state_dict.items() 
                                     if k in model_dict and model_dict[k].shape == v.shape}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict, strict=False)
                    print(f"✅ {model_name} 模型加载成功（非严格模式）")
                
                model.eval()
                self.models[model_name] = model
                
            except Exception as e:
                print(f"❌ 加载 {model_name} 模型失败: {e}")
                continue
        
        print(f"✅ 成功加载 {len(self.models)} 个模型")
    
    def forward(
        self,
        oct_images: torch.Tensor,
        col_images: torch.Tensor,
        clinical_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            oct_images: OCT图像 [B, T, C, H, W]
            col_images: Colposcopy图像 [B, T, C, H, W]
            clinical_features: 临床特征 [B, D]
        
        Returns:
            ensemble_logits: 集成后的logits
            individual_logits: 各个模型的logits字典
        """
        individual_logits = {}
        
        # 获取各个模型的预测
        with torch.no_grad():
            for model_name, model in self.models.items():
                try:
                    logits = model(oct_images, col_images, clinical_features)
                    individual_logits[model_name] = logits
                except Exception as e:
                    print(f"⚠️  {model_name} 预测失败: {e}")
                    continue
        
        if len(individual_logits) == 0:
            raise ValueError("所有模型预测都失败了")
        
        # 集成预测
        if self.ensemble_method == 'learnable' or isinstance(self.weights, nn.Parameter):
            # 可学习权重
            weights = F.softmax(self.weights, dim=0)
            ensemble_logits = sum(
                weights[i] * logits 
                for i, (name, logits) in enumerate(individual_logits.items())
            )
        
        elif self.ensemble_method == 'weighted_average':
            # 固定权重加权平均
            total_weight = sum(
                self.weights.get(name, 1.0 / len(individual_logits))
                for name in individual_logits.keys()
            )
            ensemble_logits = sum(
                (self.weights.get(name, 1.0 / len(individual_logits)) / total_weight) * logits
                for name, logits in individual_logits.items()
            )
        
        elif self.ensemble_method == 'voting':
            # 投票（使用概率）
            probs = [F.softmax(logits, dim=1) for logits in individual_logits.values()]
            ensemble_probs = torch.stack(probs).mean(dim=0)
            ensemble_logits = torch.log(ensemble_probs + 1e-8)
        
        else:
            # 简单平均
            ensemble_logits = torch.stack(list(individual_logits.values())).mean(dim=0)
        
        return ensemble_logits, individual_logits
    
    def predict_proba(
        self,
        oct_images: torch.Tensor,
        col_images: torch.Tensor,
        clinical_features: torch.Tensor
    ) -> torch.Tensor:
        """预测概率"""
        logits, _ = self.forward(oct_images, col_images, clinical_features)
        return F.softmax(logits, dim=1)
    
    def predict(
        self,
        oct_images: torch.Tensor,
        col_images: torch.Tensor,
        clinical_features: torch.Tensor
    ) -> torch.Tensor:
        """预测类别"""
        probs = self.predict_proba(oct_images, col_images, clinical_features)
        return probs.argmax(dim=1)


def create_ensemble_model(
    swint_path: str = 'models/SwinT/_results/multimodal/best_model.pth',
    cnn_path: str = 'cnn_result_unified/best_model.pth',
    vmamba_path: str = 'vmamba_result_unified/best_model.pth',
    ensemble_method: str = 'weighted_average',
    device: str = 'cuda'
) -> EnsembleModel:
    """
    创建集成模型
    
    Args:
        swint_path: Swin-T模型路径
        cnn_path: CNN模型路径
        vmamba_path: VMamba模型路径
        ensemble_method: 集成方法
        device: 设备
    
    Returns:
        EnsembleModel实例
    """
    model_paths = {
        'swint': swint_path,
        'cnn': cnn_path,
        'vmamba': vmamba_path
    }
    
    model_configs = {
        'swint': {
            'embed_dim': 768,
            'num_heads': 12,
            'dropout': 0.2,
            'clinical_dim': 7,
            'oct_num_frames': 48,
            'col_num_frames': 3,
            'swin_name': 'swin_tiny_patch4_window7_224',
            'input_size': 224,
            'use_frame_attention': False
        },
        'cnn': {
            'embed_dim': 1280,
            'num_heads': 20,
            'dropout': 0.3,
            'clinical_dim': 7,
            'oct_num_frames': 120,
            'col_num_frames': 3
        },
        'vmamba': {
            'embed_dim': 1024,
            'num_heads': 16,
            'dropout': 0.1,
            'clinical_dim': 7,
            'oct_num_frames': 120,
            'col_num_frames': 3,
            'img_size': 224,
            'patch_size': 16,
            'depth': 10,
            'd_state': 16
        }
    }
    
    return EnsembleModel(
        model_paths=model_paths,
        model_configs=model_configs,
        ensemble_method=ensemble_method,
        device=device
    )

