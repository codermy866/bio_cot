#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应因果干预CLIP（Adaptive Causal Intervention CLIP）
整合所有创新模块的统一框架

核心创新：
1. 因果干预机制（Causal Intervention）
2. 自适应因果图（Adaptive Causal Graph）
3. 分层对比学习（Hierarchical Contrastive Learning）
4. 不确定性引导的干预（Uncertainty-Guided Intervention）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

# 导入基础模块
try:
    from src.models.causal_bayesian_clip_framework import (
        BayesianCLIPEncoder,
        CausalAttentionMask
    )
    from src.models.causal_intervention import (
        CausalIntervention,
        AdaptiveInterventionScheduler
    )
    from src.models.adaptive_causal_graph import (
        AdaptiveCausalGraph,
        PersonalizedCausalGraph
    )
    from src.models.hierarchical_contrastive import (
        HierarchicalContrastiveLearning
    )
except ImportError:
    # 如果导入失败，尝试直接导入
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / 'src'))
    from models.causal_bayesian_clip_framework import (
        BayesianCLIPEncoder,
        CausalAttentionMask
    )
    from models.causal_intervention import (
        CausalIntervention,
        AdaptiveInterventionScheduler
    )
    from models.adaptive_causal_graph import (
        AdaptiveCausalGraph,
        PersonalizedCausalGraph
    )
    from models.hierarchical_contrastive import (
        HierarchicalContrastiveLearning
    )
try:
    from src.models.enhanced_causal_clip import UncertaintyDecomposition
except ImportError:
    # 如果导入失败，尝试直接导入
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / 'src'))
    from models.enhanced_causal_clip import UncertaintyDecomposition


class AdaptiveCausalInterventionCLIP(nn.Module):
    """
    自适应因果干预CLIP模型
    整合所有创新模块
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        clinical_dim: int = 7,
        num_classes: int = 2,
        temperature: float = 0.07,
        kl_weight: float = 0.01,
        use_causal_intervention: bool = True,
        use_adaptive_causal_graph: bool = True,
        use_hierarchical_contrastive: bool = True,
        use_uncertainty_guided: bool = True,
        use_uncertainty_decomposition: bool = True,
        num_centers: int = 0
    ):
        """
        Args:
            embed_dim: 特征维度
            clinical_dim: 临床特征维度
            num_classes: 分类类别数
            temperature: 对比学习温度
            kl_weight: KL散度权重
            use_causal_intervention: 是否使用因果干预
            use_adaptive_causal_graph: 是否使用自适应因果图
            use_hierarchical_contrastive: 是否使用分层对比学习
            use_uncertainty_guided: 是否使用不确定性引导的干预
            use_uncertainty_decomposition: 是否使用不确定性分解
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.clinical_dim = clinical_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.kl_weight = kl_weight
        
        # 标志位
        self.use_causal_intervention = use_causal_intervention
        self.use_adaptive_causal_graph = use_adaptive_causal_graph
        self.use_hierarchical_contrastive = use_hierarchical_contrastive
        self.use_uncertainty_guided = use_uncertainty_guided
        self.use_uncertainty_decomposition = use_uncertainty_decomposition
        
        # 临床特征投影
        self.clinical_proj = nn.Linear(clinical_dim, embed_dim)
        
        # 贝叶斯编码器
        self.oct_encoder = BayesianCLIPEncoder(embed_dim)
        self.colposcopy_encoder = BayesianCLIPEncoder(embed_dim)
        self.clinical_encoder = BayesianCLIPEncoder(embed_dim)
        
        # 因果干预模块
        if use_causal_intervention:
            self.causal_intervention = CausalIntervention(
                embed_dim=embed_dim,
                num_modalities=3
            )
        else:
            self.causal_intervention = None
        
        # 自适应因果图模块
        if use_adaptive_causal_graph:
            self.adaptive_causal_graph = AdaptiveCausalGraph(
                embed_dim=embed_dim,
                num_modalities=3
            )
            # 个性化因果图（可选）
            self.personalized_causal_graph = PersonalizedCausalGraph(
                embed_dim=embed_dim,
                num_modalities=3
            )
        else:
            self.adaptive_causal_graph = None
            self.personalized_causal_graph = None
        
        # 分层对比学习模块
        if use_hierarchical_contrastive:
            self.hierarchical_contrastive = HierarchicalContrastiveLearning(
                embed_dim=embed_dim,
                num_modalities=3,
                temperature=temperature
            )
        else:
            self.hierarchical_contrastive = None
        
        # 不确定性引导的干预调度器
        if use_uncertainty_guided:
            self.intervention_scheduler = AdaptiveInterventionScheduler(
                embed_dim=embed_dim
            )
        else:
            self.intervention_scheduler = None
        
        # 不确定性分解模块
        if use_uncertainty_decomposition:
            self.uncertainty_decomposition = UncertaintyDecomposition(embed_dim)
        else:
            self.uncertainty_decomposition = None
        
        # 跨模态注意力融合
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 融合层（增加Dropout防止过拟合）
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.6),  # 增加Dropout从0.5到0.6
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(0.3)  # 添加额外Dropout层
        )
        
        # 分类头（增加Dropout防止过拟合）
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.6),  # 增加Dropout从0.5到0.6
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 特征投影层（用于处理维度不匹配，动态创建但需要注册）
        self.oct_feat_proj = None
        self.colpo_feat_proj = None
    
    def forward(
        self,
        oct_feat: torch.Tensor,
        colpo_feat: torch.Tensor,
        clinical_feat: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_uncertainty: bool = True,
        return_intervention: bool = False,
        return_causal_graph: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            oct_feat: OCT特征 [B, embed_dim]
            colpo_feat: Colposcopy特征 [B, embed_dim]
            clinical_feat: 临床特征 [B, clinical_dim]
            labels: 标签 [B]（用于分层对比学习）
            return_uncertainty: 是否返回不确定性
            return_intervention: 是否返回干预信息
            return_causal_graph: 是否返回因果图
        
        Returns:
            输出字典
        """
        B = oct_feat.size(0)
        
        # 1. 临床特征投影
        clinical_proj = self.clinical_proj(clinical_feat)
        
        # 2. 特征维度检查和投影（确保所有特征都是embed_dim维）
        if oct_feat.size(-1) != self.embed_dim:
            if self.oct_feat_proj is None:
                self.oct_feat_proj = nn.Linear(oct_feat.size(-1), self.embed_dim).to(oct_feat.device)
                # 将投影层添加到模型参数中
                self.add_module('oct_feat_proj', self.oct_feat_proj)
            oct_feat = self.oct_feat_proj(oct_feat)
        
        if colpo_feat.size(-1) != self.embed_dim:
            if self.colpo_feat_proj is None:
                self.colpo_feat_proj = nn.Linear(colpo_feat.size(-1), self.embed_dim).to(colpo_feat.device)
                # 将投影层添加到模型参数中
                self.add_module('colpo_feat_proj', self.colpo_feat_proj)
            colpo_feat = self.colpo_feat_proj(colpo_feat)
        
        # 数值稳定性：clamp输入特征
        oct_feat = torch.clamp(oct_feat, min=-10.0, max=10.0)
        colpo_feat = torch.clamp(colpo_feat, min=-10.0, max=10.0)
        clinical_proj = torch.clamp(clinical_proj, min=-10.0, max=10.0)
        
        # 3. 贝叶斯编码（均值和方差）
        oct_mean, oct_var = self.oct_encoder(oct_feat)
        colpo_mean, colpo_var = self.colposcopy_encoder(colpo_feat)
        clinical_mean, clinical_var = self.clinical_encoder(clinical_proj)
        
        # 数值稳定性：clamp均值和方差
        oct_mean = torch.clamp(oct_mean, min=-10.0, max=10.0)
        oct_var = torch.clamp(oct_var, min=1e-6, max=10.0)
        colpo_mean = torch.clamp(colpo_mean, min=-10.0, max=10.0)
        colpo_var = torch.clamp(colpo_var, min=1e-6, max=10.0)
        clinical_mean = torch.clamp(clinical_mean, min=-10.0, max=10.0)
        clinical_var = torch.clamp(clinical_var, min=1e-6, max=10.0)
        
        # 3. 采样（训练时采样，推理时用均值）
        oct_feat_sampled = self.oct_encoder.sample(oct_mean, oct_var, self.training)
        colpo_feat_sampled = self.colposcopy_encoder.sample(colpo_mean, colpo_var, self.training)
        clinical_feat_sampled = self.clinical_encoder.sample(clinical_mean, clinical_var, self.training)
        
        # 数值稳定性：clamp采样后的特征
        oct_feat_sampled = torch.clamp(oct_feat_sampled, min=-10.0, max=10.0)
        colpo_feat_sampled = torch.clamp(colpo_feat_sampled, min=-10.0, max=10.0)
        clinical_feat_sampled = torch.clamp(clinical_feat_sampled, min=-10.0, max=10.0)
        
        features = [oct_feat_sampled, colpo_feat_sampled, clinical_feat_sampled]
        variances = [oct_var, colpo_var, clinical_var]
        
        # 4. 自适应因果图发现（暂时关闭，待AUC>0.5后逐步打开）
        causal_graph_info = None
        if self.use_adaptive_causal_graph and self.adaptive_causal_graph is not None:
            causal_graph_info = self.adaptive_causal_graph(features)
            # 使用个性化因果图
            if self.personalized_causal_graph is not None:
                personalized_info = self.personalized_causal_graph(features)
                causal_graph_info['personalized'] = personalized_info['personalized_graph']
        
        # 5. 因果干预（暂时关闭，待AUC>0.5后逐步打开）
        intervention_info = None
        if self.use_causal_intervention and self.causal_intervention is not None:
            # 计算不确定性（用于引导干预）
            uncertainty = None
            if self.use_uncertainty_guided and self.intervention_scheduler is not None:
                # 计算总不确定性
                total_var = torch.stack(variances).mean(dim=0)  # [B, embed_dim]
                uncertainty = total_var.mean(dim=-1, keepdim=True)  # [B, 1]
                
                # 计算干预强度
                intervention_strength = self.intervention_scheduler(uncertainty)
                
            # 根据不确定性选择干预目标（高不确定性时干预所有模态）
            # 简化：选择不确定性最高的模态进行干预
            intervention_targets = None  # 暂时不自动选择，让模型学习
            
            # 应用因果干预（暂时不指定具体目标，让模型学习）
            intervention_info = self.causal_intervention(
                features,
                intervention_target=None  # 暂时不进行干预，先让模型学习基础特征
            )
            features = intervention_info['intervened_features']
        
        # 6. 跨模态注意力融合
        # 数值稳定性：clamp所有输入特征
        features = [torch.clamp(f, min=-10.0, max=10.0) for f in features]
        
        # 将特征堆叠为序列
        feature_seq = torch.stack(features, dim=1)  # [B, 3, embed_dim]
        
        # 数值稳定性：clamp feature_seq
        feature_seq = torch.clamp(feature_seq, min=-10.0, max=10.0)
        
        # 强制禁用注意力掩码（暂时关闭，待AUC>0.5后逐步打开）
        attn_mask = None  # 强制设置为None，暂时禁用注意力掩码
        
        # 多头注意力
        fused_feat, attn_weights = self.multihead_attn(
            feature_seq, feature_seq, feature_seq,
            attn_mask=attn_mask  # 强制为None
        )  # [B, 3, embed_dim]
        
        # 数值稳定性：clamp注意力输出
        fused_feat = torch.clamp(fused_feat, min=-10.0, max=10.0)
        
        # 7. 融合
        fused_feat_flat = fused_feat.reshape(B, -1)  # [B, 3 * embed_dim]
        fused = self.fusion(fused_feat_flat)  # [B, embed_dim]
        
        # 检查fused特征
        if torch.isnan(fused).any() or torch.isinf(fused).any():
            print(f"⚠️  fused特征包含NaN/Inf，范围: [{fused.min().item():.4f}, {fused.max().item():.4f}]")
            fused = torch.zeros_like(fused)
        
        # 数值稳定性：clamp fused特征
        fused = torch.clamp(fused, min=-10.0, max=10.0)
        
        # 8. 分类
        logits = self.classifier(fused)  # [B, num_classes]
        
        # 检查并修复NaN（使用bias的值，通过分类器重新计算以确保需要梯度）
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"⚠️  检测到NaN/Inf logits，fused范围: [{fused.min().item():.4f}, {fused.max().item():.4f}]")
            # 如果fused特征正常，重新计算logits（只使用bias）
            if not torch.isnan(fused).any() and not torch.isinf(fused).any():
                # 创建一个简单的线性层，只使用bias
                if isinstance(self.classifier, nn.Sequential):
                    last = self.classifier[-1]
                    if hasattr(last, 'bias') and last.bias is not None:
                        # 使用bias的值，通过一个简单的线性变换
                        logits = last.bias.unsqueeze(0).expand(B, -1)
                        # 添加一个小的fused特征贡献以确保梯度流
                        if fused.requires_grad:
                            logits = logits + fused.mean(dim=-1, keepdim=True) * 0.001
                    else:
                        # 回退：使用fused特征的简单投影
                        logits = fused.mean(dim=-1, keepdim=True).expand(-1, self.num_classes)
                else:
                    if hasattr(self.classifier, 'bias') and self.classifier.bias is not None:
                        logits = self.classifier.bias.unsqueeze(0).expand(B, -1)
                        if fused.requires_grad:
                            logits = logits + fused.mean(dim=-1, keepdim=True) * 0.001
                    else:
                        logits = fused.mean(dim=-1, keepdim=True).expand(-1, self.num_classes)
            else:
                # fused特征也有问题，使用bias的默认值
                if isinstance(self.classifier, nn.Sequential):
                    last = self.classifier[-1]
                    if hasattr(last, 'bias') and last.bias is not None:
                        logits = last.bias.unsqueeze(0).expand(B, -1)
                    else:
                        # 创建一个需要梯度的张量
                        logits = torch.zeros(B, self.num_classes, device=fused.device)
                        if fused.requires_grad:
                            logits = logits + fused.mean(dim=-1, keepdim=True) * 0.0  # 确保需要梯度
                else:
                    logits = torch.zeros(B, self.num_classes, device=fused.device)
        
        logits = torch.clamp(logits, min=-50.0, max=50.0)  # 防止极端值
        
        # 9. 不确定性分解
        uncertainty_info = None
        if return_uncertainty and self.use_uncertainty_decomposition:
            total_var = torch.stack(variances).mean(dim=0)  # [B, embed_dim]
            if self.uncertainty_decomposition is not None:
                uncertainty_info = self.uncertainty_decomposition(fused, total_var)
            else:
                uncertainty_info = {'total': total_var.mean(dim=-1)}
        
        # 10. 分层对比学习（如果提供标签）
        contrastive_info = None
        if labels is not None and self.use_hierarchical_contrastive:
            if self.hierarchical_contrastive is not None:
                contrastive_info = self.hierarchical_contrastive(features, labels)
        
        # 构建输出字典
        output = {
            'logits': logits,
            'features': fused,
            'attention_weights': attn_weights
        }
        
        # 返回mean和var用于KL损失计算
        output['oct_mean'] = oct_mean
        output['oct_var'] = oct_var
        output['colpo_mean'] = colpo_mean
        output['colpo_var'] = colpo_var
        output['clinical_mean'] = clinical_mean
        output['clinical_var'] = clinical_var
        
        if return_uncertainty and uncertainty_info is not None:
            output['uncertainty'] = uncertainty_info
        
        if return_intervention and intervention_info is not None:
            output['intervention'] = intervention_info
            # 提取intervention_loss（如果存在）
            if isinstance(intervention_info, dict):
                output['intervention_loss'] = intervention_info.get('intervention_loss', torch.tensor(0.0, device=logits.device))
        
        if return_causal_graph and causal_graph_info is not None:
            output['causal_graph'] = causal_graph_info
        
        if contrastive_info is not None:
            output['contrastive'] = contrastive_info
            # 提取contrastive_loss（如果存在）
            if isinstance(contrastive_info, dict):
                output['contrastive_loss'] = contrastive_info.get('total_contrastive_loss', 
                                                                  contrastive_info.get('contrastive_loss', 
                                                                                      torch.tensor(0.0, device=logits.device)))
        
        return output

