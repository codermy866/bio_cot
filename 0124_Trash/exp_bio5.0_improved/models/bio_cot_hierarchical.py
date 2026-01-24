from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import HierarchicalViT
from .mhc_fusion import NoiseAwareMHC
from .clinical_evolver import ClinicalEvolver
from .visual_notes import VisualNotesModule


class DualHeadDisentangler(nn.Module):
    """简化版 dual-head：从 pooled feature 输出 causal/noise 两个向量。"""

    def __init__(self, in_dim: int = 768, out_dim: int = 768):
        super().__init__()
        self.causal = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.noise = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.causal(x), self.noise(x)


class BioCOT_V5_Hierarchical(nn.Module):
    """
    Bio-COT 5.5 Pro: Hierarchical Manifold-Guided Visual Reasoning (HM-VR) + Noise-Aware
    
    Key Features:
    - Multi-scale Feature Extraction (ViT intermediate layers)
    - Noise-Aware mHC Fusion (NA-mHC) at each scale
    - Dynamic Clinical Query Evolution (CoT)
    - Center-Aware Regularization
    
    Stage 1: Hierarchical NA-mHC + Clinical Query Evolution (CoT)
    Stage 2: VLM Anchor + Visual Notes (SCG)
    Stage 3: Causal/Noise disentanglement + classification
    """

    def __init__(self, config, num_classes: int = 2, num_centers: int = 5):
        super().__init__()
        self.config = config
        self.num_stages = len(config.extract_layers)
        self.num_centers = num_centers

        # Stage 1: Hierarchical ViT (with DropPath)
        print(f"🔧 Creating ViT with drop_path_rate={getattr(config, 'drop_path_rate', 0.0)}")
        self.visual_encoder = HierarchicalViT(
            model_name=config.vit_arch,
            pretrained=True,
            out_indices=config.extract_layers,
            drop_path_rate=getattr(config, 'drop_path_rate', 0.0),  # 🔥 防止过拟合的关键
        )
        visual_dim = self.visual_encoder.embed_dim
        if visual_dim != config.visual_dim:
            raise ValueError(f"visual_dim不匹配：backbone={visual_dim}, config={config.visual_dim}")

        # Clinical state init (with Dropout from config)
        dropout_rate = getattr(config, 'dropout_rate', 0.3)
        self.clinical_init = nn.Sequential(
            nn.Linear(config.clinical_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(dropout_rate),  # 🔥 使用配置的dropout_rate
            nn.GELU(),
        )

        # 🔥 升级：使用 NoiseAwareMHC 替代基础 mHC (传递dropout)
        dropout_rate = getattr(config, 'dropout_rate', 0.3)
        self.mhc_layers = nn.ModuleList(
            [
                NoiseAwareMHC(
                    img_dim=config.visual_dim,
                    clinical_dim=config.hidden_dim,
                    num_centers=num_centers,
                    latent_dim=config.mhc_latent_dim,
                    sinkhorn_iters=config.sinkhorn_iters,
                    epsilon=config.mhc_epsilon,
                    dropout=dropout_rate,  # 🔥 传递dropout参数
                )
                for _ in range(self.num_stages)
            ]
        )

        # Evolvers with dropout from config
        dropout_rate = getattr(config, 'dropout_rate', 0.2)
        self.evolvers = nn.ModuleList(
            [
                ClinicalEvolver(
                    visual_dim=config.visual_dim, 
                    clinical_dim=config.hidden_dim,
                    dropout=dropout_rate  # 🔥 传递dropout参数
                )
                for _ in range(max(self.num_stages - 1, 0))
            ]
        )

        # VLM anchor (reuse exp_bio4.0 implementation)
        self.use_vlm_anchor = bool(getattr(config, "use_vlm_anchor", True))
        if self.use_vlm_anchor:
            # exp_bio4.0 目录名包含 '.'，无法用标准包名导入；这里通过 sys.path 插入实现复用
            import sys
            from pathlib import Path

            exp_dir = Path(__file__).resolve().parents[2]  # .../experiments
            exp_bio4_dir = exp_dir / "exp_bio4.0"
            sys.path.insert(0, str(exp_bio4_dir))
            from knowledge_base.enhanced_knowledge_retriever import VLMAugmentedRetriever  # type: ignore

            self.knowledge_retriever = VLMAugmentedRetriever(
                vlm_json_path=config.vlm_json_path,
                visual_dim=config.visual_dim,
                text_model_name=config.text_model_name,
            )
        else:
            self.knowledge_retriever = None

        # Visual Notes + SCG
        self.use_visual_notes = bool(getattr(config, "use_visual_notes", True))
        if self.use_visual_notes:
            # Text adapter with dropout
            dropout_rate = getattr(config, 'dropout_rate', 0.3)
            self.text_adapter = nn.Sequential(
                nn.Linear(768, config.visual_dim),
                nn.Dropout(dropout_rate)  # 🔥 添加dropout
            )
            self.visual_notes = VisualNotesModule(
                img_dim=config.visual_dim,
                text_dim=config.visual_dim,
                hidden_dim=256,
                warmup_epochs=10,
            )
        else:
            self.text_adapter = None
            self.visual_notes = None

        # Disentangler
        self.use_dual = bool(getattr(config, "use_dual", True))
        if self.use_dual:
            self.disentangler = DualHeadDisentangler(in_dim=config.visual_dim, out_dim=config.visual_dim)
        else:
            self.disentangler = None

        # Classifier (增强Dropout from config)
        dropout_rate = getattr(config, 'dropout_rate', 0.3)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),  # 🔥 分类前必须加 Dropout
            nn.Linear(config.visual_dim, num_classes),
        )
        
        # Center Aux Classifier (optional)
        self.center_classifier = nn.Linear(config.mhc_latent_dim, num_centers)

    def set_epoch(self, epoch: int) -> None:
        """设置当前epoch，用于动态调整（如warmup、freeze策略）"""
        if self.visual_notes is not None:
            self.visual_notes.set_epoch(epoch)
    
    def freeze_backbone(self) -> None:
        """冻结ViT骨干网络（用于渐进式训练）"""
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self) -> None:
        """解冻ViT骨干网络"""
        for param in self.visual_encoder.parameters():
            param.requires_grad = True

    def forward(
        self,
        images: torch.Tensor,  # [B,3,224,224]
        clinical_features: torch.Tensor,  # [B,C]
        center_ids: Optional[torch.Tensor] = None,  # 🔥 新增：中心ID [B]
        image_names: Optional[List[str]] = None,
        clinical_info: Optional[List[str]] = None,
        return_loss_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Flow:
        Input -> [ViT Layers] 
              -> Loop: (Visual_i + Clinical_State_i) -> NA-mHC -> Visual_Fused_i
                       Visual_Fused_i -> Evolver -> Clinical_State_{i+1}
              -> Visual Note Layer -> Classifier
        """
        # Stage 1: hierarchical co-evolution
        vis_feats_list = self.visual_encoder(images)  # list of [B,N,D]
        if len(vis_feats_list) != self.num_stages:
            raise RuntimeError(f"backbone输出stage数异常：{len(vis_feats_list)} vs {self.num_stages}")

        clin_state = self.clinical_init(clinical_features)  # [B,hidden]
        final_feat = None
        all_noise_probs = []  # 🔥 收集每一层的噪声图

        # C. 循环推理 (The Loop)
        for i in range(self.num_stages):
            feat = vis_feats_list[i]
            
            # (1) NA-mHC Fusion: 净化并融合
            # feat_fused: [B, N, 768]
            # noise_prob: [B, N, 1]
            feat_fused, noise_prob = self.mhc_layers[i](feat, clin_state, center_ids)
            all_noise_probs.append(noise_prob)
            
            # 记录最后一层特征
            if i == self.num_stages - 1:
                final_feat = feat_fused
            
            # (2) Query Evolution: 更新临床意图 (除非是最后一层)
            if i < self.num_stages - 1:
                clin_state = self.evolvers[i](feat_fused, clin_state)

        assert final_feat is not None

        # Stage 2: VLM anchor + Visual Notes
        z_anchor = None
        attn_map = None
        if self.knowledge_retriever is not None and image_names is not None:
            z_anchor = self.knowledge_retriever(image_names, clinical_info, device=str(images.device))
            if self.text_adapter is not None:
                z_anchor = self.text_adapter(z_anchor)  # 🔥 通过adapter映射
            if self.visual_notes is not None:
                final_feat, attn_map = self.visual_notes(final_feat, z_anchor)

        # Stage 3: pool + disentangle + classify
        pooled = final_feat.mean(dim=1)  # [B,D]
        loss_components: Dict[str, torch.Tensor] = {}

        if self.disentangler is not None:
            z_causal, z_noise = self.disentangler(pooled)
            # orthogonality loss
            zc = F.normalize(z_causal, dim=1)
            zn = F.normalize(z_noise, dim=1)
            loss_components["L_ortho"] = torch.mean(torch.abs(torch.sum(zc * zn, dim=1)))
        else:
            z_causal, z_noise = pooled, None

        logits = self.classifier(z_causal)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "z_causal": z_causal,
        }
        if z_noise is not None:
            out["z_noise"] = z_noise
        if z_anchor is not None:
            out["z_anchor"] = z_anchor
        if attn_map is not None:
            out["attn_map"] = attn_map
        if return_loss_components:
            out["loss_components"] = loss_components
            # 🔥 新增：返回噪声概率图列表（用于 Noise Regularization Loss）
            out["noise_probs"] = all_noise_probs
        return out
