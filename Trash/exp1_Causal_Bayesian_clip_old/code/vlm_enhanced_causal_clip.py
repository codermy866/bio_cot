#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM-Enhanced Causal Bayesian CLIP
结合Vision-Language Model和因果推理的医学多模态诊断框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np

# 导入现有模块
from enhanced_causal_clip import (
    LearnableCausalGraph,
    BayesianCLIPEncoder,
    UncertaintyDecomposition
)


class VLMImageEncoder(nn.Module):
    """
    VLM图像编码器
    支持Qwen-VL、LLaVA-Med等VLM模型
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        use_medical_pretrain: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.use_medical_pretrain = use_medical_pretrain
        
        # 根据模型名称选择不同的实现
        if "Qwen" in model_name:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
        elif "LLaVA" in model_name or "llava" in model_name.lower():
            # LLaVA-Med实现
            try:
                from llava_med import LLaVAMedModel, LLaVAMedProcessor
                self.vlm = LLaVAMedModel.from_pretrained(model_name)
                self.processor = LLaVAMedProcessor.from_pretrained(model_name)
            except ImportError:
                print("LLaVA-Med not available, using Qwen-VL as fallback")
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-VL-2B-Instruct"
                )
                self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        else:
            raise ValueError(f"Unsupported VLM model: {model_name}")
        
        # 冻结VLM参数（可选，根据需求调整）
        self.freeze_vlm = False
        
        # 特征投影层（将VLM特征投影到统一维度）
        self.feature_proj = nn.Linear(
            self.vlm.config.vision_config.hidden_size if hasattr(self.vlm.config, 'vision_config') else 1024,
            768
        )
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像
        
        Args:
            images: [B, C, H, W] 或 [B, N, C, H, W] (多帧图像)
        
        Returns:
            features: [B, 768] 或 [B, N, 768]
        """
        if self.freeze_vlm:
            self.vlm.eval()
            with torch.no_grad():
                features = self._encode(images)
        else:
            features = self._encode(images)
        
        # 投影到统一维度
        if features.dim() == 3:  # [B, N, D]
            B, N, D = features.shape
            features = features.view(B * N, D)
            features = self.feature_proj(features)
            features = features.view(B, N, -1)
        else:  # [B, D]
            features = self.feature_proj(features)
        
        return features
    
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """内部编码实现"""
        # 处理输入格式
        if images.dim() == 5:  # [B, N, C, H, W] -> 多帧
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
        
        # 转换为PIL或numpy格式（根据VLM要求）
        # 这里简化处理，实际需要根据具体VLM调整
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(images.device) for k, v in inputs.items()}
        
        # 提取图像特征
        with torch.set_grad_enabled(not self.freeze_vlm):
            outputs = self.vlm.get_image_features(**inputs)
        
        return outputs
    
    def generate_captions(self, images: torch.Tensor, max_length: int = 100) -> List[str]:
        """
        生成图像描述（用于因果线索提取）
        
        Args:
            images: [B, C, H, W]
            max_length: 最大生成长度
        
        Returns:
            captions: List of strings
        """
        self.vlm.eval()
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(images.device) for k, v in inputs.items()}
            
            # 生成描述
            generated_ids = self.vlm.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7
            )
            
            captions = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
        
        return captions


class MedicalKnowledgeBase(nn.Module):
    """
    医学知识库
    存储和检索医学知识（指南、文献、专家知识）
    """
    
    def __init__(
        self,
        knowledge_path: str = "data/medical_knowledge",
        use_vector_db: bool = True
    ):
        super().__init__()
        self.knowledge_path = knowledge_path
        self.use_vector_db = use_vector_db
        
        # 加载知识库
        self.guidelines = self._load_guidelines()
        self.literature = self._load_literature()
        self.expert_knowledge = self._load_expert_knowledge()
        
        # 构建向量数据库（用于快速检索）
        if use_vector_db:
            self.vector_db = self._build_vector_db()
        else:
            self.vector_db = None
    
    def _load_guidelines(self) -> Dict:
        """加载临床指南"""
        # TODO: 实现指南加载
        return {}
    
    def _load_literature(self) -> Dict:
        """加载医学文献"""
        # TODO: 实现文献加载
        return {}
    
    def _load_expert_knowledge(self) -> Dict:
        """加载专家知识"""
        # TODO: 实现专家知识加载
        return {}
    
    def _build_vector_db(self):
        """构建向量数据库"""
        # TODO: 使用FAISS或类似工具构建向量数据库
        return None
    
    def enhance_text(self, medical_text: str, top_k: int = 5) -> str:
        """
        使用知识库增强医学文本
        
        Args:
            medical_text: 原始医学文本
            top_k: 检索top-k相关知识
        
        Returns:
            enhanced_text: 增强后的文本
        """
        # 1. 检索相关知识
        if self.vector_db is not None:
            related_knowledge = self.vector_db.search(medical_text, top_k=top_k)
        else:
            related_knowledge = self._simple_search(medical_text, top_k=top_k)
        
        # 2. 融合知识
        enhanced_text = self._fuse_knowledge(medical_text, related_knowledge)
        
        return enhanced_text
    
    def _simple_search(self, query: str, top_k: int) -> List[str]:
        """简单搜索（关键词匹配）"""
        # TODO: 实现简单搜索
        return []
    
    def _fuse_knowledge(self, text: str, knowledge: List[str]) -> str:
        """融合知识到文本"""
        if not knowledge:
            return text
        
        # 简单融合：在文本后添加相关知识
        knowledge_str = "\n".join([f"- {k}" for k in knowledge])
        enhanced = f"{text}\n\nRelated Medical Knowledge:\n{knowledge_str}"
        
        return enhanced


class CausalAwareAlignment(nn.Module):
    """
    因果感知的VLM对齐模块
    学习图像特征和医学文本之间的因果关系
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        temperature: float = 0.07
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.temperature = temperature
        
        # 图像-文本对齐网络
        self.image_proj = nn.Linear(embed_dim, embed_dim)
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        
        # 因果感知注意力
        self.causal_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )
        
        # 因果对齐损失计算
        self.alignment_loss_fn = nn.CosineEmbeddingLoss()
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        causal_graph: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        因果感知对齐
        
        Args:
            image_features: [B, N, D] 图像特征
            text_features: [B, M, D] 文本特征
            causal_graph: [B, N, M] 因果图（可选）
        
        Returns:
            aligned_features: [B, D] 对齐后的特征
            alignment_info: 对齐信息字典
        """
        # 投影到统一空间
        image_proj = self.image_proj(image_features)
        text_proj = self.text_proj(text_features)
        
        # 如果提供了因果图，使用因果图约束对齐
        if causal_graph is not None:
            # 应用因果图权重
            causal_weights = causal_graph.unsqueeze(-1)  # [B, N, M, 1]
            text_proj_weighted = text_proj.unsqueeze(1) * causal_weights  # [B, N, M, D]
            text_proj_weighted = text_proj_weighted.sum(dim=2)  # [B, N, D]
        else:
            text_proj_weighted = text_proj.mean(dim=1, keepdim=True).expand_as(image_proj)
        
        # 因果感知注意力
        aligned, attn_weights = self.causal_attention(
            image_proj,
            text_proj_weighted,
            text_proj_weighted
        )
        
        # 聚合特征
        aligned_features = aligned.mean(dim=1)  # [B, D]
        
        return aligned_features, {
            'attention_weights': attn_weights,
            'causal_graph': causal_graph
        }


class VLMGuidedCausalDiscovery(nn.Module):
    """
    VLM引导的因果图发现
    使用VLM语义和医学文本指导因果发现
    """
    
    def __init__(
        self,
        num_modalities: int = 3,
        embed_dim: int = 768,
        vlm_encoder: Optional[VLMImageEncoder] = None
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim
        self.vlm_encoder = vlm_encoder
        
        # 基础因果发现网络（继承自LearnableCausalGraph）
        self.base_causal_discovery = LearnableCausalGraph(
            num_modalities=num_modalities,
            embed_dim=embed_dim
        )
        
        # VLM语义提取网络
        self.semantic_extractor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 因果线索提取网络
        self.causal_clue_extractor = nn.Sequential(
            nn.Linear(embed_dim + 128, 256),  # 特征 + 语义
            nn.ReLU(),
            nn.Linear(256, num_modalities * num_modalities)
        )
    
    def extract_causal_clues(
        self,
        images: torch.Tensor,
        medical_texts: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        从VLM生成描述中提取因果线索
        
        Args:
            images: [B, C, H, W] 图像
            medical_texts: 医学文本描述（可选）
        
        Returns:
            causal_clues: 因果线索字典
        """
        if self.vlm_encoder is None:
            return {}
        
        # 生成图像描述
        captions = self.vlm_encoder.generate_captions(images)
        
        # 提取因果线索（简单实现，可改进）
        causal_clues = {}
        for caption in captions:
            # 简单的关键词匹配（可改进为NER或更复杂的方法）
            if "HPV" in caption or "感染" in caption:
                if "OCT" in caption or "异常" in caption:
                    causal_clues.setdefault("HPV感染", []).append("OCT异常")
            if "年龄" in caption or "age" in caption.lower():
                if "病变" in caption or "进展" in caption:
                    causal_clues.setdefault("年龄", []).append("病变进展")
        
        return causal_clues
    
    def forward(
        self,
        features: List[torch.Tensor],
        images: Optional[torch.Tensor] = None,
        medical_texts: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        VLM引导的因果发现
        
        Args:
            features: List of [B, D] 各模态特征
            images: [B, C, H, W] 图像（可选，用于VLM）
            medical_texts: 医学文本（可选）
        
        Returns:
            causal_adj: [B, num_modalities, num_modalities] 因果邻接矩阵
            discovery_info: 发现信息字典
        """
        # 1. 基础因果发现
        base_causal_adj, base_info = self.base_causal_discovery(features)
        
        # 2. VLM语义提取（如果有图像）
        if images is not None and self.vlm_encoder is not None:
            # 提取VLM特征
            vlm_features = self.vlm_encoder.encode_image(images)
            vlm_semantics = self.semantic_extractor(vlm_features.mean(dim=1))  # [B, 128]
            
            # 提取因果线索
            causal_clues = self.extract_causal_clues(images, medical_texts)
            
            # 3. 融合VLM语义到因果发现
            concat_features = torch.cat(features, dim=-1)  # [B, embed_dim * num_modalities]
            vlm_enhanced = torch.cat([concat_features, vlm_semantics], dim=-1)  # [B, embed_dim * num_modalities + 128]
            
            # 生成VLM引导的因果调整
            vlm_adjustment = self.causal_clue_extractor(vlm_enhanced)  # [B, num_modalities * num_modalities]
            vlm_adjustment = vlm_adjustment.view(-1, self.num_modalities, self.num_modalities)
            vlm_adjustment = torch.sigmoid(vlm_adjustment)
            
            # 融合基础因果图和VLM调整
            causal_adj = 0.7 * base_causal_adj + 0.3 * vlm_adjustment
        else:
            causal_adj = base_causal_adj
            causal_clues = {}
        
        return causal_adj, {
            'base_causal_adj': base_causal_adj,
            'vlm_adjustment': vlm_adjustment if images is not None else None,
            'causal_clues': causal_clues,
            **base_info
        }


class MedicalReportGenerator(nn.Module):
    """
    医学诊断报告生成器
    基于VLM生成结构化的诊断报告
    """
    
    def __init__(
        self,
        vlm_encoder: Optional[VLMImageEncoder] = None,
        max_length: int = 512
    ):
        super().__init__()
        self.vlm_encoder = vlm_encoder
        self.max_length = max_length
        
        # 报告模板
        self.report_template = """
Diagnosis Report:

Clinical Findings:
{findings}

Causal Analysis:
{causal_analysis}

Diagnosis:
{diagnosis}

Recommendations:
{recommendations}
"""
    
    def generate(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        causal_graph: torch.Tensor,
        medical_kb: Optional[MedicalKnowledgeBase] = None
    ) -> str:
        """
        生成诊断报告
        
        Args:
            images: [B, C, H, W] 图像
            predictions: [B, num_classes] 预测结果
            causal_graph: [B, num_modalities, num_modalities] 因果图
            medical_kb: 医学知识库（可选）
        
        Returns:
            report: 诊断报告字符串
        """
        if self.vlm_encoder is None:
            return self._generate_simple_report(predictions, causal_graph)
        
        # 1. 生成基础描述
        captions = self.vlm_encoder.generate_captions(images)
        findings = "\n".join(captions)
        
        # 2. 因果分析
        causal_analysis = self._analyze_causal_graph(causal_graph)
        
        # 3. 诊断
        diagnosis = self._generate_diagnosis(predictions)
        
        # 4. 建议（如果提供了知识库）
        if medical_kb is not None:
            recommendations = self._generate_recommendations(diagnosis, medical_kb)
        else:
            recommendations = "Please consult with a medical professional."
        
        # 5. 组装报告
        report = self.report_template.format(
            findings=findings,
            causal_analysis=causal_analysis,
            diagnosis=diagnosis,
            recommendations=recommendations
        )
        
        return report
    
    def _generate_simple_report(self, predictions, causal_graph):
        """简单报告生成（无VLM）"""
        prob = F.softmax(predictions, dim=-1)
        diagnosis = "Abnormal" if prob[0, 1] > 0.5 else "Normal"
        return f"Diagnosis: {diagnosis}\nConfidence: {prob[0, 1]:.2f}"
    
    def _analyze_causal_graph(self, causal_graph):
        """分析因果图"""
        # 简化实现
        return "Causal relationships identified between modalities."
    
    def _generate_diagnosis(self, predictions):
        """生成诊断"""
        prob = F.softmax(predictions, dim=-1)
        if prob[0, 1] > 0.7:
            return "High risk of cervical lesion. Biopsy recommended."
        elif prob[0, 1] > 0.5:
            return "Moderate risk. Follow-up recommended."
        else:
            return "Low risk. Routine screening recommended."
    
    def _generate_recommendations(self, diagnosis, medical_kb):
        """生成建议"""
        # 使用知识库检索相关建议
        related_knowledge = medical_kb.enhance_text(diagnosis, top_k=3)
        return related_knowledge


class VLMEnhancedCausalBayesianCLIP(nn.Module):
    """
    VLM增强的因果贝叶斯CLIP
    完整框架
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        clinical_dim: int = 7,
        num_classes: int = 2,
        vlm_model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        use_medical_kb: bool = True,
        use_vlm_guidance: bool = True,
        use_report_generation: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.clinical_dim = clinical_dim
        self.num_classes = num_classes
        self.use_medical_kb = use_medical_kb
        self.use_vlm_guidance = use_vlm_guidance
        self.use_report_generation = use_report_generation
        
        # VLM编码器
        self.vlm_encoder = VLMImageEncoder(model_name=vlm_model_name)
        
        # 医学知识库
        if use_medical_kb:
            self.medical_kb = MedicalKnowledgeBase()
        else:
            self.medical_kb = None
        
        # 临床特征投影
        self.clinical_proj = nn.Linear(clinical_dim, embed_dim)
        
        # 贝叶斯编码器（保留原有）
        self.oct_encoder = BayesianCLIPEncoder(embed_dim)
        self.colposcopy_encoder = BayesianCLIPEncoder(embed_dim)
        self.clinical_encoder = BayesianCLIPEncoder(embed_dim)
        
        # VLM引导的因果发现
        if use_vlm_guidance:
            self.causal_discovery = VLMGuidedCausalDiscovery(
                num_modalities=3,
                embed_dim=embed_dim,
                vlm_encoder=self.vlm_encoder
            )
        else:
            self.causal_discovery = LearnableCausalGraph(
                num_modalities=3,
                embed_dim=embed_dim
            )
        
        # 因果感知对齐
        self.causal_alignment = CausalAwareAlignment(embed_dim=embed_dim)
        
        # 不确定性分解
        self.uncertainty_decomp = UncertaintyDecomposition(embed_dim=embed_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(384, num_classes)
        )
        
        # 诊断报告生成器
        if use_report_generation:
            self.report_generator = MedicalReportGenerator(
                vlm_encoder=self.vlm_encoder
            )
        else:
            self.report_generator = None
    
    def forward(
        self,
        oct_images: torch.Tensor,
        colposcopy_images: torch.Tensor,
        clinical_features: torch.Tensor,
        medical_texts: Optional[List[str]] = None
    ) -> Dict:
        """
        前向传播
        
        Args:
            oct_images: [B, 120, 3, 224, 224] OCT图像
            colposcopy_images: [B, 3, 3, 224, 224] Colposcopy图像
            clinical_features: [B, 7] 临床特征
            medical_texts: 医学文本描述（可选）
        
        Returns:
            output: 包含预测、因果图、不确定性、报告等的字典
        """
        B = oct_images.size(0)
        
        # 1. 特征提取（使用VLM）
        # OCT: 平均池化多帧
        oct_batch = oct_images.view(B * 120, 3, 224, 224)
        oct_vlm_features = self.vlm_encoder.encode_image(oct_batch)  # [B*120, 768]
        oct_vlm_features = oct_vlm_features.view(B, 120, -1).mean(dim=1)  # [B, 768]
        
        # Colposcopy: 平均池化多帧
        colpo_batch = colposcopy_images.view(B * 3, 3, 224, 224)
        colpo_vlm_features = self.vlm_encoder.encode_image(colpo_batch)  # [B*3, 768]
        colpo_vlm_features = colpo_vlm_features.view(B, 3, -1).mean(dim=1)  # [B, 768]
        
        # Clinical: 投影
        clinical_proj = self.clinical_proj(clinical_features)  # [B, 768]
        
        # 2. 贝叶斯编码
        oct_mean, oct_var = self.oct_encoder(oct_vlm_features)
        colpo_mean, colpo_var = self.colposcopy_encoder(colpo_vlm_features)
        clinical_mean, clinical_var = self.clinical_encoder(clinical_proj)
        
        # 训练时采样，推理时使用均值
        if self.training:
            oct_feat = oct_mean + torch.randn_like(oct_mean) * torch.sqrt(oct_var + 1e-8)
            colpo_feat = colpo_mean + torch.randn_like(colpo_mean) * torch.sqrt(colpo_var + 1e-8)
            clinical_feat = clinical_mean + torch.randn_like(clinical_mean) * torch.sqrt(clinical_var + 1e-8)
        else:
            oct_feat = oct_mean
            colpo_feat = colpo_mean
            clinical_feat = clinical_mean
        
        # 3. VLM引导的因果发现
        features_list = [oct_feat, colpo_feat, clinical_feat]
        
        # 准备图像用于VLM（使用第一帧OCT和第一帧Colposcopy）
        oct_first = oct_images[:, 0]  # [B, 3, 224, 224]
        colpo_first = colposcopy_images[:, 0]  # [B, 3, 224, 224]
        combined_images = torch.cat([oct_first, colpo_first], dim=0)  # [2B, 3, 224, 224]
        
        causal_adj, causal_info = self.causal_discovery(
            features_list,
            images=combined_images if self.use_vlm_guidance else None,
            medical_texts=medical_texts
        )
        
        # 4. 因果感知对齐
        # 准备文本特征（如果有医学知识库）
        if self.medical_kb is not None and medical_texts is not None:
            enhanced_texts = [self.medical_kb.enhance_text(t) for t in medical_texts]
            # 这里简化处理，实际需要文本编码器
            text_features = torch.zeros(B, 1, self.embed_dim, device=oct_feat.device)
        else:
            text_features = None
        
        # 构建多模态序列
        multimodal_seq = torch.stack([oct_feat, colpo_feat, clinical_feat], dim=1)  # [B, 3, 768]
        
        aligned_features, alignment_info = self.causal_alignment(
            multimodal_seq,
            text_features if text_features is not None else multimodal_seq,
            causal_graph=causal_adj
        )
        
        # 5. 不确定性分解
        variances = torch.stack([oct_var, colpo_var, clinical_var], dim=1)  # [B, 3, 768]
        uncertainty_info = self.uncertainty_decomp(aligned_features, variances.mean(dim=1))
        
        # 6. 分类
        logits = self.classifier(aligned_features)
        
        # 7. 生成诊断报告（可选）
        report = None
        if self.report_generator is not None:
            report = self.report_generator.generate(
                combined_images,
                logits,
                causal_adj,
                self.medical_kb
            )
        
        return {
            'logits': logits,
            'causal_adj': causal_adj,
            'uncertainty': uncertainty_info,
            'report': report,
            'causal_info': causal_info,
            'alignment_info': alignment_info
        }

