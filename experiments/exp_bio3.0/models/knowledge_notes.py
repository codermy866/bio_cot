#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识笔记生成模块 (Knowledge Notes Generation)
借鉴NoteMR方法，结合外部医学知识库生成诊断摘要
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import numpy as np

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    print("⚠️ transformers未安装，请安装: pip install transformers")


class KnowledgeRetriever:
    """
    医学知识检索器
    根据临床数据从知识库中检索相关指南
    """
    
    def __init__(self, knowledge_base_path: str):
        """
        Args:
            knowledge_base_path: 知识库JSON文件路径
        """
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
    
    def _load_knowledge_base(self, path: str) -> Dict[str, str]:
        """加载知识库"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def retrieve(
        self, 
        clinical_data: Dict,
        top_k: int = 5
    ) -> List[str]:
        """
        检索相关医学知识
        
        Args:
            clinical_data: dict {'hpv': int, 'tct': str/int, 'age': int}
            top_k: 检索top-k条指南
        
        Returns:
            retrieved_knowledge: 检索到的知识文本列表
        """
        hpv = clinical_data.get('hpv', 0)
        tct = clinical_data.get('tct', 'NILM')
        age = clinical_data.get('age', 50)
        
        # 构建查询键
        query_keys = []
        
        # 1. HPV相关
        if hpv == 1:
            query_keys.append("HPV_HR_POS_Age>30")
            if age > 30:
                query_keys.append("HPV16_POS_Age>30")
        
        # 2. TCT相关
        if isinstance(tct, str):
            tct_upper = tct.upper()
            if 'NILM' in tct_upper:
                query_keys.append("TCT_NILM_HPV_POS" if hpv == 1 else "TCT_NILM")
            elif 'ASCUS' in tct_upper or 'ASC-US' in tct_upper:
                query_keys.append("TCT_ASCUS_HPV_POS" if hpv == 1 else "TCT_ASCUS")
            elif 'LSIL' in tct_upper:
                query_keys.append("TCT_LSIL_HPV_POS" if hpv == 1 else "TCT_LSIL")
            elif 'HSIL' in tct_upper:
                query_keys.append("TCT_HSIL")
        elif isinstance(tct, int):
            if tct == 0:
                query_keys.append("TCT_NILM_HPV_POS" if hpv == 1 else "TCT_NILM")
            elif tct == 1:
                query_keys.append("TCT_ASCUS_HPV_POS" if hpv == 1 else "TCT_ASCUS")
            elif tct == 2:
                query_keys.append("TCT_LSIL_HPV_POS" if hpv == 1 else "TCT_LSIL")
            elif tct >= 3:
                query_keys.append("TCT_HSIL")
        
        # 3. 年龄相关
        if age < 21:
            query_keys.append("Age<21")
        elif 21 <= age <= 24:
            query_keys.append("Age_21-24")
        elif age > 65:
            query_keys.append("Age>65")
        
        # 4. 组合状态
        if hpv == 1 and isinstance(tct, str) and 'LSIL' in tct.upper():
            query_keys.append("HPV16_POS_TCT_LSIL")
        if hpv == 1 and isinstance(tct, str) and 'ASCUS' in tct.upper():
            query_keys.append("HPV18_POS_TCT_ASCUS")
        
        # 5. 高风险组合
        if (hpv == 1 and isinstance(tct, (str, int)) and 
            (isinstance(tct, str) and tct.upper() not in ['NILM', 'NORMAL'] or 
             isinstance(tct, int) and tct > 0)):
            query_keys.append("HIGH_RISK_COMBINATION")
        
        # 检索知识（去重）
        retrieved_knowledge = []
        seen_keys = set()
        
        for key in query_keys:
            if key in self.knowledge_base and key not in seen_keys:
                retrieved_knowledge.append(self.knowledge_base[key])
                seen_keys.add(key)
                if len(retrieved_knowledge) >= top_k:
                    break
        
        return retrieved_knowledge


class KnowledgeNotesGenerator(nn.Module):
    """
    知识笔记生成器
    使用冻结的医学LLM生成诊断摘要
    """
    
    def __init__(
        self,
        llm_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        device: str = "cuda:0",
        use_8bit: bool = False,
        embed_dim: int = 768
    ):
        """
        Args:
            llm_model_name: 医学LLM模型名称
            device: 设备
            use_8bit: 是否使用8bit量化
            embed_dim: 输出嵌入维度
        """
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型（冻结参数）
        try:
            if use_8bit and "cuda" in device:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModel.from_pretrained(
                    llm_model_name,
                    device_map="auto",
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16
                )
            else:
                self.model = AutoModel.from_pretrained(llm_model_name)
                self.model = self.model.to(device)
                self.model.eval()
            
            # 冻结参数
            for param in self.model.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            self.device = "cpu"
            self.model = AutoModel.from_pretrained(llm_model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        
        # 文本投影器（将LLM嵌入投影到目标维度）
        self.text_projector = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, embed_dim)
        )
    
    def generate_note_text(
        self,
        clinical_data: Dict,
        retrieved_knowledge: List[str]
    ) -> str:
        """
        生成诊断摘要文本
        
        Args:
            clinical_data: 临床数据
            retrieved_knowledge: 检索到的知识列表
        
        Returns:
            note_text: 诊断摘要文本
        """
        hpv = clinical_data.get('hpv', 0)
        tct = clinical_data.get('tct', 'NILM')
        age = clinical_data.get('age', 50)
        
        hpv_desc = "Positive" if hpv == 1 else "Negative"
        tct_desc = str(tct) if isinstance(tct, str) else f"TCT_{tct}"
        
        # 构建Prompt
        knowledge_text = "\n".join([f"- {k}" for k in retrieved_knowledge])
        
        prompt = f"""Patient Profile: A {age}-year-old female patient.
HPV Status: {hpv_desc}.
Cytology Result: {tct_desc}.

Retrieved Medical Guidelines:
{knowledge_text}

Based on the patient's clinical information and the above guidelines, provide a concise diagnostic summary that:
1. Highlights key risk factors for cervical cancer screening
2. Identifies relevant clinical patterns
3. Suggests important considerations for diagnosis

Diagnostic Summary:"""
        
        return prompt
    
    def forward(
        self,
        clinical_data: Dict,
        retrieved_knowledge: List[str]
    ) -> torch.Tensor:
        """
        生成知识笔记嵌入
        
        Args:
            clinical_data: 临床数据
            retrieved_knowledge: 检索到的知识列表
        
        Returns:
            z_sem: [B, embed_dim] 语义锚点特征
        """
        # 生成诊断摘要文本
        note_text = self.generate_note_text(clinical_data, retrieved_knowledge)
        
        # Tokenize
        inputs = self.tokenizer(
            note_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取嵌入
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # 获取特征
            if hasattr(outputs, 'pooler_output'):
                llm_embedding = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                # 平均池化
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_hidden = outputs.last_hidden_state * attention_mask
                llm_embedding = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
            else:
                llm_embedding = outputs[0][:, 0, :]
        
        # 投影到目标维度
        z_sem = self.text_projector(llm_embedding)
        
        return z_sem


class KnowledgeNotesModule(nn.Module):
    """
    知识笔记模块（整合检索器和生成器）
    """
    
    def __init__(
        self,
        knowledge_base_path: str,
        llm_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        device: str = "cuda:0",
        use_8bit: bool = False,
        embed_dim: int = 768,
        top_k: int = 5
    ):
        """
        Args:
            knowledge_base_path: 知识库路径
            llm_model_name: LLM模型名称
            device: 设备
            use_8bit: 是否使用8bit量化
            embed_dim: 嵌入维度
            top_k: 检索top-k条指南
        """
        super().__init__()
        self.top_k = top_k
        
        # 知识检索器
        self.retriever = KnowledgeRetriever(knowledge_base_path)
        
        # 知识笔记生成器
        self.generator = KnowledgeNotesGenerator(
            llm_model_name=llm_model_name,
            device=device,
            use_8bit=use_8bit,
            embed_dim=embed_dim
        )
    
    def forward(self, clinical_data: Dict) -> Tuple[torch.Tensor, str]:
        """
        生成知识笔记
        
        Args:
            clinical_data: 临床数据字典
        
        Returns:
            z_sem: [1, embed_dim] 语义锚点特征
            note_text: 诊断摘要文本
        """
        # 检索知识
        retrieved_knowledge = self.retriever.retrieve(clinical_data, top_k=self.top_k)
        
        # 生成知识笔记嵌入
        z_sem = self.generator(clinical_data, retrieved_knowledge)
        
        # 生成文本（用于可视化）
        note_text = self.generator.generate_note_text(clinical_data, retrieved_knowledge)
        
        return z_sem, note_text

