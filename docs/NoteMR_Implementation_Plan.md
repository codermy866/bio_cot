# NoteMR思想集成到Bio-COT 2.0的实施方案

## 📋 总体方案概述

### 三阶段实施计划

```
阶段1: 候选结果优化（1-2天，立即实施）
  ↓
阶段2: Prompt增强（2-3天，中期实施）
  ↓
阶段3: 视觉笔记生成（3-5天，长期实施）
```

### 预期效果

| 阶段 | 改进内容 | 预期AUC提升 | 实施难度 | 优先级 |
|------|---------|------------|---------|--------|
| **阶段1** | 候选结果优化 | +1-2% | ⭐ 低 | 🔴 最高 |
| **阶段2** | Prompt增强（MLLM） | +0.5-1% | ⭐⭐ 中 | 🟡 高 |
| **阶段3** | 视觉笔记生成 | +2-3% | ⭐⭐⭐ 高 | 🟢 中 |
| **组合** | 所有改进 | +3-5% | - | - |

---

## 🚀 阶段1：候选结果优化（立即实施）

### 1.1 目标

**无需修改训练流程，仅在推理时使用**，提升预测可靠性。

### 1.2 实施步骤

#### Step 1: 创建候选结果优化模块

**文件**：`src/models/bida/candidate_refinement.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
候选结果优化模块（借鉴NoteMR方法）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class CandidateGenerator(nn.Module):
    """
    候选预测生成器
    使用MC Dropout生成多个候选预测
    """
    
    def __init__(self, num_candidates: int = 5, temperature: float = 1.0):
        """
        Args:
            num_candidates: 候选数量（默认5）
            temperature: 温度参数（用于softmax，默认1.0）
        """
        super().__init__()
        self.num_candidates = num_candidates
        self.temperature = temperature
    
    def generate_candidates(self, model, features: torch.Tensor, num_candidates: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成多个候选预测
        
        Args:
            model: 分类模型（需要支持dropout）
            features: [B, D] 融合特征
            num_candidates: 候选数量（如果None，使用self.num_candidates）
        
        Returns:
            candidate_logits: [B, num_candidates, num_classes] 候选logits
            candidate_probs: [B, num_candidates, num_classes] 候选概率
        """
        if num_candidates is None:
            num_candidates = self.num_candidates
        
        B = features.shape[0]
        candidate_logits = []
        candidate_probs = []
        
        # 启用dropout（MC Dropout）
        model.train()  # 重要：启用dropout
        
        with torch.no_grad():  # 推理时不计算梯度
            for _ in range(num_candidates):
                logits = model.classifier(features)  # [B, num_classes]
                probs = F.softmax(logits / self.temperature, dim=-1)
                candidate_logits.append(logits)
                candidate_probs.append(probs)
        
        candidate_logits = torch.stack(candidate_logits, dim=1)  # [B, num_candidates, num_classes]
        candidate_probs = torch.stack(candidate_probs, dim=1)  # [B, num_candidates, num_classes]
        
        return candidate_logits, candidate_probs


class CandidateRefiner(nn.Module):
    """
    候选结果优化器
    选择最一致、最准确的候选结果
    """
    
    def __init__(self, consistency_threshold: float = 0.8):
        """
        Args:
            consistency_threshold: 一致性阈值（默认0.8）
        """
        super().__init__()
        self.consistency_threshold = consistency_threshold
    
    def refine_candidates(
        self, 
        candidate_probs: torch.Tensor, 
        candidate_logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        优化候选结果
        
        Args:
            candidate_probs: [B, num_candidates, num_classes] 候选概率
            candidate_logits: [B, num_candidates, num_classes] 候选logits
        
        Returns:
            dict: 包含优化后的概率、预测、一致性分数等
        """
        B, num_candidates, num_classes = candidate_probs.shape
        
        # 方法1: 平均一致性（Average Consensus）
        avg_probs = candidate_probs.mean(dim=1)  # [B, num_classes]
        
        # 方法2: 加权一致性（Weighted Consensus）
        # 根据每个候选的置信度加权
        confidences = candidate_probs.max(dim=-1)[0]  # [B, num_candidates]
        weights = F.softmax(confidences, dim=-1)  # [B, num_candidates]
        weighted_probs = (candidate_probs * weights.unsqueeze(-1)).sum(dim=1)  # [B, num_classes]
        
        # 方法3: 一致性投票（Consistency Voting）
        consistency_scores = []
        for i in range(B):
            probs_i = candidate_probs[i]  # [num_candidates, num_classes]
            # 计算两两之间的KL散度
            kl_divs = []
            for j in range(num_candidates):
                for k in range(j+1, num_candidates):
                    kl = F.kl_div(
                        F.log_softmax(probs_i[j], dim=0),
                        probs_i[k],
                        reduction='sum'
                    )
                    kl_divs.append(kl.item())
            
            if len(kl_divs) > 0:
                consistency_score = 1.0 / (1.0 + np.mean(kl_divs))  # 一致性分数（越高越一致）
            else:
                consistency_score = 1.0
            consistency_scores.append(consistency_score)
        
        consistency_scores = torch.tensor(consistency_scores, device=candidate_probs.device)
        
        # 选择最一致的候选
        # 如果一致性高，使用加权平均
        # 如果一致性低，使用最高置信度的候选
        refined_probs = torch.where(
            consistency_scores.unsqueeze(-1) > self.consistency_threshold,
            weighted_probs,
            candidate_probs.max(dim=1)[0]  # 使用最高置信度的候选
        )
        
        refined_preds = refined_probs.argmax(dim=-1)
        
        return {
            'refined_probs': refined_probs,
            'refined_preds': refined_preds,
            'consistency_scores': consistency_scores,
            'avg_probs': avg_probs,
            'weighted_probs': weighted_probs,
            'candidate_probs': candidate_probs
        }
```

#### Step 2: 修改Bio-COT v2模型，添加候选结果优化

**文件**：`src/models/bida/bio_cot_v2.py`（修改）

```python
# 在文件开头添加导入
from .candidate_refinement import CandidateGenerator, CandidateRefiner

# 在BioCOT_v2类的__init__方法中添加
class BioCOT_v2(nn.Module):
    def __init__(self, ..., use_candidate_refinement: bool = True, num_candidates: int = 5):
        super().__init__()
        # ... 现有代码 ...
        
        # 候选结果优化组件（新增）
        self.use_candidate_refinement = use_candidate_refinement
        if use_candidate_refinement:
            self.candidate_generator = CandidateGenerator(num_candidates=num_candidates)
            self.candidate_refiner = CandidateRefiner()
    
    def forward(self, ..., return_refined: bool = False):
        # ... 现有前向传播代码 ...
        
        # 分类预测
        logits = self.classifier(fused_feat)  # [B, num_classes]
        
        output = {
            'logits': logits,
            'z_causal': z_causal,
            'z_sem': z_sem,
        }
        
        # 候选结果优化（推理时使用）
        if return_refined and self.use_candidate_refinement and not self.training:
            # 生成候选预测
            candidate_logits, candidate_probs = self.candidate_generator.generate_candidates(
                self, fused_feat, num_candidates=self.candidate_generator.num_candidates
            )
            
            # 优化候选结果
            refined_results = self.candidate_refiner.refine_candidates(
                candidate_probs, candidate_logits
            )
            
            output.update(refined_results)
        
        return output
```

#### Step 3: 修改验证/测试脚本，使用候选结果优化

**文件**：`experiments/exp_5centers/train_bio_cot_v2.py`（修改validate函数）

```python
def validate(model, dataloader, criterion, device, epoch, args, log_print=None):
    """验证（加入候选结果优化）"""
    if log_print is None:
        log_print = print
    
    model.eval()  # 重要：设置为eval模式
    total_loss = 0.0
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_refined_preds = []  # 新增
    all_consistency_scores = []  # 新增
    
    log_print(f"\n  📊 Epoch {epoch}/{args.num_epochs} - 验证阶段")
    log_print(f"     总batch数: {len(dataloader)}")
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.num_epochs} [Val]')
        
        for batch_idx, batch in enumerate(pbar):
            # ... 现有代码 ...
            
            # 前向传播（加入return_refined=True）
            outputs = model(
                oct_features=oct_features,
                colpo_features=colpo_features,
                clinical_embeddings=clinical_embeddings,
                clinical_data=clinical_data,
                center_labels=center_labels,
                return_loss_components=False,
                return_refined=True  # 新增：启用候选结果优化
            )
            
            logits = outputs['logits']
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # 使用优化后的预测（如果可用）
            if 'refined_probs' in outputs:
                refined_probs = outputs['refined_probs']
                refined_preds = outputs['refined_preds']
                consistency_scores = outputs['consistency_scores']
                
                all_refined_preds.extend(refined_preds.detach().cpu().numpy())
                all_consistency_scores.extend(consistency_scores.detach().cpu().numpy())
                all_probs.extend(refined_probs[:, 1].detach().cpu().numpy())  # 使用优化后的概率
            else:
                # Fallback：使用原始预测
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_probs.extend(probs[:, 1].detach().cpu().numpy())
            
            # ... 现有代码 ...
    
    # 计算指标（使用优化后的预测）
    if len(all_refined_preds) > 0:
        acc = accuracy_score(all_labels, all_refined_preds)
        preds_for_auc = all_refined_preds
    else:
        acc = accuracy_score(all_labels, all_preds)
        preds_for_auc = all_preds
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    try:
        f1 = f1_score(all_labels, preds_for_auc)
    except:
        f1 = 0.0
    
    # 打印一致性分数统计
    if len(all_consistency_scores) > 0:
        avg_consistency = np.mean(all_consistency_scores)
        log_print(f"     - 平均一致性分数: {avg_consistency:.4f}")
    
    log_print(f"\n  📊 Epoch {epoch} 验证统计:")
    log_print(f"     - 平均损失: {avg_loss:.6f}")
    log_print(f"     - 准确率: {acc:.4f}")
    log_print(f"     - AUC: {auc:.4f}")
    log_print(f"     - F1-Score: {f1:.4f}")
    
    return {
        'loss': avg_loss,
        'acc': acc,
        'auc': auc,
        'f1': f1,
        'all_preds': preds_for_auc,
        'all_labels': all_labels,
        'all_probs': all_probs,
        'consistency_scores': all_consistency_scores if len(all_consistency_scores) > 0 else None
    }
```

### 1.3 测试验证

**创建测试脚本**：`experiments/exp_5centers/test_candidate_refinement.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试候选结果优化模块
"""

import torch
from src.models.bida.candidate_refinement import CandidateGenerator, CandidateRefiner

# 测试代码
def test_candidate_refinement():
    """测试候选结果优化"""
    B, D, num_classes = 4, 768, 2
    
    # 模拟模型
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(D, num_classes),
                nn.Dropout(0.2)  # 重要：需要有dropout
            )
        
        def forward(self, x):
            return self.classifier(x)
    
    model = MockModel()
    features = torch.randn(B, D)
    
    # 测试候选生成
    generator = CandidateGenerator(num_candidates=5)
    candidate_logits, candidate_probs = generator.generate_candidates(model, features)
    
    print(f"候选logits形状: {candidate_logits.shape}")  # [B, 5, 2]
    print(f"候选概率形状: {candidate_probs.shape}")  # [B, 5, 2]
    
    # 测试候选优化
    refiner = CandidateRefiner()
    refined_results = refiner.refine_candidates(candidate_probs, candidate_logits)
    
    print(f"优化后的概率形状: {refined_results['refined_probs'].shape}")  # [B, 2]
    print(f"优化后的预测形状: {refined_results['refined_preds'].shape}")  # [B]
    print(f"一致性分数形状: {refined_results['consistency_scores'].shape}")  # [B]
    
    print("✅ 测试通过！")

if __name__ == '__main__':
    test_candidate_refinement()
```

### 1.4 预期效果

- **AUC提升**：+1-2%
- **准确率提升**：+1-2%
- **不确定性量化**：一致性分数可以作为不确定性指标

---

## 🚀 阶段2：Prompt增强（中期实施）

### 2.1 目标

使用MLLM增强当前Prompt，提升语义质量。

### 2.2 实施步骤

#### Step 1: 创建增强版Prompt生成器

**文件**：`experiments/exp_5centers/enhanced_prompt_generator.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版Prompt生成器（借鉴NoteMR方法）
使用MLLM增强语义理解
"""

import torch
from transformers import AutoModel, AutoTokenizer
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

from experiments.exp_5centers.preprocess_llm import PromptGenerator


class EnhancedPromptGenerator:
    """
    增强版Prompt生成器
    使用冻结的医学MLLM增强语义理解
    """
    
    def __init__(
        self,
        mllm_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        device: str = "cuda:0",
        use_8bit: bool = False
    ):
        """
        Args:
            mllm_model_name: 医学MLLM模型名称
            device: 设备
            use_8bit: 是否使用8bit量化
        """
        self.device = device
        self.mllm_model_name = mllm_model_name
        
        print(f"📥 正在加载医学MLLM: {mllm_model_name}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(mllm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        try:
            if use_8bit and "cuda" in device:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModel.from_pretrained(
                    mllm_model_name,
                    device_map="auto",
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16
                )
            else:
                self.model = AutoModel.from_pretrained(mllm_model_name)
                self.model = self.model.to(device)
                self.model.eval()
            
            # 冻结参数
            for param in self.model.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            print("   使用CPU模式...")
            self.device = "cpu"
            self.model = AutoModel.from_pretrained(mllm_model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        
        print(f"✅ MLLM加载完成")
    
    def generate_enhanced_prompt(self, age: int, hpv: int, tct: int) -> str:
        """
        生成增强版Prompt
        
        Args:
            age: 年龄
            hpv: HPV状态
            tct: TCT结果
        
        Returns:
            enhanced_prompt: 增强后的Prompt
        """
        # Step 1: 生成基础Prompt
        base_prompt = PromptGenerator.generate_clinical_prompt(age, hpv, tct)
        
        # Step 2: 使用MLLM增强语义理解
        enhancement_prompt = f"""
Given the following patient clinical information:
{base_prompt}

Please provide a concise medical summary that:
1. Highlights key risk factors for cervical cancer screening
2. Identifies relevant clinical patterns
3. Suggests important considerations for diagnosis

Medical Summary:
"""
        
        # Tokenize
        inputs = self.tokenizer(
            enhancement_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成增强摘要（使用模型的pooler_output或last_hidden_state）
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # 获取特征（用于后续编码）
            if hasattr(outputs, 'pooler_output'):
                enhanced_features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                # 平均池化
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_hidden = outputs.last_hidden_state * attention_mask
                enhanced_features = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
            else:
                enhanced_features = outputs[0][:, 0, :]
        
        # Step 3: 组合基础Prompt和增强特征
        # 注意：这里我们返回增强后的特征，而不是文本
        # 实际使用时，可以直接使用enhanced_features作为嵌入
        
        # 为了兼容性，我们也可以生成文本摘要（如果需要）
        # 但更推荐直接使用特征向量
        
        return base_prompt, enhanced_features.squeeze(0).cpu().numpy()
    
    def generate_enhanced_embeddings_batch(
        self, 
        ages: list, 
        hpvs: list, 
        tcts: list,
        batch_size: int = 8
    ) -> dict:
        """
        批量生成增强嵌入
        
        Args:
            ages: 年龄列表
            hpvs: HPV状态列表
            tcts: TCT结果列表
            batch_size: 批处理大小
        
        Returns:
            embeddings_dict: {patient_id: embedding} 字典
        """
        embeddings_dict = {}
        
        for i in range(0, len(ages), batch_size):
            batch_ages = ages[i:i+batch_size]
            batch_hpvs = hpvs[i:i+batch_size]
            batch_tcts = tcts[i:i+batch_size]
            
            batch_prompts = []
            for age, hpv, tct in zip(batch_ages, batch_hpvs, batch_tcts):
                base_prompt = PromptGenerator.generate_clinical_prompt(age, hpv, tct)
                batch_prompts.append(base_prompt)
            
            # 批量处理
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                if hasattr(outputs, 'pooler_output'):
                    batch_embeddings = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    masked_hidden = outputs.last_hidden_state * attention_mask
                    batch_embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
                else:
                    batch_embeddings = outputs[0][:, 0, :]
            
            # 保存嵌入
            for j, (age, hpv, tct) in enumerate(zip(batch_ages, batch_hpvs, batch_tcts)):
                patient_id = f"{age}_{hpv}_{tct}_{i+j}"
                embeddings_dict[patient_id] = batch_embeddings[j].cpu().numpy()
        
        return embeddings_dict
```

#### Step 2: 修改预处理脚本，使用增强版Prompt

**文件**：`experiments/exp_5centers/preprocess_llm.py`（修改）

```python
# 在文件开头添加导入
from experiments.exp_5centers.enhanced_prompt_generator import EnhancedPromptGenerator

# 修改preprocess_all_samples函数
def preprocess_all_samples(
    excel_path: str = None,
    csv_paths: list = None,
    output_path: str = None,
    model_name: str = "bert-base-uncased",
    device: str = "cuda:0",
    use_8bit: bool = False,
    batch_size: int = 32,
    use_enhanced_prompt: bool = False  # 新增参数
):
    """
    预处理所有样本，生成LLM嵌入
    
    Args:
        use_enhanced_prompt: 是否使用增强版Prompt（新增）
    """
    # ... 现有代码 ...
    
    # 选择Prompt生成器
    if use_enhanced_prompt:
        print("✅ 使用增强版Prompt生成器（MLLM增强）")
        prompt_generator = EnhancedPromptGenerator(
            mllm_model_name=model_name,
            device=device,
            use_8bit=use_8bit
        )
    else:
        print("✅ 使用标准Prompt生成器")
        prompt_generator = PromptGenerator()
    
    # ... 后续处理代码 ...
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取嵌入"):
        try:
            # ... 获取临床数据 ...
            
            # 生成Prompt（根据use_enhanced_prompt选择）
            if use_enhanced_prompt:
                prompt, enhanced_embedding = prompt_generator.generate_enhanced_prompt(age, hpv, tct)
                embedding = enhanced_embedding  # 直接使用增强后的嵌入
            else:
                prompt = prompt_generator.generate_clinical_prompt(age, hpv, tct)
                embedding = extractor.extract_embedding(prompt)
            
            # ... 保存嵌入 ...
```

#### Step 3: 重新生成LLM嵌入

**运行命令**：

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics
source my_retfound/bin/activate

python experiments/exp_5centers/preprocess_llm.py \
    --csv_paths data/5centers_multi_leave_centers_out/train_labels.csv \
                 data/5centers_multi_leave_centers_out/val_labels.csv \
                 data/5centers_multi_leave_centers_out/external_test_labels.csv \
    --output_path experiments/exp_5centers/data/clinical_embeddings_enhanced.pkl \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
    --device cuda:1 \
    --use_enhanced_prompt \
    --batch_size 16
```

### 2.3 预期效果

- **Prompt质量提升**：+30-50%
- **AUC提升**：+0.5-1%
- **语义丰富性**：显著提升

---

## 🚀 阶段3：视觉笔记生成（长期实施）

### 3.1 目标

使用GradCAM生成视觉笔记，突出关键区域。

### 3.2 实施步骤

#### Step 1: 创建跨模态GradCAM模块

**文件**：`src/models/bida/cross_modal_gradcam.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨模态GradCAM（借鉴NoteMR方法）
计算图像patch与临床特征之间的注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class CrossModalGradCAM:
    """
    跨模态GradCAM
    计算图像patch与临床特征/知识笔记之间的注意力
    """
    
    def __init__(self, model, image_encoder_layer_name: str = None):
        """
        Args:
            model: Bio-COT模型
            image_encoder_layer_name: 图像编码器的目标层名称
        """
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.target_layer = None
        
        # 注册钩子
        self._register_hooks(image_encoder_layer_name)
    
    def _register_hooks(self, layer_name: Optional[str]):
        """注册前向和反向钩子"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
        
        # 找到目标层
        if layer_name:
            named_modules = dict([*self.model.named_modules()])
            if layer_name in named_modules:
                self.target_layer = named_modules[layer_name]
            else:
                raise ValueError(f"找不到指定的层: {layer_name}")
        else:
            # 自动查找图像编码器的最后一层
            if hasattr(self.model, 'image_encoder'):
                if isinstance(self.model.image_encoder, nn.Sequential):
                    self.target_layer = self.model.image_encoder[-1]
                else:
                    # 查找最后一个Linear层
                    for name, module in self.model.image_encoder.named_modules():
                        if isinstance(module, nn.Linear):
                            self.target_layer = module
            else:
                raise ValueError("无法找到图像编码器")
        
        # 注册钩子
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def compute_attention(
        self, 
        images: torch.Tensor, 
        clinical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算跨模态注意力
        
        Args:
            images: [B, C, H, W] 图像
            clinical_features: [B, D] 临床特征（或知识笔记特征）
        
        Returns:
            attention_map: [B, M] 注意力分数（M是patch数量）
        """
        self.model.zero_grad()
        self.gradients = None
        self.activations = None
        
        # 前向传播
        # 注意：这里需要根据实际模型结构调整
        # 假设模型有extract_features方法
        img_features = self.model.image_encoder(images)  # [B, M, D] 或 [B, D]
        
        # 计算跨模态相似度
        if len(img_features.shape) == 3:  # [B, M, D]
            # 图像特征投影
            img_proj = self.model.img_projection(img_features)  # [B, M, D]
            # 临床特征投影
            clin_proj = self.model.clin_projection(clinical_features)  # [B, D]
            # 计算注意力
            attention_scores = torch.matmul(
                img_proj, clin_proj.unsqueeze(1)
            ).squeeze(-1)  # [B, M]
        else:  # [B, D]
            # 如果图像特征已经是全局特征，需要重新提取patch特征
            # 这里简化处理，实际需要根据模型结构调整
            attention_scores = torch.ones(images.shape[0], 196, device=images.device)  # 假设196个patch
        
        attention_scores = F.softmax(attention_scores, dim=-1)
        
        # 反向传播计算梯度
        loss = attention_scores.sum()
        loss.backward()
        
        # 使用GradCAM计算注意力权重
        if self.gradients is not None and self.activations is not None:
            # 加权平均
            weights = self.gradients.mean(dim=-1, keepdim=True)  # [B, M, 1] 或 [B, 1]
            if len(weights.shape) == 3:
                cam = (weights * self.activations).sum(dim=-1)  # [B, M]
            else:
                cam = (weights * self.activations).sum(dim=-1)  # [B]
            cam = F.relu(cam)
            cam = cam / (cam.max(dim=-1, keepdim=True)[0] + 1e-8)
            
            # 结合跨模态注意力
            final_attention = (attention_scores + cam) / 2
        else:
            final_attention = attention_scores
        
        return final_attention


class VisualNotesGenerator:
    """
    视觉笔记生成器
    根据跨模态注意力生成掩码，突出关键区域
    """
    
    def __init__(self, patch_size: int = 16, threshold: float = 0.6):
        """
        Args:
            patch_size: Patch大小（ViT默认16）
            threshold: 注意力阈值（默认0.6）
        """
        self.patch_size = patch_size
        self.threshold = threshold
    
    def generate_visual_notes(
        self, 
        images: torch.Tensor, 
        attention_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成视觉笔记
        
        Args:
            images: [B, C, H, W] 原始图像
            attention_map: [B, M] 注意力分数（M是patch数量）
        
        Returns:
            visual_notes: [B, C, H, W] 视觉笔记（掩码后的图像）
            mask: [B, H, W] 二进制掩码
        """
        B, C, H, W = images.shape
        M = attention_map.shape[1]  # patch数量
        
        # 将注意力分数reshape为空间掩码
        # 假设图像被划分为 sqrt(M) × sqrt(M) 个patch
        patch_h = int(np.sqrt(M))
        patch_w = int(np.sqrt(M))
        
        attention_2d = attention_map.view(B, patch_h, patch_w)
        
        # 上采样到原始图像尺寸
        attention_2d = F.interpolate(
            attention_2d.unsqueeze(1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [B, H, W]
        
        # 生成二进制掩码
        mask = (attention_2d > self.threshold).float()  # [B, H, W]
        
        # 应用掩码到图像
        visual_notes = images * mask.unsqueeze(1)  # [B, C, H, W]
        
        return visual_notes, mask
```

#### Step 2: 集成到Bio-COT v2模型

**文件**：`src/models/bida/bio_cot_v2.py`（修改）

```python
# 在文件开头添加导入
from .cross_modal_gradcam import CrossModalGradCAM, VisualNotesGenerator

# 在BioCOT_v2类的__init__方法中添加
class BioCOT_v2(nn.Module):
    def __init__(self, ..., use_visual_notes: bool = True):
        super().__init__()
        # ... 现有代码 ...
        
        # 视觉笔记组件（新增）
        self.use_visual_notes = use_visual_notes
        if use_visual_notes:
            self.gradcam = CrossModalGradCAM(self)
            self.visual_notes_gen = VisualNotesGenerator(patch_size=16, threshold=0.6)
            
            # 投影层（用于跨模态注意力）
            self.img_projection = nn.Linear(self.embed_dim, self.embed_dim)
            self.clin_projection = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, oct_images, colpo_images, ..., use_visual_notes: bool = False):
        # ... 现有代码 ...
        
        # 生成视觉笔记（如果启用）
        if use_visual_notes and self.use_visual_notes:
            # 对OCT图像
            oct_attention = self.gradcam.compute_attention(oct_images, z_sem)
            oct_visual_notes, oct_mask = self.visual_notes_gen.generate_visual_notes(
                oct_images, oct_attention
            )
            
            # 对Colposcopy图像
            colpo_attention = self.gradcam.compute_attention(colpo_images, z_sem)
            colpo_visual_notes, colpo_mask = self.visual_notes_gen.generate_visual_notes(
                colpo_images, colpo_attention
            )
            
            # 使用视觉笔记提取特征（而不是原始图像）
            oct_features = extract_features_with_vit(oct_visual_notes, device)
            colpo_features = extract_features_with_vit(colpo_visual_notes, device)
        else:
            # 使用原始图像
            oct_features = extract_features_with_vit(oct_images, device)
            colpo_features = extract_features_with_vit(colpo_images, device)
        
        # ... 后续流程 ...
```

### 3.3 预期效果

- **AUC提升**：+2-3%
- **关键区域突出**：减少幻觉，提升细粒度感知

---

## 📊 完整实施时间表

| 阶段 | 内容 | 时间 | 优先级 | 状态 |
|------|------|------|--------|------|
| **阶段1** | 候选结果优化 | 1-2天 | 🔴 最高 | ⬜ 待实施 |
| **阶段2** | Prompt增强 | 2-3天 | 🟡 高 | ⬜ 待实施 |
| **阶段3** | 视觉笔记生成 | 3-5天 | 🟢 中 | ⬜ 待实施 |
| **总计** | 所有改进 | 6-10天 | - | - |

---

## ✅ 实施检查清单

### 阶段1检查清单

- [ ] 创建`candidate_refinement.py`文件
- [ ] 修改`bio_cot_v2.py`，添加候选结果优化
- [ ] 修改`train_bio_cot_v2.py`，在验证时使用候选结果优化
- [ ] 运行测试脚本验证功能
- [ ] 重新训练模型，验证AUC提升

### 阶段2检查清单

- [ ] 创建`enhanced_prompt_generator.py`文件
- [ ] 修改`preprocess_llm.py`，支持增强版Prompt
- [ ] 重新生成LLM嵌入（使用增强版Prompt）
- [ ] 重新训练模型，验证AUC提升

### 阶段3检查清单

- [ ] 创建`cross_modal_gradcam.py`文件
- [ ] 修改`bio_cot_v2.py`，添加视觉笔记生成
- [ ] 修改训练脚本，支持视觉笔记
- [ ] 重新训练模型，验证AUC提升

---

## 🎯 预期最终效果

### 性能提升

| 指标 | 基线 | 阶段1后 | 阶段2后 | 阶段3后 | 总提升 |
|------|------|---------|---------|---------|--------|
| **AUC** | 0.84 | 0.85-0.86 | 0.855-0.87 | 0.87-0.89 | +3-5% |
| **准确率** | 78% | 79-80% | 79.5-81% | 81-83% | +3-5% |
| **F1-Score** | 0.66 | 0.67-0.68 | 0.675-0.69 | 0.69-0.71 | +3-5% |

### 创新性提升

- ✅ **候选结果优化**：借鉴NoteMR，提升预测可靠性
- ✅ **Prompt增强**：使用MLLM增强语义理解
- ✅ **视觉笔记**：突出关键区域，减少幻觉

---

**文档版本**：v1.0  
**最后更新**：2025-01-08  
**实施状态**：待开始

