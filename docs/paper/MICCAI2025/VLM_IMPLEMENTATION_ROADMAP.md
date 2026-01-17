# VLM增强方案实施路线图

## 🎯 快速开始指南

### 第一步：环境准备

```bash
# 1. 安装VLM相关依赖
pip install transformers>=4.40.0
pip install qwen-vl-utils  # Qwen-VL工具
# 或
pip install llava-med  # LLaVA-Med（如果使用）

# 2. 安装向量数据库（用于知识库）
pip install faiss-cpu  # 或 faiss-gpu
pip install sentence-transformers  # 用于文本向量化

# 3. 验证安装
python -c "from transformers import Qwen2VLForConditionalGeneration; print('OK')"
```

### 第二步：模型选择

#### 选项1：Qwen-VL（推荐，中文友好）

**优势**：
- 支持中文医学文本
- 性能优秀
- 开源免费
- 支持多图像输入

**模型选择**：
- `Qwen/Qwen2-VL-2B-Instruct`：轻量级，适合快速实验
- `Qwen/Qwen2-VL-7B-Instruct`：性能更强，需要更多显存

**代码示例**：
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
```

#### 选项2：LLaVA-Med（医学专用）

**优势**：
- 专门针对医学领域训练
- 医学术语理解能力强

**注意**：需要额外安装和配置

#### 选项3：GPT-4V（如果预算允许）

**优势**：
- 性能最强
- 但需要API调用

---

## 📋 实施步骤

### 阶段1：基础VLM集成（1周）

#### 任务清单

- [ ] **Day 1-2**: 安装和测试VLM模型
  ```bash
  # 测试Qwen-VL
  python code/test_vlm.py
  ```

- [ ] **Day 3-4**: 实现VLMImageEncoder
  - 完成`vlm_enhanced_causal_clip.py`中的`VLMImageEncoder`类
  - 测试图像编码功能

- [ ] **Day 5-7**: 集成到现有框架
  - 替换现有的Swin-T编码器为VLM编码器
  - 确保特征维度一致（768维）
  - 基础训练测试

#### 交付物
- ✅ VLM编码器实现
- ✅ 基础训练脚本
- ✅ 性能对比（VLM vs Swin-T）

---

### 阶段2：医学知识库构建（1-2周）

#### 任务清单

- [ ] **Week 1**: 知识收集
  - 收集宫颈癌筛查指南（ASCCP, ACOG, 中国指南）
  - 收集相关PubMed文献摘要
  - 整理专家知识（如果有）

- [ ] **Week 2**: 知识库实现
  - 实现`MedicalKnowledgeBase`类
  - 构建向量数据库
  - 实现检索和增强功能

#### 知识库结构

```
data/medical_knowledge/
├── guidelines/
│   ├── asccp_2024.txt
│   ├── acog_2023.txt
│   └── china_guidelines_2024.txt
├── literature/
│   ├── pubmed_abstracts.json
│   └── systematic_reviews.json
└── expert_knowledge/
    └── causal_rules.json
```

#### 交付物
- ✅ 医学知识库数据
- ✅ 知识库实现代码
- ✅ 检索功能测试

---

### 阶段3：因果感知对齐（1周）

#### 任务清单

- [ ] **Day 1-3**: 实现CausalAwareAlignment
  - 完成对齐网络
  - 实现因果图约束

- [ ] **Day 4-5**: 训练和测试
  - 集成到主模型
  - 训练测试

- [ ] **Day 6-7**: 评估对齐效果
  - 可视化对齐结果
  - 评估对齐质量

#### 交付物
- ✅ 因果感知对齐模块
- ✅ 对齐效果评估报告

---

### 阶段4：VLM引导因果发现（1-2周）

#### 任务清单

- [ ] **Week 1**: 实现VLMGuidedCausalDiscovery
  - 实现语义提取
  - 实现因果线索提取
  - 实现VLM引导机制

- [ ] **Week 2**: 验证和优化
  - 与专家标注的因果图对比
  - 优化引导机制
  - 评估因果发现准确性

#### 交付物
- ✅ VLM引导因果发现模块
- ✅ 因果发现评估报告
- ✅ 与专家标注的对比结果

---

### 阶段5：诊断报告生成（1周）

#### 任务清单

- [ ] **Day 1-3**: 实现MedicalReportGenerator
  - 实现报告模板
  - 实现VLM生成功能
  - 实现知识库增强

- [ ] **Day 4-5**: 报告质量评估
  - 自动评估（BLEU, ROUGE）
  - 临床专家评估（如果可能）

- [ ] **Day 6-7**: 优化和改进
  - 根据反馈优化
  - 改进报告模板

#### 交付物
- ✅ 诊断报告生成器
- ✅ 报告质量评估结果

---

### 阶段6：完整实验（2-3周）

#### 任务清单

- [ ] **Week 1**: 运行所有实验
  - 实验1：VLM vs CLIP对比
  - 实验2：知识库影响
  - 实验3：因果发现验证
  - 实验4：可解释性评估
  - 实验5：多中心验证

- [ ] **Week 2**: 数据分析和可视化
  - 结果统计分析
  - 生成可视化图表
  - 准备论文图表

- [ ] **Week 3**: 论文撰写
  - 撰写方法部分
  - 撰写结果部分
  - 撰写讨论部分

#### 交付物
- ✅ 完整实验结果
- ✅ 可视化图表
- ✅ 论文初稿

---

## 🔧 关键技术细节

### 1. VLM模型选择建议

#### 如果显存充足（>24GB）
- 使用 `Qwen/Qwen2-VL-7B-Instruct`
- 或 `LLaVA-Med-7B`

#### 如果显存有限（<24GB）
- 使用 `Qwen/Qwen2-VL-2B-Instruct`
- 或使用量化版本

#### 量化方案
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    quantization_config=quantization_config
)
```

### 2. 医学知识库构建

#### 向量数据库选择

**选项1：FAISS（推荐）**
```python
import faiss
from sentence_transformers import SentenceTransformer

# 加载编码器
encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 构建索引
dimension = 384
index = faiss.IndexFlatL2(dimension)

# 添加向量
vectors = encoder.encode(knowledge_texts)
index.add(vectors)
```

**选项2：Chroma（更易用）**
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("medical_knowledge")

# 添加文档
collection.add(
    documents=knowledge_texts,
    ids=[f"doc_{i}" for i in range(len(knowledge_texts))]
)
```

### 3. 训练策略

#### 分阶段训练

**阶段1：VLM微调（可选）**
```python
# 如果需要在医学数据上微调VLM
# 冻结大部分层，只微调最后几层
for param in model.parameters():
    param.requires_grad = False

for param in model.vision_model.layers[-2:].parameters():
    param.requires_grad = True
```

**阶段2：端到端训练**
```python
# 训练整个框架
# 使用较小的学习率
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,  # 较小的学习率
    weight_decay=1e-4
)
```

### 4. 内存优化

#### 梯度检查点
```python
from torch.utils.checkpoint import checkpoint

# 在forward中使用
output = checkpoint(self.vlm_encoder.encode_image, images)
```

#### 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(...)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 📊 评估指标

### 性能指标
- AUC, Accuracy, Sensitivity, Specificity, F1-Score

### 可解释性指标
- **自动评估**:
  - BLEU, ROUGE, METEOR（报告质量）
  - 因果图与专家标注的一致性

- **人工评估**（如果可能）:
  - 临床专家评分（1-5分）
  - 准确性、完整性、可理解性、实用性

### 因果发现指标
- **因果图相似度**: 与专家标注的F1-Score
- **因果效应估计**: 估计的因果效应是否合理
- **反事实推理**: 反事实预测的准确性

---

## ⚠️ 常见问题与解决方案

### 问题1：VLM显存占用过大

**解决方案**：
1. 使用量化模型
2. 使用梯度检查点
3. 减小batch size
4. 使用更小的VLM模型（2B版本）

### 问题2：VLM生成速度慢

**解决方案**：
1. 使用缓存机制
2. 批量处理
3. 使用更快的生成策略（greedy而非sampling）

### 问题3：医学知识库检索不准确

**解决方案**：
1. 改进向量化模型（使用医学领域模型）
2. 使用混合检索（关键词+向量）
3. 增加知识库规模

### 问题4：因果发现不稳定

**解决方案**：
1. 增加正则化
2. 使用更稳定的因果发现算法
3. 融合多个因果发现结果

---

## 📅 时间表总结

| 阶段 | 时间 | 关键任务 | 交付物 |
|------|------|---------|--------|
| 阶段1 | 1周 | VLM集成 | VLM编码器 |
| 阶段2 | 1-2周 | 知识库构建 | 知识库系统 |
| 阶段3 | 1周 | 因果对齐 | 对齐模块 |
| 阶段4 | 1-2周 | VLM引导因果发现 | 因果发现模块 |
| 阶段5 | 1周 | 报告生成 | 报告生成器 |
| 阶段6 | 2-3周 | 完整实验 | 实验结果+论文 |

**总计**: 7-10周

---

## 🎯 成功标准

### 技术指标
- ✅ AUC ≥ 0.90（vs 当前0.85-0.90）
- ✅ 准确率 ≥ 85%（vs 当前80-85%）
- ✅ 可解释性评分 ≥ 4.0/5.0

### 创新指标
- ✅ 首次将VLM与因果推理结合
- ✅ 医学知识引导的因果发现
- ✅ 可解释的诊断报告生成

### 发表指标
- 🎯 目标期刊：Nature Medicine, The Lancet Digital Health, Nature Machine Intelligence
- 🎯 论文质量：方法创新 + 临床价值 + 严格验证

---

**最后更新**: 2025-12-17

**状态**: 实施路线图完成，待开始实施

