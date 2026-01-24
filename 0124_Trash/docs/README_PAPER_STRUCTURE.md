# 论文文件夹结构说明

## 📁 文件夹组织架构

本文档说明 `docs/` 目录下的论文文件夹组织结构，方便管理和归类每篇论文的相关文档。

---

## 📂 文件夹结构

```
docs/
├── Paper1_Multimodal_Fusion/          # 论文1：多模态融合基础方法
├── Paper2_Causal_CLIP/                # 论文2：因果约束的CLIP方法
├── Paper3_Bayesian_CLIP/              # 论文3：贝叶斯CLIP不确定性量化
├── Paper4_Causal_GNN/                 # 论文4：因果GNN多模态融合
├── Paper5_MultiCenter_Validation/    # 论文5：多中心验证与临床评估
└── shared/                            # 共享文档（所有论文共用）
```

---

## 📋 各文件夹内容说明

### 📂 Paper1_Multimodal_Fusion/
**论文标题**：Multimodal Deep Learning for Cervical Lesion Diagnosis: Integrating OCT, Colposcopy, and Clinical Features

**核心内容**：
- 多模态融合框架（OCT + Colposcopy + 临床特征）
- 多种backbone对比（CNN, Swin-T, VMamba, ViT）
- 跨模态注意力融合机制
- 5个中心的外部验证

**包含文档**：
- `METHODOLOGY_DETAILS.md` - 详细方法说明
- `PROJECT_METHODOLOGY_AND_INNOVATION_ANALYSIS.md` - 方法和创新点分析
- `CURRENT_ARCHITECTURE_ANALYSIS.md` - 当前架构分析
- `COMPLETE_SUMMARY.md` - 完整技术总结
- `FINAL_COMPLETE_SUMMARY.md` - 最终完整总结
- `FINAL_SUMMARY_CHINESE.md` - 中文技术总结
- `TECHNICAL_DETAILS_2CLASS.md` - 2分类技术细节

**后续可添加**：
- 实验脚本
- 结果分析报告
- 图表文件
- 论文草稿

---

### 📂 Paper2_Causal_CLIP/
**论文标题**：Causal-Constrained CLIP for Medical Multimodal Learning: From Association to Causation

**核心内容**：
- 因果约束的CLIP方法
- 基于医学先验知识构建因果图
- 消除虚假关联
- 因果约束的注意力掩码机制

**包含文档**：
- `CLIP_CONTRIBUTION_AND_IMPLEMENTATION.md` - CLIP贡献与实现
- `CLIP_DETAILED_EXPLANATION_AND_IMPLEMENTATION.md` - CLIP详解与实现
- `CLIP_INNOVATION_SUMMARY.md` - CLIP创新总结
- `CLIP_TRAINING_RESEARCH_PLAN.md` - CLIP训练研究计划
- `CLIP_TRAINING_STATUS_CLARIFICATION.md` - CLIP训练状态说明

**后续可添加**：
- 因果图构建方法文档
- 虚假关联消除实验报告
- 因果约束注意力实现代码
- 论文草稿

---

### 📂 Paper3_Bayesian_CLIP/
**论文标题**：Bayesian CLIP for Uncertainty-Aware Medical Diagnosis: Quantifying Prediction Confidence

**核心内容**：
- 贝叶斯CLIP不确定性量化
- 变分推断方法
- 不确定性分解（认知不确定性 + 偶然不确定性）
- 临床决策支持

**包含文档**：
- `CAUSAL_BAYESIAN_CLIP_PROPOSAL.md` - 因果贝叶斯CLIP提案

**后续可添加**：
- 贝叶斯编码器实现文档
- 不确定性校准实验报告
- 可靠性图分析
- 临床决策支持案例
- 论文草稿

---

### 📂 Paper4_Causal_GNN/
**论文标题**：Causal Graph Neural Networks for Multimodal Medical Diagnosis: Learning Causal Representations with Intervention Capability

**核心内容**：
- 因果GNN多模态融合
- CausalEncoder和CausalGNN架构
- 干预机制（do-operator）
- 反事实推理

**包含文档**：
- `ADAPTIVE_CAUSAL_INTERVENTION_COMPLETE.md` - 自适应因果干预完整说明
- `ADAPTIVE_CAUSAL_INTERVENTION_EXECUTION_PLAN.md` - 自适应因果干预执行计划
- `ENHANCED_CAUSAL_CLIP_EXPERIMENT_PLAN.md` - 增强因果CLIP实验计划
- `ENHANCED_CAUSAL_CLIP_IMPLEMENTATION_SUMMARY.md` - 增强因果CLIP实现总结

**后续可添加**：
- 因果图学习验证报告
- 干预实验报告
- 反事实推理案例
- 论文草稿

---

### 📂 Paper5_MultiCenter_Validation/
**论文标题**：External Validation and Clinical Utility Assessment of a Multimodal Deep Learning System for Cervical Lesion Diagnosis: A Multi-Center Study

**核心内容**：
- 5个独立中心的外部验证
- 决策曲线分析（DCA）
- 临床实用性评估
- 亚组分析

**包含文档**：
- `EXTERNAL_VALIDATION_PLAN.md` - 外部验证计划
- `EXTERNAL_VALIDATION_QUICK_START.md` - 外部验证快速开始
- `INTERNAL_EXTERNAL_VALIDATION_GUIDE.md` - 内外验证指南
- `COMPREHENSIVE_EVALUATION_PIPELINE.md` - 综合评估管道

**后续可添加**：
- 分中心验证结果报告
- DCA分析结果
- 临床实用性评估报告
- 论文草稿

---

### 📂 shared/
**共享文档**：所有论文共用的文档

**包含文档**：
- `PAPER_SPLITTING_ANALYSIS.md` - 论文拆分完整分析
- `PAPER_SPLITTING_SUMMARY.md` - 论文拆分策略总结
- `PAPER_WRITING_GUIDE.md` - 论文写作详细指南
- `EXPERIMENT_SUPPLEMENT_CHECKLIST.md` - 实验补充清单
- `IMPLEMENTATION_TIMELINE.md` - 实施时间表
- `COMPLETE_EXPERIMENT_PLAN.md` - 完整实验计划
- `EXPERIMENT_CHECKLIST.md` - 实验检查清单
- `LANCET_PUBLICATION_EXPERIMENT_PLAN.md` - Lancet发表实验计划
- `LANCET_EXPERIMENTS_CHINESE.md` - Lancet实验中文版
- `LANCET_PUBLICATION_ASSESSMENT.md` - Lancet发表评估

**用途**：
- 论文拆分策略和规划
- 通用的写作指南
- 实验计划和检查清单
- 时间规划

---

## 📝 使用建议

### 1. 添加新文档
- 根据文档内容，将其放入对应的论文文件夹
- 如果是通用文档，放入 `shared/` 文件夹
- 在文档开头添加简要说明

### 2. 文件命名规范
- 使用英文命名，单词首字母大写
- 使用下划线连接单词
- 文件名应清晰描述文档内容
- 示例：`Method_Implementation_Details.md`

### 3. 文档分类原则
- **方法相关**：放入对应论文文件夹
- **实验相关**：放入对应论文文件夹
- **结果相关**：放入对应论文文件夹
- **通用规划**：放入 `shared/` 文件夹

### 4. 维护建议
- 定期整理和归类新文档
- 删除过时或重复的文档
- 保持文件夹结构清晰
- 更新本README文件

---

## 🔄 文件夹创建时间

创建日期：2024年11月18日

---

## 📚 相关文档

- 详细的论文拆分分析：`shared/PAPER_SPLITTING_ANALYSIS.md`
- 论文写作指南：`shared/PAPER_WRITING_GUIDE.md`
- 实验补充清单：`shared/EXPERIMENT_SUPPLEMENT_CHECKLIST.md`
- 实施时间表：`shared/IMPLEMENTATION_TIMELINE.md`

---

**提示**：如有疑问或需要调整文件夹结构，请参考 `shared/PAPER_SPLITTING_SUMMARY.md` 了解整体规划。

