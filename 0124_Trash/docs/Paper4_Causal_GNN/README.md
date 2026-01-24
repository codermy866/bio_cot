# 论文4：因果GNN多模态融合

## 📋 论文信息

**英文标题**：Causal Graph Neural Networks for Multimodal Medical Diagnosis: Learning Causal Representations with Intervention Capability

**中文标题**：用于多模态医学诊断的因果图神经网络：具有干预能力的因果表示学习

**目标期刊**：
- IEEE Transactions on Knowledge and Data Engineering (IF 9.2) - 首选
- Artificial Intelligence (IF 14.0) - 备选
- Nature Machine Intelligence (IF 25.9) - 如果结果非常强

**优先级**：⭐⭐⭐（方法最复杂，但创新性最强）

---

## 🎯 核心创新点

1. **CausalEncoder**：学习因果因子表示
2. **CausalGNN**：图神经网络学习因果结构
3. **干预机制**：支持do-operator干预操作
4. **反事实推理**：支持"如果...会怎样"的推理

---

## 📂 文件夹内容

### 当前文档
- `ADAPTIVE_CAUSAL_INTERVENTION_COMPLETE.md` - 自适应因果干预完整说明
- `ADAPTIVE_CAUSAL_INTERVENTION_EXECUTION_PLAN.md` - 自适应因果干预执行计划
- `ENHANCED_CAUSAL_CLIP_EXPERIMENT_PLAN.md` - 增强因果CLIP实验计划
- `ENHANCED_CAUSAL_CLIP_IMPLEMENTATION_SUMMARY.md` - 增强因果CLIP实现总结

### 后续可添加
- `causal_graph_learning/` - 因果图学习验证
- `intervention_experiments/` - 干预实验
- `counterfactual_analysis/` - 反事实推理案例
- `figures/` - 因果图、干预效果可视化
- `drafts/` - 论文草稿
- `code/` - CausalEncoder、CausalGNN实现

---

## 📝 实验补充清单

### 高优先级
- [ ] 因果图学习验证（学习到的因果图 vs 医学先验知识）
- [ ] 干预实验（单因子干预、多因子干预）
- [ ] 反事实推理（"如果HPV阴性会怎样"等案例）

### 中优先级
- [ ] 与标准GNN对比
- [ ] 与标准多模态方法对比
- [ ] 消融实验（无因果约束、无干预机制、无反事实推理）

### 低优先级
- [ ] 因果因子分析
- [ ] 因果路径分析
- [ ] 干预强度分析

---

## 📅 时间规划

- **Month 7-8**：方法完善
- **Month 9-10**：实验执行
- **Month 11-12**：撰写与投稿

---

## 🔗 相关文档

- 论文拆分分析：`../shared/PAPER_SPLITTING_ANALYSIS.md`
- 写作指南：`../shared/PAPER_WRITING_GUIDE.md`
- 实验清单：`../shared/EXPERIMENT_SUPPLEMENT_CHECKLIST.md`

