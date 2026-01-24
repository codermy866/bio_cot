# 论文3：贝叶斯CLIP不确定性量化

## 📋 论文信息

**英文标题**：Bayesian CLIP for Uncertainty-Aware Medical Diagnosis: Quantifying Prediction Confidence in Multimodal Learning

**中文标题**：用于不确定性感知医学诊断的贝叶斯CLIP：量化多模态学习中的预测置信度

**目标期刊**：
- Machine Learning (IF 7.5) - 首选
- Statistics in Medicine (IF 2.0) - 备选
- Journal of Medical Internet Research (IF 7.1) - 备选

**优先级**：⭐⭐⭐⭐（方法创新突出）

---

## 🎯 核心创新点

1. **贝叶斯编码器**：使用变分推断量化不确定性
2. **不确定性分解**：认知不确定性 + 偶然不确定性
3. **不确定性传播**：通过融合层传播不确定性
4. **临床决策支持**：高不确定性提示进一步检查

---

## 📂 文件夹内容

### 当前文档
- `CAUSAL_BAYESIAN_CLIP_PROPOSAL.md` - 因果贝叶斯CLIP提案

### 后续可添加
- `bayesian_encoder/` - 贝叶斯编码器实现
- `uncertainty_analysis/` - 不确定性分析实验
- `calibration/` - 校准分析（可靠性图）
- `clinical_cases/` - 临床决策支持案例
- `figures/` - 不确定性可视化图表
- `drafts/` - 论文草稿
- `code/` - 贝叶斯CLIP实现代码

---

## 📝 实验补充清单

### 高优先级
- [ ] 不确定性校准（可靠性图、ECE、Brier Score）
- [ ] 不确定性分解（认知不确定性 vs 偶然不确定性）
- [ ] 临床决策支持（高不确定性样本识别、决策阈值优化）

### 中优先级
- [ ] 与标准CLIP对比（性能、不确定性估计）
- [ ] 与其他不确定性方法对比（MC-Dropout、Deep Ensemble、Conformal Prediction）
- [ ] 不确定性与性能关系分析

### 低优先级
- [ ] 不确定性分布分析
- [ ] 温度缩放对比

---

## 📅 时间规划

- **Month 3-4**：方法完善与实验设计
- **Month 5-6**：实验执行
- **Month 7-8**：撰写与投稿

---

## 🔗 相关文档

- 论文拆分分析：`../shared/PAPER_SPLITTING_ANALYSIS.md`
- 写作指南：`../shared/PAPER_WRITING_GUIDE.md`
- 实验清单：`../shared/EXPERIMENT_SUPPLEMENT_CHECKLIST.md`

