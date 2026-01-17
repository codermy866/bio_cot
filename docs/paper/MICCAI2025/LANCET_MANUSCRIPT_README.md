# The Lancet Primary Care Manuscript - Usage Guide

## 📄 文档说明

本文档 (`LANCET_PRIMARY_CARE_MANUSCRIPT.md`) 是按照 **The Lancet Primary Care** 期刊格式撰写的完整论文草稿。

## 📋 文档结构

### 主要章节

1. **Title** - 标题
2. **Abstract** - 摘要（Background, Methods, Findings, Interpretation）
3. **Introduction** - 引言（背景、研究目标、创新点）
4. **Methods** - 方法（详细完整）
   - 研究设计与参与者
   - 数据收集
   - 模型架构
   - 训练策略
   - 评估指标
   - 统计分析
5. **Results** - 结果
   - 参与者特征
   - 模型性能
   - 因果图分析
   - 不确定性量化
   - 消融研究
   - 多中心验证
6. **Discussion** - 讨论
   - 主要发现
   - 临床意义
   - 局限性
   - 与现有方法对比
   - 未来方向
7. **Conclusion** - 结论
8. **Contributors** - 贡献者
9. **Declaration of Interests** - 利益声明
10. **Data Sharing** - 数据共享
11. **Acknowledgments** - 致谢
12. **References** - 参考文献
13. **Figures and Tables** - 图表
14. **Supplementary Material** - 补充材料
15. **Appendix** - 附录

## ✅ 已完成部分

- ✅ 完整的论文结构框架
- ✅ 详细的Methods部分（模型架构、训练策略、评估指标）
- ✅ 数据集描述（5个中心，985个患者）
- ✅ 模型架构详细说明
- ✅ 损失函数和训练配置
- ✅ 初步结果（训练中）

## ⚠️ 待完善部分

### 1. 结果部分（Results）

**需要更新**：
- [ ] 内部验证集最终性能指标（AUC, Accuracy, Sensitivity, Specificity, F1-Score）
- [ ] 外部测试集最终性能指标
- [ ] 与基线方法的对比表格
- [ ] 学习到的因果图权重和结构
- [ ] 不确定性分布统计
- [ ] 消融实验结果
- [ ] 各中心性能对比

**数据来源**：
- 训练日志：`enhanced_causal_clip_results/run_bs24_e100_noamp_fullOCT_noclc_causal001_ls001_kl0001/train.log`
- 性能指标：`enhanced_causal_clip_results/run_bs24_e100_noamp_fullOCT_noclc_causal001_ls001_kl0001/metrics.json`
- 最佳模型：`enhanced_causal_clip_results/run_bs24_e100_noamp_fullOCT_noclc_causal001_ls001_kl0001/best_model.pth`

### 2. 作者信息

**需要填写**：
- [ ] 作者姓名和单位
- [ ] 通讯作者信息
- [ ] 作者贡献声明

### 3. 伦理和法规

**需要确认**：
- [ ] 伦理审查委员会批准信息
- [ ] 知情同意书信息
- [ ] 数据收集日期范围

### 4. 参考文献

**需要补充**：
- [ ] 完整的参考文献列表（按照The Lancet格式）
- [ ] 关键文献引用（CLIP, NOTEARS, Bayesian Deep Learning, 医学AI等）

### 5. 图表

**需要生成/更新**：
- [ ] Figure 1: 模型架构图（PDF格式）
- [ ] Figure 2: 数据集分布图（已有PDF）
- [ ] Figure 3: 学习到的因果图（已有部分）
- [ ] Figure 4: 性能对比图（ROC曲线、混淆矩阵）
- [ ] Figure 5: 不确定性分析图
- [ ] Table 1-4: 各种统计表格

### 6. 补充材料

**需要准备**：
- [ ] 详细的超参数表格
- [ ] 各中心详细性能
- [ ] 统计检验结果
- [ ] 额外的因果图可视化

## 🔧 使用建议

### 1. 训练完成后更新结果

```bash
# 查看训练完成后的最佳性能
cat enhanced_causal_clip_results/run_bs24_e100_noamp_fullOCT_noclc_causal001_ls001_kl0001/metrics.json

# 提取关键指标并更新到Results部分
```

### 2. 生成性能对比图

```python
# 使用现有的可视化脚本生成ROC曲线、混淆矩阵等
# 保存为PDF格式用于论文
```

### 3. 完善参考文献

参考The Lancet的参考文献格式：
- 期刊文章：Author A, Author B. Title. Journal. Year;Volume:Pages.
- 会议文章：Author A, Author B. Title. Conference. Year.

### 4. 检查字数限制

The Lancet Primary Care通常要求：
- 摘要：≤250 words
- 正文：根据文章类型（Original Article通常≤3000 words）
- 参考文献：根据期刊要求

## 📝 写作注意事项

### The Lancet格式要求

1. **摘要**：
   - 结构化摘要（Background, Methods, Findings, Interpretation）
   - 不超过250词
   - 包含关键数字和统计结果

2. **方法**：
   - 详细描述，确保可重现
   - 统计方法明确
   - 伦理声明完整

3. **结果**：
   - 客观描述，不解释
   - 包含置信区间和P值
   - 图表清晰

4. **讨论**：
   - 解释结果的意义
   - 讨论局限性
   - 与现有研究对比

5. **图表**：
   - 高质量（300 DPI或矢量图）
   - PDF格式优先
   - 清晰的图例和标题

## 📊 当前状态

- **训练状态**: 进行中（100 epochs，当前约6 epochs）
- **结果状态**: 初步结果可用，最终结果待训练完成
- **文档状态**: 框架完整，待填充最终数据

## 🎯 下一步行动

1. **等待训练完成**（100 epochs）
2. **提取最终性能指标**
3. **生成所有必要的图表**（PDF格式）
4. **完善参考文献**
5. **填写作者信息**
6. **最终审校和格式检查**

## 📞 联系信息

如有问题或需要帮助，请联系：
- [通讯作者邮箱]

---

**最后更新**: 2025-12-17

**版本**: v1.0

