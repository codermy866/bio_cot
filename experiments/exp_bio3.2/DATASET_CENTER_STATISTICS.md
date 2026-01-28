# 五中心多模态宫颈病变数据集：按中心统计表（SCI论文用）

## 📊 数据集概述

本研究使用了一个来自**五个医疗中心**的多模态宫颈病变数据集，采用**Leave-Centers-Out**策略进行数据划分：
- **内部开发集**（Internal Development Set）：用于模型训练和验证，包含3个中心
- **外部独立测试集**（External Test Set）：用于最终泛化能力评估，包含2个中心（严格不参与训练）

---

## 📋 表1：详细中心分布（按数据子集）

| Subset | Center | Code | Total (n) | Positive (n₊) | Negative (n₋) | Positive Rate (%) |
|:-------|:-------|:-----|----------:|---------------:|---------------:|------------------:|
| **Train (internal)** | Enshi（恩施） | M22105 | 320 | 102 | 218 | 31.9% |
| **Train (internal)** | Wuda（武大） | M20105 | 73 | 73 | 0 | 100.0% |
| **Train (internal)** | Xiangyang（襄阳） | M22102 | 276 | 43 | 233 | 15.6% |
| **Val (internal)** | Enshi（恩施） | M22105 | 84 | 29 | 55 | 34.5% |
| **Val (internal)** | Wuda（武大） | M20203 | 16 | 16 | 0 | 100.0% |
| **Val (internal)** | Xiangyang（襄阳） | M22102 | 68 | 10 | 58 | 14.7% |
| **External Test** | Jingzhou（荆州） | M22101 | 79 | 28 | 51 | 35.4% |
| **External Test** | Shiyan（十堰） | M22104 | 69 | 21 | 48 | 30.4% |

**说明**：
- **Train (internal)**：内部训练集，共669例
- **Val (internal)**：内部验证集，共168例
- **External Test**：外部独立测试集，共148例

---

## 📋 表2：中心汇总统计表（按中心汇总所有子集）

| Center | Code | Train (n) | Val (n) | External Test (n) | **Total (n)** | **Positive (n₊)** | **Negative (n₋)** | **Positive Rate (%)** |
|:-------|:-----|----------:|--------:|-------------------:|-------------:|------------------:|------------------:|---------------------:|
| **Enshi（恩施）** | M22105 | 320 | 84 | 0 | **404** | **131** | **273** | **32.4%** |
| **Jingzhou（荆州）** | M22101 | 0 | 0 | 79 | **79** | **28** | **51** | **35.4%** |
| **Shiyan（十堰）** | M22104 | 0 | 0 | 69 | **69** | **21** | **48** | **30.4%** |
| **Wuda（武大）** | M20105/M20203 | 73 | 16 | 0 | **89** | **89** | **0** | **100.0%** |
| **Xiangyang（襄阳）** | M22102 | 276 | 68 | 0 | **344** | **53** | **291** | **15.4%** |

**说明**：
- **内部开发集中心**（参与训练/验证）：Enshi（恩施）、Wuda（武大）、Xiangyang（襄阳）
- **外部测试集中心**（严格不参与训练）：Jingzhou（荆州）、Shiyan（十堰）
- **Wuda（武大）中心**：该中心在内部开发集中全部为阳性样本（100%阳性率），这反映了该中心可能专门收治高风险患者的特点

---

## 📋 表3：数据子集总体统计（补充表）

| Subset | Total (n) | Positive (n₊) | Negative (n₋) | Positive Rate (%) | Centers |
|:-------|----------:|---------------:|---------------:|------------------:|:--------|
| **Train (internal)** | 669 | 218 | 451 | 32.6% | Enshi, Wuda, Xiangyang |
| **Val (internal)** | 168 | 55 | 113 | 32.7% | Enshi, Wuda, Xiangyang |
| **External Test** | 148 | 49 | 99 | 33.1% | Jingzhou, Shiyan |
| **Internal (Train+Val)** | 837 | 273 | 564 | 32.6% | Enshi, Wuda, Xiangyang |
| **All Subjects** | 985 | 322 | 663 | 32.7% | All 5 centers |

---

## 🔬 数据特点分析

### 1. 中心分布特点
- **内部开发集**：3个中心，共837例（Train: 669, Val: 168）
- **外部测试集**：2个中心，共148例
- **总体**：5个中心，共985例

### 2. 阳性率分布
- **总体阳性率**：32.7%（322/985）
- **各中心阳性率范围**：15.4% - 100.0%
  - 最低：Xiangyang（襄阳）15.4%
  - 最高：Wuda（武大）100.0%（该中心可能为专科高风险中心）
  - 其他中心：30.4% - 35.4%（相对均衡）

### 3. Leave-Centers-Out设计
- **严格分离**：外部测试集的两个中心（Jingzhou、Shiyan）完全未参与训练
- **泛化评估**：确保模型在未见过的中心上的真实泛化能力

---

## 📝 LaTeX表格代码（可直接用于论文）

### 表1：详细分布表

```latex
\begin{table}[t]
\centering
\caption{Detailed center-wise distribution across data subsets.}
\label{tab:center_distribution_detailed}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllcccc}
\toprule
Subset & Center & Code & Total $(n)$ & Positive $(n_{+})$ & Negative $(n_{-})$ & Positive Rate (\%)\\
\midrule
Train (internal) & Enshi（恩施） & M22105 & 320 & 102 & 218 & 31.9\% \\
Train (internal) & Wuda（武大） & M20105 & 73 & 73 & 0 & 100.0\% \\
Train (internal) & Xiangyang（襄阳） & M22102 & 276 & 43 & 233 & 15.6\% \\
Val (internal) & Enshi（恩施） & M22105 & 84 & 29 & 55 & 34.5\% \\
Val (internal) & Wuda（武大） & M20203 & 16 & 16 & 0 & 100.0\% \\
Val (internal) & Xiangyang（襄阳） & M22102 & 68 & 10 & 58 & 14.7\% \\
External Test & Jingzhou（荆州） & M22101 & 79 & 28 & 51 & 35.4\% \\
External Test & Shiyan（十堰） & M22104 & 69 & 21 & 48 & 30.4\% \\
\bottomrule
\end{tabular}%
}
\end{table}
```

### 表2：中心汇总表（推荐用于主文）

```latex
\begin{table}[t]
\centering
\caption{Center-wise summary statistics of the five-center multimodal cervical lesion dataset.}
\label{tab:center_summary}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccl}
\toprule
Center & Code & Train $(n)$ & Val $(n)$ & External Test $(n)$ & Total $(n)$ & Positive $(n_{+})$ & Negative $(n_{-})$ & Positive Rate (\%)\\
\midrule
Enshi（恩施） & M22105 & 320 & 84 & 0 & 404 & 131 & 273 & 32.4\% \\
Jingzhou（荆州） & M22101 & 0 & 0 & 79 & 79 & 28 & 51 & 35.4\% \\
Shiyan（十堰） & M22104 & 0 & 0 & 69 & 69 & 21 & 48 & 30.4\% \\
Wuda（武大） & M20105/M20203 & 73 & 16 & 0 & 89 & 89 & 0 & 100.0\% \\
Xiangyang（襄阳） & M22102 & 276 & 68 & 0 & 344 & 53 & 291 & 15.4\% \\
\bottomrule
\end{tabular}%
}
\end{table}
```

### 表3：数据子集总体统计（补充）

```latex
\begin{table}[t]
\centering
\caption{Overall dataset statistics by subset.}
\label{tab:dataset_overview}
\begin{tabular}{lccccc}
\toprule
Subset & Total $(n)$ & Positive $(n_{+})$ & Negative $(n_{-})$ & Positive Rate (\%) & Centers\\
\midrule
Train (internal) & 669 & 218 & 451 & 32.6\% & Enshi, Wuda, Xiangyang\\
Val (internal) & 168 & 55 & 113 & 32.7\% & Enshi, Wuda, Xiangyang\\
External Test & 148 & 49 & 99 & 33.1\% & Jingzhou, Shiyan\\
\midrule
Internal (Train+Val) & 837 & 273 & 564 & 32.6\% & Enshi, Wuda, Xiangyang\\
All Subjects & 985 & 322 & 663 & 32.7\% & All 5 centers\\
\bottomrule
\end{tabular}
\end{table}
```

---

## 📌 论文写作建议

### 在Methods/Data部分可以这样描述：

> **Dataset and Study Population**  
> We utilized a five-center multimodal cervical lesion dataset comprising 985 subjects from five medical centers in China. The dataset was split using a **Leave-Centers-Out** strategy to ensure rigorous evaluation of model generalizability:
> - **Internal development set** (837 subjects from 3 centers: Enshi, Wuda, and Xiangyang) was used for model training and validation
> - **External test set** (148 subjects from 2 centers: Jingzhou and Shiyan) was strictly held out and used only for final evaluation
> 
> The overall positive rate was 32.7% (322/985), with center-specific positive rates ranging from 15.4% to 100.0% (Table X). Each subject included three modalities: OCT structural images, colposcopy images, and structured clinical information (HPV, TCT, age).

### 在Results部分可以引用：

> The model achieved an AUC of X.XX on the internal validation set and X.XX on the external test set, demonstrating robust generalization across different medical centers (Table X).

---

## 📁 相关文件

- `center_distribution_detailed.csv` - 详细分布表（CSV格式）
- `center_summary.csv` - 中心汇总表（CSV格式）
- `center_statistics_latex.tex` - LaTeX表格代码

---

**生成时间**：2026-01-23  
**数据来源**：`/data2/hmy/VLM_Caus_Rm_Mics/data/5centers_multi_leave_centers_out`

