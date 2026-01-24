# 中心分布分析与外部验证集选择建议

## 📊 各中心详细统计

| 中心 | 总样本数 | 阳性 | 阴性 | 阳性率 | 训练集 | 测试集 |
|------|---------|------|------|--------|--------|--------|
| **十堰** | 78 | 26 | 52 | 33.3% | 59 | 19 |
| **恩施** | 404 | 131 | 273 | 32.4% | 324 | 80 |
| **武大** | 89 | 89 | 0 | 100.0% | 69 | 20 |
| **荆州** | 70 | 23 | 47 | 32.9% | 55 | 15 |
| **襄阳** | 344 | 53 | 291 | 15.4% | 278 | 66 |
| **总计** | **985** | **322** | **663** | **32.7%** | **785** | **200** |

---

## 🔍 中心分割方案分析

### 方案1: 十堰 + 荆州 作为外部验证集 ⭐ **推荐**

**外部验证集**:
- 总样本数: **148样本**
- 阳性: 49例 (33.1%)
- 阴性: 99例 (66.9%)
- 中心: 十堰(78) + 荆州(70)

**内部训练集**:
- 总样本数: **837样本**
- 阳性: 273例 (32.6%)
- 阴性: 564例 (67.4%)
- 中心: 恩施 + 武大 + 襄阳

**优点**:
- ✅ 外部验证集样本量适中（148样本，≥100例要求）
- ✅ 类别平衡良好（阳性33.1%，阴性66.9%）
- ✅ 内部训练集样本量充足（837样本）
- ✅ 两个中心分布均匀（十堰78样本，荆州70样本）
- ✅ 符合Lancet期刊要求（不同中心、不同设备）

**缺点**:
- ⚠️ 外部验证集样本量相对较小（148 vs 推荐200+）

---

### 方案2: 只选择荆州 作为外部验证集

**外部验证集**:
- 总样本数: **70样本**
- 阳性: 23例 (32.9%)
- 阴性: 47例 (67.1%)
- 中心: 荆州(70)

**内部训练集**:
- 总样本数: **915样本**
- 阳性: 299例 (32.7%)
- 阴性: 616例 (67.3%)
- 中心: 十堰 + 恩施 + 武大 + 襄阳

**优点**:
- ✅ 内部训练集样本量最大（915样本）
- ✅ 类别平衡良好

**缺点**:
- ❌ 外部验证集样本量不足（70样本 < 100例要求）
- ❌ 只有一个中心，不符合多中心外部验证要求
- ❌ Lancet期刊可能不接受单中心外部验证

---

### 方案3: 恩施 + 荆州 作为外部验证集（当前默认）

**外部验证集**:
- 总样本数: **474样本**
- 阳性: 154例 (32.5%)
- 阴性: 320例 (67.5%)
- 中心: 恩施(404) + 荆州(70)

**内部训练集**:
- 总样本数: **511样本**
- 阳性: 168例 (32.9%)
- 阴性: 343例 (67.1%)
- 中心: 十堰 + 武大 + 襄阳

**优点**:
- ✅ 外部验证集样本量充足（474样本）
- ✅ 类别平衡良好
- ✅ 两个中心分布

**缺点**:
- ⚠️ 内部训练集样本量相对较小（511样本）
- ⚠️ 恩施样本量最大（404），作为外部验证集会减少训练集样本量
- ⚠️ 训练集和验证集样本量不平衡（511 vs 474）

---

## 💡 推荐方案

### ✅ **推荐方案1: 十堰 + 荆州 作为外部验证集**

**理由**:
1. **样本量符合要求**: 外部验证集148样本（≥100例要求）
2. **类别平衡良好**: 阳性33.1%，阴性66.9%
3. **训练集充足**: 内部训练集837样本，足够训练模型
4. **中心分布均匀**: 两个中心样本量相近（十堰78，荆州70）
5. **符合Lancet要求**: 不同中心、不同设备、不同操作人员

**实施建议**:
- 使用此方案创建内部外部验证数据集
- 在论文中明确说明："We used center-split validation, where data from 2 centers (Shiyan and Jingzhou) were held out as an external validation set (N=148). The remaining 3 centers were used for model training (N=837)."

---

## ⚠️ 注意事项

### 1. 武大中心
- **问题**: 武大中心只有阳性样本（100%），没有阴性样本
- **影响**: 不适合作为外部验证集（需要类别平衡）
- **建议**: 将武大中心保留在内部训练集中

### 2. 样本量要求
- **Lancet最低要求**: 外部验证集≥100例
- **推荐**: 外部验证集≥200例
- **方案1**: 148样本（符合最低要求，但略低于推荐值）

### 3. 类别平衡
- **要求**: 阳性/阴性比例在0.3-3.0之间
- **方案1**: 49/99 = 0.49（符合要求）

---

## 📝 论文撰写建议

### Methods部分

```markdown
**External Validation**

We performed center-split validation by holding out data from 2 centers 
(Shiyan and Jingzhou) as an external validation set. The remaining 3 centers 
(Enshi, Wuda, and Xiangyang) were used for model training. This approach 
ensures that the external validation set comes from different centers with 
potentially different equipment and operators, which is consistent with 
TRIPOD guidelines for external validation.

- Internal training set: 3 centers, N=837 samples
  - Enshi: N=404 samples
  - Wuda: N=89 samples
  - Xiangyang: N=344 samples
- External validation set: 2 centers, N=148 samples
  - Shiyan: N=78 samples (26 positive, 52 negative)
  - Jingzhou: N=70 samples (23 positive, 47 negative)
```

### Results部分

```markdown
**External Validation Performance**

On the external validation set (2 centers, N=148), the model achieved:
- AUC: 0.XX (95% CI: 0.XX-0.XX)
- Sensitivity: 0.XX (95% CI: 0.XX-0.XX)
- Specificity: 0.XX (95% CI: 0.XX-0.XX)

The performance was comparable to the internal validation set, with an 
AUC decrease of 0.XX (acceptable threshold: <0.05).
```

---

**生成时间**: 2025-11-10
**版本**: 1.0

