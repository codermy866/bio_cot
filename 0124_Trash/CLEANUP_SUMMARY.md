# 代码清理汇总报告

**清理日期**: 2025-01-24  
**清理目的**: 为Git上传准备干净的代码库，只保留最新方法（exp_bio3.2）

---

## 一、移动到垃圾文件夹的内容

以下目录/文件夹已移动到 `0124_Trash/` 文件夹：

### 1. 旧版本实验目录
- `experiments/exp_3.1` - Bio-COT 3.1版本
- `experiments/exp_5centers` - 5中心实验
- `experiments/exp_bio3.0` - Bio-COT 3.0版本
- `experiments/exp_bio3.0_improved` - Bio-COT 3.0改进版
- `experiments/exp_bio4.0` - Bio-COT 4.0版本
- `experiments/exp_bio5.0` - Bio-COT 5.0版本
- `experiments/exp_bio5.0_improved` - Bio-COT 5.0改进版
- `experiments/exp_xiangyang` - 襄阳实验
- `experiments/exp1_causal_bayesian_clip` - 因果贝叶斯CLIP实验
- `experiments/exp2_bida` - BIDA实验

### 2. 其他目录
- `cursor_file/` - Cursor相关文件
- `docs/` - 文档目录
- `analysis/` - 分析目录
- `Trash/` - 原有垃圾文件夹
- `src/` - 源代码目录
- `scripts/` - 脚本目录

---

## 二、保留的当前实验

**保留的实验目录**:
- `experiments/exp_bio3.2/` - **当前最新版本，保留用于Git上传**

**保留的其他目录**:
- `experiments/baseline/` - 基线实验
- `experiments/comparison_experiments/` - 对比实验
- `data/` - 数据目录
- `logs/` - 日志目录
- 其他项目根目录文件

---

## 三、清理统计

- **移动的目录数量**: 16个
- **占用空间**: 29GB
- **清理后experiments目录**: 仅保留 `exp_bio3.2` 及相关文件

---

## 四、Git上传建议

### 4.1 添加到.gitignore

建议在 `.gitignore` 文件中添加：

```
# 垃圾文件夹
0124_Trash/
```

### 4.2 提交前检查

在Git提交前，确认：
1. ✅ `experiments/exp_bio3.2/` 是唯一保留的实验目录
2. ✅ 所有旧版本实验已移动到 `0124_Trash/`
3. ✅ `.gitignore` 已更新，排除垃圾文件夹
4. ✅ 代码库结构清晰，只包含最新方法

---

## 五、恢复说明

如果需要恢复某个目录，可以从 `0124_Trash/` 中移回：

```bash
# 示例：恢复某个实验目录
mv 0124_Trash/experiments/exp_bio4.0 experiments/
```

---

**清理完成时间**: 2025-01-24 13:16  
**清理操作**: 所有指定目录已成功移动到 `0124_Trash/` 文件夹

