# Bio-COT 3.0 可视化数据归档

## 📁 文件说明

### 可视化图表
- `*.png`: 2D可视化图表（PNG格式）
- `*_3d_*.pdf`: 3D可视化图表（PDF格式，矢量图）
- `distribution_3d_*.pdf`: 3D分布可视化（PDF格式）

### 数据文件
- `visualization_data_*.pkl`: 完整的特征数据（pickle格式）
- `visualization_data_*.csv`: 关键数据（CSV格式，便于查看）
- `visualization_config_*.json`: 配置和统计信息

### 代码文件
- `visualization_code_*.py`: 可视化生成代码
- `generate_sci_visualizations.py`: 2D可视化脚本
- `generate_3d_visualizations.py`: 3D可视化脚本

## 🔄 使用方法

### 加载数据
```python
import pickle
import pandas as pd

# 加载完整特征数据
with open('visualization_data_*.pkl', 'rb') as f:
    features_dict = pickle.load(f)

# 加载CSV数据
df = pd.read_csv('visualization_data_*.csv')
```

### 重新生成可视化
```bash
python generate_sci_visualizations.py  # 2D可视化
python generate_3d_visualizations.py  # 3D可视化
```

## 📊 数据说明

- **样本数量**: 168个验证集样本
- **特征维度**: 
  - z_causal: [168, 768]
  - z_noise: [168, 768]
  - z_sem: [168, 768]
- **标签分布**: 阴性=113, 阳性=55
- **中心分布**: 4个中心（0, 1, 2, 3）

## 📝 注意事项

1. PDF文件是矢量图，可以无损缩放
2. PNG文件是位图，300 DPI，适合打印
3. pickle文件需要Python环境加载
4. CSV文件可以用Excel或其他工具打开

---
**归档时间**: 2026年 01月 13日 星期二 15:01:50 CST
