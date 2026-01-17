# 因果图可视化结果输出位置说明

## 📍 可视化结果输出位置总览

### 1. 主要输出目录

#### 1.1 `adaptive_causal_intervention_results/` 目录
**位置**: `/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/adaptive_causal_intervention_results/`

**包含的可视化文件**:
```
✅ innovation2_adaptive_causal_graph.png      - 自适应因果图流程图
✅ innovation2_adaptive_causal_graph.pdf     - PDF版本
✅ innovation2_detailed_causal_graph.png       - 详细因果图
✅ innovation2_detailed_causal_graph.pdf      - PDF版本
✅ innovation3_causal_intervention.png        - 因果干预流程图
✅ innovation3_causal_intervention.pdf        - PDF版本
✅ clip_architecture.png                      - CLIP架构图
✅ clip_architecture.pdf                      - PDF版本
✅ model_architecture.png                     - 模型架构图
✅ model_architecture.pdf                     - PDF版本
```

**生成方式**: 
- 由 `visualization/draw_innovation_details.py` 生成
- 由 `visualization/draw_clip_architecture.py` 生成
- 由 `visualization/draw_model_architecture.py` 生成

#### 1.2 `causal_analysis/` 目录（训练时生成）
**位置**: `/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/causal_analysis/`

#### 1.3 `causal_analysis_detailed/` 目录（详细医学因果图）
**位置**: `/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/causal_analysis_detailed/`

**包含的详细可视化文件**:
```
✅ detailed_causal_graph_epoch_*.png          - 详细因果图网络（30个epoch）
✅ detailed_causal_adjacency_epoch_*.png       - 详细因果邻接矩阵（30个epoch）
✅ causal_paths_analysis_epoch_*.png          - 因果路径分析（每5个epoch）
✅ detailed_causal_adj_epoch_*.npy            - 因果邻接矩阵数值（30个epoch）
✅ detailed_causal_graph_final.png            - 最终详细因果图
✅ detailed_causal_adjacency_final.png        - 最终因果邻接矩阵
✅ causal_paths_analysis_final.png            - 最终因果路径分析
✅ node_descriptions.md                        - 节点说明文档
```

**生成方式**: 
- 由 `generate_detailed_medical_causal_graph.py` 生成
- 包含8个节点的详细医学因果图（HPV、Age、TCT、Clinical、OCT、Colposcopy、Multimodal、Diagnosis）

**预期输出文件**（训练时自动生成）:
```
📊 causal_graph_epoch_{epoch}.png           - 因果邻接矩阵热力图（每个epoch）
📊 causal_graph_network_epoch_{epoch}.png   - 因果图网络图（每个epoch）
📊 causal_adj_epoch_{epoch}.npy              - 因果邻接矩阵数值（每个epoch）
📊 causal_effects_epoch_{epoch}.png        - 因果效应分析图（每个epoch）
📊 causal_effects_epoch_{epoch}.npy        - 因果效应数值（每个epoch）
```

**生成方式**: 
- 由 `main_causal_finetune.py` 在训练过程中自动生成
- 需要运行训练脚本才会创建此目录

**代码位置**: `main_causal_finetune.py` 第465-471行
```python
# 可视化因果图（热力图 + 带权重网络图）
causal_adj = outputs['clinical_output']['causal_adj']
visualize_causal_graph(causal_adj, f'./causal_analysis/causal_graph_epoch_{epoch}.png')
visualize_causal_graph_networkx(causal_adj, f'./causal_analysis/causal_graph_network_epoch_{epoch}.png', title=f"Causal Graph (Epoch {epoch})")
# 保存数值化邻接矩阵
avg_adj_np = causal_adj.mean(dim=0).detach().cpu().numpy()
np.save(f'./causal_analysis/causal_adj_epoch_{epoch}.npy', avg_adj_np)
```

#### 1.3 项目根目录（测试可视化）
**位置**: `/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/`

**预期输出文件**（运行测试脚本时生成）:
```
📊 causal_adjacency_matrix.png              - 因果邻接矩阵（测试）
📊 causal_graph_network.png                 - 因果图网络（测试）
📊 project_structure.png                    - 项目结构图
📊 experiment_workflow.png                   - 实验流程图
```

**生成方式**: 
- 运行 `utils/visualize_causal_graph.py` 脚本
- 命令: `python utils/visualize_causal_graph.py`

---

## 2. 可视化函数说明

### 2.1 `visualize_causal_adjacency_matrix()`
**位置**: `utils/visualize_causal_graph.py` 第22-53行

**功能**: 生成因果邻接矩阵热力图

**输出**:
- 热力图（seaborn heatmap）
- 保存路径: 由 `save_path` 参数指定
- 默认: 当前目录或 `./causal_analysis/`

**示例调用**:
```python
from utils.visualize_causal_graph import visualize_causal_adjacency_matrix

# 假设有因果邻接矩阵
causal_adj = model.get_causal_adjacency()  # [B, N, N] 或 [N, N]

# 可视化并保存
visualize_causal_adjacency_matrix(
    causal_adj,
    save_path='./causal_analysis/causal_adjacency_matrix.png',
    title="Causal Adjacency Matrix"
)
```

### 2.2 `visualize_causal_graph_networkx()`
**位置**: `utils/visualize_causal_graph.py` 第55-121行

**功能**: 使用NetworkX生成因果图网络图

**输出**:
- 有向图（节点、边、权重）
- 保存路径: 由 `save_path` 参数指定
- 默认: 当前目录或 `./causal_analysis/`

**示例调用**:
```python
from utils.visualize_causal_graph import visualize_causal_graph_networkx

# 可视化因果图网络
visualize_causal_graph_networkx(
    causal_adj,
    save_path='./causal_analysis/causal_graph_network.png',
    title="Causal Graph Network"
)
```

### 2.3 `analyze_causal_effects()`
**位置**: `main_causal_finetune.py` 第98-134行

**功能**: 分析因果效应（直接、间接、总效应）

**输出**:
- 3面板热力图（直接效应、间接效应、总效应）
- 保存路径: `./causal_analysis/causal_effects_epoch_{epoch}.png`
- 数值结果: `./causal_analysis/causal_effects_epoch_{epoch}.npy`

**示例调用**:
```python
analyze_causal_effects(
    model, 
    outputs, 
    targets, 
    epoch=10,
    save_dir='./causal_analysis'
)
```

---

## 3. 如何生成可视化结果

### 方法1: 运行训练脚本（自动生成）
```bash
# 运行因果微调训练
python main_causal_finetune.py \
    --data_path 5centers_multi \
    --epochs 30 \
    --save_causal_analysis

# 可视化结果会自动保存到:
# ./causal_analysis/causal_graph_epoch_{epoch}.png
# ./causal_analysis/causal_graph_network_epoch_{epoch}.png
```

### 方法2: 运行可视化测试脚本
```bash
# 运行可视化测试
python utils/visualize_causal_graph.py

# 输出文件:
# - causal_adjacency_matrix.png (当前目录)
# - causal_graph_network.png (当前目录)
# - project_structure.png (当前目录)
# - experiment_workflow.png (当前目录)
```

### 方法3: 从已训练模型生成
```python
import torch
from utils.visualize_causal_graph import visualize_causal_adjacency_matrix, visualize_causal_graph_networkx

# 加载模型
model = load_model('path/to/model.pth')
model.eval()

# 获取因果邻接矩阵
with torch.no_grad():
    # 假设有测试数据
    outputs = model(oct_images, col_images, clinical_features)
    causal_adj = outputs.get('causal_adj', None)
    
    if causal_adj is not None:
        # 可视化
        visualize_causal_adjacency_matrix(
            causal_adj,
            save_path='./causal_analysis/final_causal_adjacency.png'
        )
        visualize_causal_graph_networkx(
            causal_adj,
            save_path='./causal_analysis/final_causal_graph.png'
        )
```

### 方法4: 生成详细医学因果图
```bash
# 生成详细的8节点医学因果图
python generate_detailed_medical_causal_graph.py --num_epochs 30

# 输出目录: ./causal_analysis_detailed/
# 包含: 8个节点的详细因果图、因果路径分析、节点说明文档
```

### 方法5: 从增强因果CLIP结果生成
```bash
# 检查增强因果CLIP结果
ls -la enhanced_causal_clip_results/

# 如果有模型检查点，可以加载并可视化
python -c "
import torch
from src.models.enhanced_causal_clip import EnhancedCausalBayesianCLIP
from utils.visualize_causal_graph import visualize_causal_adjacency_matrix

# 加载模型
checkpoint = torch.load('enhanced_causal_clip_results/best_model.pth')
model = EnhancedCausalBayesianCLIP(...)
model.load_state_dict(checkpoint['model_state_dict'])

# 获取因果图（如果模型支持）
# ...
"
```

---

## 4. 可视化文件格式说明

### 4.1 PNG格式（图像）
- **分辨率**: 300 DPI（适合论文）
- **尺寸**: 
  - 热力图: 12×10 英寸
  - 网络图: 15×12 英寸
- **颜色方案**: 
  - 热力图: RdBu_r (红-蓝)
  - 网络图: 绿色（正向），红色（负向）

### 4.2 PDF格式（矢量图）
- **优势**: 可缩放，适合论文
- **生成**: 部分脚本会同时生成PDF版本

### 4.3 NPY格式（数值数据）
- **内容**: 因果邻接矩阵的数值
- **用途**: 后续分析和处理
- **加载**: `np.load('causal_adj_epoch_10.npy')`

---

## 5. 当前已有的可视化文件

### 5.1 已存在的文件
根据搜索结果，以下文件已存在：

```
✅ adaptive_causal_intervention_results/
   ├── innovation2_adaptive_causal_graph.png
   ├── innovation2_adaptive_causal_graph.pdf
   ├── innovation2_detailed_causal_graph.png
   ├── innovation2_detailed_causal_graph.pdf
   ├── innovation3_causal_intervention.png
   ├── innovation3_causal_intervention.pdf
   ├── clip_architecture.png
   ├── clip_architecture.pdf
   ├── model_architecture.png
   └── model_architecture.pdf
```

### 5.2 需要运行训练才能生成的文件
以下文件需要在训练过程中生成：

```
❌ causal_analysis/
   ├── causal_graph_epoch_1.png
   ├── causal_graph_epoch_2.png
   ├── ...
   ├── causal_graph_network_epoch_1.png
   ├── causal_graph_network_epoch_2.png
   ├── ...
   ├── causal_adj_epoch_1.npy
   ├── causal_adj_epoch_2.npy
   ├── ...
   ├── causal_effects_epoch_1.png
   ├── causal_effects_epoch_2.png
   └── ...
```

---

## 6. 快速生成可视化结果

### 脚本1: 从现有模型生成可视化
创建 `generate_causal_visualizations.py`:

```python
#!/usr/bin/env python3
"""
从已训练模型生成因果图可视化
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.visualize_causal_graph import (
    visualize_causal_adjacency_matrix,
    visualize_causal_graph_networkx
)

def generate_from_model(model_path, output_dir='./causal_analysis'):
    """从模型生成可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 如果有保存的因果邻接矩阵
    if 'causal_adj' in checkpoint:
        causal_adj = checkpoint['causal_adj']
        
        # 可视化
        visualize_causal_adjacency_matrix(
            causal_adj,
            save_path=f'{output_dir}/causal_adjacency_matrix.png'
        )
        
        visualize_causal_graph_networkx(
            causal_adj,
            save_path=f'{output_dir}/causal_graph_network.png'
        )
        
        print(f"✅ 可视化已保存到: {output_dir}")
    else:
        print("⚠️  模型检查点中没有保存的因果邻接矩阵")

if __name__ == '__main__':
    # 检查增强因果CLIP结果
    model_path = 'enhanced_causal_clip_results/best_model.pth'
    if os.path.exists(model_path):
        generate_from_model(model_path)
    else:
        print(f"❌ 未找到模型: {model_path}")
```

### 脚本2: 运行测试可视化
```bash
# 直接运行可视化测试
cd /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713
python utils/visualize_causal_graph.py

# 输出会在当前目录生成:
# - causal_adjacency_matrix.png
# - causal_graph_network.png
# - project_structure.png
# - experiment_workflow.png
```

---

## 7. 可视化结果解读

### 7.1 因果邻接矩阵热力图
- **横轴/纵轴**: 因果因子索引（0=OCT, 1=Colposcopy, 2=Clinical）
- **颜色**: 
  - 蓝色（正值）: 正向因果关系
  - 红色（负值）: 负向因果关系
  - 白色（0）: 无因果关系
- **数值**: 因果强度 [0, 1]

### 7.2 因果图网络图
- **节点**: 模态（OCT, Colposcopy, Clinical）
- **边**: 因果关系
- **边颜色**: 绿色（正向），红色（负向）
- **边宽度**: 与因果强度成正比
- **边标签**: 显示因果强度数值

---

## 8. 总结

### 当前状态
- ✅ **已有可视化**: `adaptive_causal_intervention_results/` 目录中的创新点流程图
- ❌ **训练时可视化**: `causal_analysis/` 目录（需要运行训练才能生成）

### 建议操作
1. **查看已有可视化**: 
   ```bash
   ls -la adaptive_causal_intervention_results/*.png
   ```

2. **生成测试可视化**:
   ```bash
   python utils/visualize_causal_graph.py
   ```

3. **运行训练生成完整可视化**:
   ```bash
   python main_causal_finetune.py --epochs 30
   ```

4. **从已训练模型生成**:
   ```bash
   python generate_causal_visualizations.py
   ```

---

**文档版本**: v1.0  
**最后更新**: 2024年  
**维护者**: 项目团队

