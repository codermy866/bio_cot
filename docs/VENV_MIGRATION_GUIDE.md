# 虚拟环境迁移指南

## 📋 迁移状态

### 源虚拟环境
- **路径**: `/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/my_retfound`
- **大小**: 检查中...

### 目标虚拟环境
- **路径**: `/data2/hmy/VLM_Caus_Rm/my_retfound`
- **状态**: 复制进行中...

---

## 🔄 复制进度

### 查看复制进度

```bash
# 查看实时日志
tail -f /data2/hmy/VLM_Caus_Rm/venv_copy.log

# 检查复制状态
bash /data2/hmy/VLM_Caus_Rm/check_venv_copy_status.sh

# 查看当前大小
du -sh /data2/hmy/VLM_Caus_Rm/my_retfound
```

---

## 🔧 复制完成后的操作

### 1. 修复路径引用

虚拟环境复制后，需要更新激活脚本中的路径：

```bash
# 运行路径修复脚本
bash /data2/hmy/VLM_Caus_Rm/fix_venv_paths.sh
```

### 2. 测试虚拟环境

```bash
# 激活虚拟环境
source /data2/hmy/VLM_Caus_Rm/my_retfound/bin/activate

# 检查Python版本
python --version

# 检查Python路径
which python

# 检查已安装的包
pip list

# 测试导入关键包
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

### 3. 如果路径修复失败

如果虚拟环境无法正常工作，可以：

**选项1: 重新创建虚拟环境（推荐）**

```bash
cd /data2/hmy/VLM_Caus_Rm
python3 -m venv my_retfound
source my_retfound/bin/activate
pip install -r exp1_Causal_Bayesian_clip/requirements_enhanced.txt
```

**选项2: 使用原虚拟环境（通过软链接）**

```bash
cd /data2/hmy/VLM_Caus_Rm
ln -s /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/my_retfound my_retfound
```

---

## 📝 使用新虚拟环境

### 在训练脚本中使用

```bash
# 激活虚拟环境
source /data2/hmy/VLM_Caus_Rm/my_retfound/bin/activate

# 运行训练
cd /data2/hmy/VLM_Caus_Rm/exp1_Causal_Bayesian_clip/code
python train_vlm_causal_clip.py --batch_size 10
```

### 在shell脚本中使用

```bash
#!/bin/bash
cd /data2/hmy/VLM_Caus_Rm
source my_retfound/bin/activate

# 运行训练
python exp1_Causal_Bayesian_clip/code/train_vlm_causal_clip.py \
    --batch_size 10 \
    --use_amp
```

---

## ⚠️ 注意事项

1. **复制时间**: 虚拟环境可能很大，复制需要一些时间
2. **路径问题**: 复制后可能需要修复路径引用
3. **测试**: 复制完成后务必测试虚拟环境是否正常工作
4. **备份**: 原虚拟环境保持不变，可以随时回退

---

## 🔍 故障排查

### 问题1: 虚拟环境激活失败

**症状**: `source my_retfound/bin/activate` 后路径不正确

**解决**:
```bash
bash /data2/hmy/VLM_Caus_Rm/fix_venv_paths.sh
```

### 问题2: Python命令找不到

**症状**: `python --version` 报错

**解决**: 重新创建虚拟环境或使用原虚拟环境

### 问题3: 包导入失败

**症状**: `import torch` 等失败

**解决**: 
1. 检查虚拟环境是否正确激活
2. 重新安装依赖: `pip install -r requirements_enhanced.txt`

---

## ✅ 验证清单

- [ ] 虚拟环境复制完成
- [ ] 路径修复完成
- [ ] 虚拟环境可以激活
- [ ] Python版本正确
- [ ] 关键包可以导入
- [ ] 训练脚本可以运行

---

**最后更新**: 2025-12-17

