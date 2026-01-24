# 虚拟环境配置说明

## 虚拟环境位置
- **路径**: `/data2/hmy/VLM_Caus_Rm_Mics/my_retfound`
- **类型**: 符号链接，指向 `/data2/hmy/RETFound_MAE-main-new-good/my_retfound`
- **Python版本**: Python 3.11.13

## 激活方式

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics
source my_retfound/bin/activate
```

## 环境信息

- **PyTorch版本**: 2.2.0+cu118
- **CUDA版本**: 11.8
- **GPU数量**: 2
- **CUDA可用**: True

## 训练启动命令

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics
source my_retfound/bin/activate
python experiments/exp2_bida/train_bio_cot.py
```

## 后台运行

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="experiments/exp2_bida/exp_bio_cot/logs/train_bio_cot_${TIMESTAMP}.log"
source my_retfound/bin/activate
nohup python experiments/exp2_bida/train_bio_cot.py > "$LOG_FILE" 2>&1 &
```

## 验证环境

```bash
source my_retfound/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from src.models.bida.bio_cot_model import BioCOTModel; print('✅ 模型导入成功')"
```

