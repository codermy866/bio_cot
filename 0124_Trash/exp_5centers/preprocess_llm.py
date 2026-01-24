#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 2.0: 离线LLM预处理脚本
目的：预先生成临床数据的LLM语义嵌入，避免训练时在线计算
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pickle
from typing import Dict, List, Optional

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F
except ImportError:
    print("❌ 请安装transformers: pip install transformers")
    sys.exit(1)


class PromptGenerator:
    """临床数据Prompt生成器（Chain-of-Thought风格）"""
    
    TCT_MAP = {
        0: "Normal (NILM)",
        1: "ASC-US (Atypical Squamous Cells of Undetermined Significance)",
        2: "LSIL (Low-grade Squamous Intraepithelial Lesion)",
        3: "HSIL (High-grade Squamous Intraepithelial Lesion)",
        4: "Cancer (Squamous Cell Carcinoma)"
    }
    
    @staticmethod
    def generate_clinical_prompt(age: int, hpv: int, tct: int) -> str:
        """
        生成临床数据的自然语言描述（Chain-of-Thought风格）
        
        Args:
            age: 年龄
            hpv: 0 (Negative) or 1 (Positive)
            tct: 0 (Normal) to 4 (Cancer)
        
        Returns:
            str: 医学描述Prompt
        """
        tct_desc = PromptGenerator.TCT_MAP.get(tct, "Unknown")
        hpv_desc = "Positive" if hpv == 1 else "Negative"
        
        # 2026 SOTA Prompt Template: Chain-of-Thought style
        prompt = (
            f"Patient Profile: A {age}-year-old female patient. "
            f"HPV Status: {hpv_desc}. "
            f"Cytology Result: {tct_desc}. "
            f"Clinical Context: The combination of age {age} and {tct_desc} cytology "
            f"suggests a correlated risk for cervical pathology. "
            f"Analysis: This patient presents clinical features consistent with "
            f"{'elevated' if hpv == 1 or tct >= 2 else 'low'} risk for cervical abnormalities. "
            f"Age-related considerations: Patients in this age group typically show "
            f"{'higher' if age >= 50 else 'moderate'} sensitivity to screening protocols."
        )
        return prompt


class LLMEmbeddingExtractor:
    """LLM嵌入提取器"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "cuda:0",
        use_8bit: bool = False,
        max_length: int = 512
    ):
        """
        Args:
            model_name: HuggingFace模型名称
                - 推荐医学LLM: "epfl-llm/meditron-7b" (需要较大显存)
                - 调试用: "bert-base-uncased" (轻量级)
            device: 设备
            use_8bit: 是否使用8bit量化（节省显存）
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        print(f"📥 正在加载LLM模型: {model_name}")
        print(f"   设备: {device}")
        print(f"   8bit量化: {use_8bit}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        try:
            if use_8bit and "cuda" in device:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModel.from_pretrained(
                    model_name,
                    device_map="auto",
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16
                )
            else:
                self.model = AutoModel.from_pretrained(model_name)
                self.model = self.model.to(device)
                self.model.eval()
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            print("   使用CPU模式...")
            self.device = "cpu"
            self.model = AutoModel.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
        
        # 获取模型输出维度
        with torch.no_grad():
            test_input = self.tokenizer("test", return_tensors="pt", padding=True, truncation=True)
            test_output = self.model(**test_input.to(self.device))
            if hasattr(test_output, 'last_hidden_state'):
                self.embed_dim = test_output.last_hidden_state.shape[-1]
            elif hasattr(test_output, 'pooler_output'):
                self.embed_dim = test_output.pooler_output.shape[-1]
            else:
                self.embed_dim = 768  # 默认值
                print(f"⚠️ 无法自动检测维度，使用默认值: {self.embed_dim}")
        
        print(f"✅ 模型加载完成，嵌入维度: {self.embed_dim}")
    
    def extract_embedding(self, prompt: str) -> np.ndarray:
        """
        从Prompt提取嵌入向量
        
        Args:
            prompt: 文本Prompt
        
        Returns:
            np.ndarray: 嵌入向量 [embed_dim]
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # 获取last_hidden_state并平均池化
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state  # [1, seq_len, embed_dim]
                # 平均池化（排除padding）
                attention_mask = inputs['attention_mask'].unsqueeze(-1)  # [1, seq_len, 1]
                masked_hidden = hidden_states * attention_mask
                embedding = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)  # [1, embed_dim]
                embedding = embedding.squeeze(0).cpu().numpy()
            elif hasattr(outputs, 'pooler_output'):
                embedding = outputs.pooler_output.squeeze(0).cpu().numpy()
            else:
                # Fallback: 使用第一个token
                embedding = outputs[0][:, 0, :].squeeze(0).cpu().numpy()
        
        return embedding


def load_clinical_data_from_csv(csv_paths: list) -> pd.DataFrame:
    """
    从CSV文件加载临床数据（支持多个CSV文件合并）
    
    Args:
        csv_paths: CSV文件路径列表（例如：train_labels.csv, val_labels.csv, external_test_labels.csv）
    
    Returns:
        pd.DataFrame: 包含age, hpv, tct, oct_id列的DataFrame
    """
    print(f"📊 正在加载CSV文件: {csv_paths}")
    
    dfs = []
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"⚠️ 文件不存在，跳过: {csv_path}")
            continue
        
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(csv_path, encoding='gbk')
            except Exception as e:
                print(f"⚠️ 读取CSV失败，跳过: {csv_path}, 错误: {e}")
                continue
        
        print(f"   ✅ 加载 {csv_path.name}: {len(df)} 条记录")
        dfs.append(df)
    
    if len(dfs) == 0:
        raise ValueError("❌ 没有成功加载任何CSV文件！")
    
    # 合并所有DataFrame
    df = pd.concat(dfs, ignore_index=True)
    
    # 标准化列名（不区分大小写）
    df.columns = df.columns.str.strip()
    
    # 映射列名（支持多种可能的列名）
    column_mapping = {
        'age': ['age', '年龄', 'AGE'],
        'oct_id': ['oct', 'oct_id', 'OCT', 'OCT图像Id', 'oct图像id'],
        'hpv': ['hpv', 'hpv清洗', 'HPV清洗', 'HPV清洗（高亮表示阳性）', 'hpv清洗（高亮表示阳性）'],
        'tct': ['tct', 'tct清洗', 'TCT清洗', 'TCT清洗（高亮表示阳性）', 'tct清洗（高亮表示阳性）'],
        'patient_id': ['id', 'ID', 'patient_id', 'oct_id', 'oct', 'OCT']
    }
    
    # 查找并重命名列
    for target_col, possible_names in column_mapping.items():
        for name in possible_names:
            if name in df.columns:
                if target_col not in df.columns or df[target_col].isna().all():
                    df[target_col] = df[name]
                break
    
    # 确保有oct_id列（用于匹配）
    if 'oct_id' not in df.columns:
        if 'OCT' in df.columns:
            df['oct_id'] = df['OCT']
        elif 'oct' in df.columns:
            df['oct_id'] = df['oct']
        elif 'patient_id' in df.columns:
            df['oct_id'] = df['patient_id']
        else:
            df['oct_id'] = df.index
    
    # 确保有patient_id列（用于保存嵌入）
    if 'patient_id' not in df.columns:
        df['patient_id'] = df['oct_id']
    
    print(f"✅ 合并后共 {len(df)} 条记录")
    print(f"   列名: {df.columns.tolist()}")
    print(f"   唯一oct_id数量: {df['oct_id'].nunique()}")
    
    return df


def load_clinical_data(excel_path: str = None, csv_paths: list = None) -> pd.DataFrame:
    """
    从Excel或CSV文件加载临床数据（兼容旧接口）
    
    Args:
        excel_path: Excel文件路径（可选）
        csv_paths: CSV文件路径列表（可选，优先使用）
    
    Returns:
        pd.DataFrame: 包含age, hpv, tct列的DataFrame
    """
    if csv_paths is not None and len(csv_paths) > 0:
        return load_clinical_data_from_csv(csv_paths)
    
    if excel_path is not None:
        print(f"📊 正在加载Excel文件: {excel_path}")
        try:
            df = pd.read_excel(excel_path)
        except Exception as e:
            print(f"❌ 读取Excel失败: {e}")
            raise
        
        # 标准化列名（不区分大小写）
        df.columns = df.columns.str.lower()
        
        # 映射中文列名到英文列名
        column_mapping = {
            '年龄': 'age',
            'oct图像id': 'oct_id',
            'hpv清洗（高亮表示阳性）': 'hpv',
            'tct清洗（高亮表示阳性）': 'tct',
        }
        
        for chinese_col, english_col in column_mapping.items():
            if chinese_col in df.columns and english_col not in df.columns:
                df[english_col] = df[chinese_col]
        
        # 确保有patient_id或oct_id列用于标识
        if 'patient_id' not in df.columns:
            if 'oct_id' in df.columns:
                df['patient_id'] = df['oct_id']
            else:
                df['patient_id'] = df.index
        
        print(f"✅ 加载了 {len(df)} 条记录")
        print(f"   列名: {df.columns.tolist()}")
        
        return df
    
    raise ValueError("❌ 必须提供excel_path或csv_paths参数！")


def preprocess_all_samples(
    excel_path: str = None,
    csv_paths: list = None,
    output_path: str = None,
    model_name: str = "bert-base-uncased",
    device: str = "cuda:0",
    use_8bit: bool = False,
    batch_size: int = 32
):
    """
    预处理所有样本，生成LLM嵌入
    
    Args:
        excel_path: Excel文件路径（可选，如果提供csv_paths则忽略）
        csv_paths: CSV文件路径列表（优先使用，例如：train_labels.csv, val_labels.csv, external_test_labels.csv）
        output_path: 输出文件路径（.pkl或.npy）
        model_name: LLM模型名称
        device: 设备
        use_8bit: 是否使用8bit量化
        batch_size: 批处理大小（用于加速）
    """
    # 加载数据（优先使用CSV文件）
    df = load_clinical_data(excel_path=excel_path, csv_paths=csv_paths)
    
    # 初始化提取器
    extractor = LLMEmbeddingExtractor(
        model_name=model_name,
        device=device,
        use_8bit=use_8bit
    )
    
    # 生成Prompt并提取嵌入
    embeddings_dict = {}
    prompt_generator = PromptGenerator()
    
    print(f"🔄 开始提取嵌入向量...")
    print(f"   总样本数: {len(df)}")
    print(f"   批处理大小: {batch_size}")
    
    # 获取patient_id列（支持中文列名）
    if 'patient_id' in df.columns:
        id_col = 'patient_id'
    elif 'oct_id' in df.columns:
        id_col = 'oct_id'
    elif 'oct图像id' in df.columns:
        id_col = 'oct图像id'
        df['oct_id'] = df['oct图像id']  # 创建别名
        id_col = 'oct_id'
    else:
        id_col = None  # 使用索引
    
    failed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取嵌入"):
        try:
            # 获取ID（优先使用oct_id/OCT列，因为这是训练时使用的标识符）
            if 'oct_id' in row.index and pd.notna(row['oct_id']):
                patient_id = str(row['oct_id']).strip()
            elif 'OCT' in row.index and pd.notna(row['OCT']):
                patient_id = str(row['OCT']).strip()
            elif 'oct' in row.index and pd.notna(row['oct']):
                patient_id = str(row['oct']).strip()
            elif 'patient_id' in row.index and pd.notna(row['patient_id']):
                patient_id = str(row['patient_id']).strip()
            elif id_col is not None and id_col in row.index:
                patient_id = str(row[id_col]).strip()
            else:
                patient_id = str(idx)
            
            # 获取临床数据（支持多种列名格式）
            age_val = None
            for col in ['age', 'AGE', '年龄']:
                if col in row.index:
                    age_val = row[col]
                    break
            age = int(age_val) if pd.notna(age_val) and isinstance(age_val, (int, float, np.integer, np.floating)) else 50
            
            hpv_val = None
            for col in ['hpv', 'HPV清洗', 'hpv清洗', 'HPV清洗（高亮表示阳性）', 'hpv清洗（高亮表示阳性）']:
                if col in row.index:
                    hpv_val = row[col]
                    break
            
            tct_val = None
            for col in ['tct', 'TCT清洗', 'tct清洗', 'TCT清洗（高亮表示阳性）', 'tct清洗（高亮表示阳性）']:
                if col in row.index:
                    tct_val = row[col]
                    break
            
            # 标准化HPV值
            if pd.isna(hpv_val):
                hpv = 0
            elif isinstance(hpv_val, (int, float)):
                hpv = 1 if float(hpv_val) > 0 else 0
            else:
                hpv_str = str(hpv_val).lower()
                hpv = 1 if any(k in hpv_str for k in ['16', '18', 'positive', '阳性', '高危', '1']) else 0
            
            # 标准化TCT值
            if pd.isna(tct_val):
                tct = 0
            elif isinstance(tct_val, (int, float)):
                tct = int(tct_val)
            else:
                tct_str = str(tct_val).upper()
                if 'ASC-US' in tct_str or 'ASCUS' in tct_str:
                    tct = 1
                elif 'LSIL' in tct_str:
                    tct = 2
                elif 'HSIL' in tct_str:
                    tct = 3
                elif 'SCC' in tct_str or '癌' in tct_str:
                    tct = 4
                else:
                    tct = 0
            
            # 生成Prompt
            prompt = prompt_generator.generate_clinical_prompt(age, hpv, tct)
            
            # 提取嵌入
            embedding = extractor.extract_embedding(prompt)
            
            # 保存
            embeddings_dict[patient_id] = embedding
            
        except Exception as e:
            failed_count += 1
            print(f"⚠️ 样本 {idx} 处理失败: {e}")
            continue
    
    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.pkl':
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings_dict, f)
    elif output_path.suffix == '.npy':
        # 保存为numpy格式（需要额外保存ID映射）
        np.save(output_path, embeddings_dict)
    else:
        # 默认使用pkl
        output_path = output_path.with_suffix('.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings_dict, f)
    
    print(f"\n✅ 预处理完成！")
    print(f"   成功: {len(embeddings_dict)} 个样本")
    print(f"   失败: {failed_count} 个样本")
    print(f"   输出文件: {output_path}")
    print(f"   嵌入维度: {extractor.embed_dim}")
    print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Bio-COT 2.0: 离线LLM预处理')
    parser.add_argument('--excel_path', type=str, default=None,
                       help='Excel文件路径（可选，如果提供csv_paths则忽略）')
    parser.add_argument('--csv_paths', type=str, nargs='+', default=None,
                       help='CSV文件路径列表（优先使用，例如：train_labels.csv val_labels.csv external_test_labels.csv）')
    parser.add_argument('--output_path', type=str,
                       default='/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_5centers/data/clinical_embeddings.pkl',
                       help='输出文件路径')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='LLM模型名称（推荐: bert-base-uncased 或 epfl-llm/meditron-7b）')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备 (cuda:0, cuda:1, cpu)')
    parser.add_argument('--use_8bit', action='store_true',
                       help='使用8bit量化（节省显存）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批处理大小')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Bio-COT 2.0: 离线LLM预处理")
    print("=" * 80)
    if args.csv_paths:
        print(f"CSV路径: {args.csv_paths}")
    elif args.excel_path:
        print(f"Excel路径: {args.excel_path}")
    else:
        print("❌ 错误：必须提供--excel_path或--csv_paths参数！")
        parser.print_help()
        sys.exit(1)
    print(f"输出路径: {args.output_path}")
    print(f"模型: {args.model_name}")
    print(f"设备: {args.device}")
    print(f"8bit量化: {args.use_8bit}")
    print("=" * 80)
    
    preprocess_all_samples(
        excel_path=args.excel_path,
        csv_paths=args.csv_paths,
        output_path=args.output_path,
        model_name=args.model_name,
        device=args.device,
        use_8bit=args.use_8bit,
        batch_size=args.batch_size
    )

