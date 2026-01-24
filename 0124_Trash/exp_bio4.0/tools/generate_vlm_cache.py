#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 4.0: VLM缓存生成工具
离线生成VLM描述，避免训练时实时运行VLM（太慢且显存不够）
"""

import os
import json
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse

# 假设使用 Qwen2-VL，你需要先 pip install git+https://github.com/huggingface/transformers
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    VLM_AVAILABLE = True
except ImportError:
    print("⚠️ transformers库未安装，将使用模拟模式生成VLM缓存")
    VLM_AVAILABLE = False


def generate_vlm_cache(data_root, output_json, use_vlm=True):
    """
    生成VLM缓存（离线生成图像描述）
    
    Args:
        data_root: 数据集根目录（包含图像）
        output_json: 输出JSON文件路径（存储VLM描述）
        use_vlm: 是否使用真实的VLM模型（如果False，生成模拟数据）
    """
    print("🚀 Bio-COT 4.0: VLM缓存生成工具")
    print("=" * 80)
    
    # 专门设计的 Prompt (核心：提取语义不变量)
    PROMPT = "Describe the cervical lesion in this image efficiently. Focus on acetowhite changes, margins, and vessels. Do NOT describe the background or lighting."
    
    results = {}
    
    if use_vlm and VLM_AVAILABLE:
        print("📥 正在加载Frozen VLM Teacher (Qwen2-VL)...")
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct", 
                torch_dtype=torch.bfloat16, 
                device_map="cuda"
            ).eval()  # ❄️ 绝对冻结
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            print("✅ VLM模型加载成功")
        except Exception as e:
            print(f"⚠️ VLM模型加载失败: {e}")
            print("   切换到模拟模式...")
            use_vlm = False
    
    # 收集所有图像文件
    image_files = []
    data_path = Path(data_root)
    
    # 支持多种目录结构
    search_paths = [
        data_path,
        data_path / "internal_train",
        data_path / "internal_val",
        data_path / "images",
    ]
    
    for search_path in search_paths:
        if search_path.exists():
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(search_path.rglob(f"*{ext}"))
                image_files.extend(search_path.rglob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"⚠️ 未找到图像文件，尝试直接遍历 {data_path}")
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(Path(root) / file)
    
    if not image_files:
        raise ValueError(f"❌ 在 {data_root} 中未找到任何图像文件！")
    
    print(f"📊 找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像
    print("\n🔄 开始处理图像...")
    for img_path in tqdm(image_files, desc="生成VLM缓存"):
        try:
            # 使用文件名作为key（不含路径）
            file_key = img_path.name
            
            # 如果已经处理过，跳过
            if file_key in results:
                continue
            
            image = Image.open(img_path).convert('RGB')
            
            if use_vlm and VLM_AVAILABLE:
                # 使用真实的VLM模型
                inputs = processor(text=[f"<image>{PROMPT}"], images=image, return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=100)
                    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # 清理输出文本（移除提示词）
                if PROMPT in output_text:
                    output_text = output_text.replace(PROMPT, "").strip()
                
                results[file_key] = output_text
            else:
                # 模拟模式：生成占位描述
                # 可以根据文件名或其他信息生成合理的描述
                results[file_key] = f"Medical image showing cervical tissue with potential lesions. Acetowhite changes may be present. {file_key}"
            
        except Exception as e:
            print(f"⚠️ 处理 {img_path.name} 时出错: {e}")
            # 即使出错也记录一个默认描述
            results[img_path.name] = f"Medical image with potential cervical lesion. Error during processing: {str(e)[:50]}"
    
    # 保存结果
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ VLM缓存已保存到: {output_json}")
    print(f"   共处理 {len(results)} 个图像")
    
    # 显示示例
    if results:
        sample_key = list(results.keys())[0]
        print(f"\n📝 示例输出:")
        print(f"   文件: {sample_key}")
        print(f"   描述: {results[sample_key][:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bio-COT 4.0: 生成VLM缓存")
    parser.add_argument("--data_root", type=str, required=True,
                        help="数据集根目录路径")
    parser.add_argument("--output_json", type=str, 
                        default="data/vlm_profiles_v1.json",
                        help="输出JSON文件路径")
    parser.add_argument("--use_vlm", action="store_true", default=False,
                        help="是否使用真实的VLM模型（需要GPU和transformers库）")
    parser.add_argument("--mock", action="store_true", default=False,
                        help="使用模拟模式（不加载VLM模型）")
    
    args = parser.parse_args()
    
    # 如果指定了--mock，强制不使用VLM
    use_vlm = args.use_vlm and not args.mock
    
    generate_vlm_cache(
        data_root=args.data_root,
        output_json=args.output_json,
        use_vlm=use_vlm
    )

