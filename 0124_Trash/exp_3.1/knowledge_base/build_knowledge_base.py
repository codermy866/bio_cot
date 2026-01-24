#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
医学知识库构建脚本
根据ASCCP指南等医学知识构建JSON知识库
"""

import json
from pathlib import Path
from typing import Dict, List


def build_medical_guidelines() -> Dict[str, str]:
    """
    构建医学知识库（基于ASCCP指南等）
    使用简化的Key格式，便于检索
    
    Returns:
        guidelines: {key: value} 字典，key为临床状态组合，value为医学解释文本
    """
    guidelines = {}
    
    # 1. HPV相关指南（简化Key格式）
    guidelines["HPV_High_Age_30+"] = (
        "Patients aged 30+ with High-risk HPV are at significant risk for CIN2+. "
        "Colposcopy should focus on acetowhite changes near the transformation zone. "
        "Look for dense acetowhite areas and coarse punctuation patterns."
    )
    
    guidelines["HPV_High_Age_under_30"] = (
        "Patients under 30 with High-risk HPV: Conservative management recommended. "
        "Most lesions regress spontaneously. Focus on transformation zone changes."
    )
    
    guidelines["HPV_Neg"] = (
        "HPV Negative patients: Lower risk for high-grade lesions. "
        "Visual features might be subtle; look for fine mosaicism or minor vascular changes."
    )
    
    # 2. TCT相关指南
    guidelines["HPV_Neg_TCT_ASCUS"] = (
        "HPV Negative with ASC-US usually suggests inflammation or atrophy. "
        "Visual features might be subtle; look for fine mosaicism. "
        "Low probability of high-grade lesion."
    )
    
    guidelines["TCT_HSIL"] = (
        "HSIL cytology: Immediate colposcopy mandatory. "
        "High probability of high-grade lesion. Focus on dense acetowhite areas, "
        "coarse punctuation, and mosaic patterns."
    )
    
    guidelines["TCT_LSIL"] = (
        "LSIL cytology: Moderate risk for underlying CIN2+. "
        "Look for acetowhite changes, fine punctuation, and mosaic patterns."
    )
    
    # 3. 年龄相关指南
    guidelines["Age_45+_TCT_LSIL"] = (
        "Older patients (45+) with LSIL often have lesions receding into the endocervical canal. "
        "Attention should be paid to the canal opening. "
        "Lesions may be less visible on surface examination."
    )
    
    guidelines["Age_30+"] = (
        "Patients aged 30+: Standard screening protocols apply. "
        "Co-testing (HPV + TCT) is recommended. "
        "Focus on transformation zone and endocervical canal."
    )
    
    guidelines["Age_under_30"] = (
        "Patients under 30: Conservative management. "
        "Most low-grade lesions regress spontaneously. "
        "Repeat testing in 12 months if HPV positive with normal cytology."
    )
    
    # 4. 组合状态指南
    guidelines["HPV_High_TCT_HSIL"] = (
        "High-risk combination: HPV positive with HSIL cytology. "
        "Immediate colposcopy mandatory. Risk of CIN2+ exceeds 70%. "
        "Focus on dense acetowhite areas, coarse punctuation, and vascular changes."
    )
    
    guidelines["HPV_High_TCT_LSIL"] = (
        "HPV positive with LSIL: Moderate-high risk combination. "
        "Colposcopy recommended. Risk of CIN2+ is 15-30%. "
        "Look for acetowhite changes and fine punctuation."
    )
    
    # 5. 通用指南（Fallback）
    guidelines["General"] = (
        "General colposcopy guidelines apply. "
        "Focus on transformation zone, acetowhite changes, and vascular patterns. "
        "Document all findings for proper risk assessment."
    )
    
    guidelines["HPV18_POS_Age>30"] = (
        "High-risk HPV type 18 in patients over 30 years old is associated with "
        "increased risk of cervical adenocarcinoma. Colposcopy with endocervical "
        "sampling is recommended."
    )
    
    guidelines["HPV_HR_POS_Age>30"] = (
        "High-risk HPV (non-16/18) in patients over 30 years old with normal cytology "
        "can be managed with repeat testing in 12 months. However, if cytology shows "
        "ASC-US or worse, colposcopy is indicated."
    )
    
    # 2. TCT结果相关指南
    guidelines["TCT_NILM_HPV_POS"] = (
        "Normal cytology (NILM) with HPV positive status: For patients over 30, "
        "co-testing is recommended. If HPV16/18 positive, immediate colposcopy. "
        "If other HR-HPV positive, repeat co-testing in 12 months."
    )
    
    guidelines["TCT_ASCUS_HPV_POS"] = (
        "ASC-US cytology with HPV positive: Colposcopy is recommended regardless of "
        "HPV type. This combination indicates increased risk of underlying CIN2+."
    )
    
    guidelines["TCT_LSIL_HPV_POS"] = (
        "LSIL cytology with HPV positive: Colposcopy is recommended. This combination "
        "has a 15-30% risk of underlying CIN2+. Endocervical sampling should be "
        "performed if colposcopy is satisfactory."
    )
    
    guidelines["TCT_HSIL"] = (
        "HSIL cytology: Immediate colposcopy is mandatory regardless of HPV status. "
        "This finding has a 70-75% risk of underlying CIN2+ and requires immediate "
        "evaluation and potential treatment."
    )
    
    # 3. 年龄相关指南
    guidelines["Age<21"] = (
        "Patients under 21 years old: Conservative management is recommended even "
        "with abnormal cytology or HPV positive. Most lesions in this age group "
        "regress spontaneously."
    )
    
    guidelines["Age_21-24"] = (
        "Patients aged 21-24: Management differs from older patients. ASC-US or "
        "LSIL can be followed with repeat cytology in 12 months rather than "
        "immediate colposcopy."
    )
    
    guidelines["Age>65"] = (
        "Patients over 65 years old: If adequate prior screening is documented "
        "and recent tests are normal, screening can be discontinued. However, "
        "abnormal findings still require evaluation."
    )
    
    # 4. 组合状态指南
    guidelines["HPV16_POS_TCT_LSIL"] = (
        "HPV16 positive with LSIL cytology: High-risk combination requiring "
        "immediate colposcopy. Risk of CIN2+ is approximately 50-60%. "
        "Endocervical sampling is essential."
    )
    
    guidelines["HPV18_POS_TCT_ASCUS"] = (
        "HPV18 positive with ASC-US cytology: Colposcopy is recommended due to "
        "increased risk of adenocarcinoma. Endocervical sampling should be performed."
    )
    
    guidelines["HPV_HR_POS_TCT_NILM_Age>30"] = (
        "High-risk HPV positive with normal cytology in patients over 30: "
        "Repeat co-testing in 12 months is acceptable. However, if HPV16/18 "
        "positive, immediate colposcopy is recommended."
    )
    
    # 5. 高风险组合
    guidelines["HIGH_RISK_COMBINATION"] = (
        "High-risk combination: HPV16/18 positive with any abnormal cytology, "
        "or HSIL cytology regardless of HPV status. Immediate colposcopy is "
        "mandatory. Risk of CIN2+ exceeds 50%."
    )
    
    return guidelines


def build_knowledge_base(output_path: str = "medical_guidelines.json"):
    """
    构建并保存医学知识库
    
    Args:
        output_path: 输出文件路径
    """
    guidelines = build_medical_guidelines()
    
    # 保存为JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(guidelines, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 医学知识库已构建完成！")
    print(f"   文件路径: {output_path}")
    print(f"   条目数量: {len(guidelines)}")
    print(f"   文件大小: {output_path.stat().st_size / 1024:.2f} KB")


def load_knowledge_base(knowledge_base_path: str) -> Dict[str, str]:
    """
    加载医学知识库
    
    Args:
        knowledge_base_path: 知识库文件路径
    
    Returns:
        guidelines: 知识库字典
    """
    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        guidelines = json.load(f)
    return guidelines


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='构建医学知识库')
    parser.add_argument(
        '--output',
        type=str,
        default='knowledge_base/medical_guidelines.json',
        help='输出文件路径'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("医学知识库构建工具")
    print("=" * 80)
    
    build_knowledge_base(args.output)
    
    print("=" * 80)

