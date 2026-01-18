#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速开始：使用增强版知识检索器
演示如何集成公开知识库资源
"""

import sys
from pathlib import Path

# 添加项目路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from knowledge_base.enhanced_knowledge_retriever import EnhancedKnowledgeRetriever


def main():
    """主函数：演示增强版检索器的使用"""
    
    print("=" * 80)
    print("增强版医学知识检索器 - 快速开始")
    print("=" * 80)
    
    # 配置知识库路径
    local_kb_path = Path(__file__).parent / "medical_guidelines.json"
    
    if not local_kb_path.exists():
        print(f"⚠️ 本地知识库不存在: {local_kb_path}")
        print("   请先运行: python build_knowledge_base.py")
        return
    
    # 创建增强版检索器（最小配置：仅使用本地知识库）
    print("\n📚 创建检索器（使用本地知识库）...")
    retriever = EnhancedKnowledgeRetriever(
        local_kb_path=str(local_kb_path),
        use_asccp=False,  # 如果已有ASCCP缓存，设为True
        use_pubmed=False,  # 需要网络，暂时关闭
        use_umls=False  # 需要API Key，暂时关闭
    )
    
    # 测试检索
    print("\n🔍 测试检索功能...")
    test_cases = [
        {'hpv': 1, 'tct': 'HSIL', 'age': 35, 'name': '高风险案例'},
        {'hpv': 0, 'tct': 'NILM', 'age': 28, 'name': '低风险案例'},
        {'hpv': 1, 'tct': 'LSIL', 'age': 45, 'name': '中等风险案例'},
    ]
    
    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"测试案例: {case['name']}")
        print(f"临床数据: HPV={case['hpv']}, TCT={case['tct']}, Age={case['age']}")
        print(f"{'='*60}")
        
        knowledge = retriever.retrieve(
            clinical_data={k: v for k, v in case.items() if k != 'name'},
            top_k=5
        )
        
        print(f"✅ 检索到 {len(knowledge)} 条知识:")
        for i, k in enumerate(knowledge, 1):
            print(f"\n  {i}. {k[:150]}...")
    
    # 展示如何启用其他知识源
    print("\n" + "=" * 80)
    print("💡 如何启用其他知识源:")
    print("=" * 80)
    print("""
    1. 启用PubMed检索（需要网络）:
       retriever = EnhancedKnowledgeRetriever(
           local_kb_path='knowledge_base/medical_guidelines.json',
           use_pubmed=True,
           pubmed_api_key=None  # 可选，有key可以提高限制
       )
    
    2. 启用UMLS检索（需要API Key）:
       retriever = EnhancedKnowledgeRetriever(
           local_kb_path='knowledge_base/medical_guidelines.json',
           use_umls=True,
           umls_api_key='your_umls_key'  # 从 https://uts.nlm.nih.gov/uts/ 获取
       )
    
    3. 使用所有知识源:
       retriever = EnhancedKnowledgeRetriever(
           local_kb_path='knowledge_base/medical_guidelines.json',
           use_asccp=True,
           use_pubmed=True,
           use_umls=True,
           pubmed_api_key='your_pubmed_key',
           umls_api_key='your_umls_key'
       )
    """)
    
    print("\n" + "=" * 80)
    print("✅ 演示完成！")
    print("=" * 80)
    print("\n📖 更多信息请查看: README_PUBLIC_KNOWLEDGE_BASES.md")


if __name__ == '__main__':
    main()

