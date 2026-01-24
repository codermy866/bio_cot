#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版医学知识检索器
支持多种公开知识库资源：
1. ASCCP Guidelines (公开指南)
2. UMLS (需要注册，但免费)
3. PubMed API (公开文献)
4. 本地知识库 (JSON格式)
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
import time
from urllib.parse import quote


class ASCCPGuidelinesRetriever:
    """
    ASCCP指南检索器
    从ASCCP官网或本地缓存的指南文本中检索
    """
    
    def __init__(self, local_cache_path: Optional[str] = None):
        """
        Args:
            local_cache_path: 本地缓存的ASCCP指南JSON文件路径
        """
        self.local_cache_path = local_cache_path
        self.guidelines = self._load_guidelines()
    
    def _load_guidelines(self) -> Dict[str, str]:
        """加载ASCCP指南"""
        if self.local_cache_path and Path(self.local_cache_path).exists():
            with open(self.local_cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 返回空字典，需要先下载指南
            return {}
    
    def retrieve(self, clinical_data: Dict, top_k: int = 5) -> List[str]:
        """
        检索ASCCP指南
        
        Args:
            clinical_data: {'hpv': int, 'tct': str, 'age': int}
            top_k: 返回top-k条指南
        
        Returns:
            guidelines: 指南文本列表
        """
        # 使用与原有检索器相同的逻辑
        hpv = clinical_data.get('hpv', 0)
        tct = clinical_data.get('tct', 'NILM')
        age = clinical_data.get('age', 50)
        
        query_keys = []
        
        # 构建查询键（与原有逻辑一致）
        if hpv == 1:
            query_keys.append("HPV_HR_POS_Age>30" if age > 30 else "HPV_High_Age_under_30")
        
        if isinstance(tct, str):
            tct_upper = tct.upper()
            if 'HSIL' in tct_upper:
                query_keys.append("TCT_HSIL")
            elif 'LSIL' in tct_upper:
                query_keys.append("TCT_LSIL_HPV_POS" if hpv == 1 else "TCT_LSIL")
            elif 'ASCUS' in tct_upper:
                query_keys.append("TCT_ASCUS_HPV_POS" if hpv == 1 else "TCT_ASCUS")
        
        if age < 21:
            query_keys.append("Age<21")
        elif age > 65:
            query_keys.append("Age>65")
        
        # 检索指南
        retrieved = []
        for key in query_keys:
            if key in self.guidelines:
                retrieved.append(self.guidelines[key])
                if len(retrieved) >= top_k:
                    break
        
        return retrieved


class PubMedRetriever:
    """
    PubMed文献检索器
    使用PubMed API检索相关医学文献摘要
    """
    
    def __init__(self, api_key: Optional[str] = None, max_results: int = 5):
        """
        Args:
            api_key: PubMed API Key (可选，有key可以提高请求限制)
            max_results: 最大返回结果数
        """
        self.api_key = api_key
        self.max_results = max_results
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def retrieve(self, clinical_data: Dict, top_k: int = 5) -> List[str]:
        """
        从PubMed检索相关文献摘要
        
        Args:
            clinical_data: {'hpv': int, 'tct': str, 'age': int}
            top_k: 返回top-k条摘要
        
        Returns:
            abstracts: 文献摘要列表
        """
        # 构建查询词
        query_terms = []
        
        hpv = clinical_data.get('hpv', 0)
        tct = clinical_data.get('tct', 'NILM')
        age = clinical_data.get('age', 50)
        
        if hpv == 1:
            query_terms.append("HPV positive")
        else:
            query_terms.append("HPV negative")
        
        if isinstance(tct, str):
            tct_upper = tct.upper()
            if 'HSIL' in tct_upper:
                query_terms.append("HSIL")
            elif 'LSIL' in tct_upper:
                query_terms.append("LSIL")
            elif 'ASCUS' in tct_upper:
                query_terms.append("ASCUS")
        
        query_terms.append("cervical cancer screening")
        query_terms.append("colposcopy")
        
        query = " AND ".join(query_terms)
        
        try:
            # 搜索PubMed
            search_url = f"{self.base_url}/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': top_k,
                'retmode': 'json'
            }
            if self.api_key:
                params['api_key'] = self.api_key
            
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            pmids = data.get('esearchresult', {}).get('idlist', [])
            
            if not pmids:
                return []
            
            # 获取摘要
            fetch_url = f"{self.base_url}/efetch.fcgi"
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml'
            }
            if self.api_key:
                fetch_params['api_key'] = self.api_key
            
            fetch_response = requests.get(fetch_url, params=fetch_params, timeout=10)
            fetch_response.raise_for_status()
            
            # 解析XML获取摘要（简化版，实际需要XML解析）
            # 这里返回一个占位符，实际使用时需要解析XML
            abstracts = [f"PubMed文献摘要 {i+1}: 关于{query}的研究发现..." 
                        for i in range(min(len(pmids), top_k))]
            
            # 添加延迟避免请求过快
            time.sleep(0.34)  # PubMed要求每秒最多3次请求
            
            return abstracts
            
        except Exception as e:
            print(f"⚠️ PubMed检索失败: {e}")
            return []


class UMLSRetriever:
    """
    UMLS术语检索器
    需要先注册UMLS账户获取API Key
    文档: https://documentation.uts.nlm.nih.gov/rest/home.html
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: UMLS API Key (需要从 https://uts.nlm.nih.gov/uts/ 注册获取)
        """
        self.api_key = api_key
        self.base_url = "https://uts-ws.nlm.nih.gov/rest"
    
    def retrieve(self, clinical_data: Dict, top_k: int = 5) -> List[str]:
        """
        从UMLS检索相关医学概念和定义
        
        Args:
            clinical_data: {'hpv': int, 'tct': str, 'age': int}
            top_k: 返回top-k条概念
        
        Returns:
            concepts: 医学概念定义列表
        """
        if not self.api_key:
            print("⚠️ UMLS API Key未设置，跳过UMLS检索")
            return []
        
        # 构建查询词
        query_terms = []
        hpv = clinical_data.get('hpv', 0)
        tct = clinical_data.get('tct', 'NILM')
        
        if hpv == 1:
            query_terms.append("human papillomavirus")
        
        if isinstance(tct, str):
            tct_upper = tct.upper()
            if 'HSIL' in tct_upper:
                query_terms.append("high-grade squamous intraepithelial lesion")
            elif 'LSIL' in tct_upper:
                query_terms.append("low-grade squamous intraepithelial lesion")
        
        concepts = []
        for term in query_terms[:top_k]:
            try:
                # 搜索UMLS概念
                search_url = f"{self.base_url}/search/current"
                params = {
                    'string': term,
                    'apiKey': self.api_key,
                    'pageSize': 1
                }
                
                response = requests.get(search_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                results = data.get('result', {}).get('results', [])
                if results:
                    concept = results[0]
                    name = concept.get('name', term)
                    definition = concept.get('definition', '')
                    concepts.append(f"{name}: {definition}")
                
                time.sleep(0.1)  # 避免请求过快
                
            except Exception as e:
                print(f"⚠️ UMLS检索失败 ({term}): {e}")
                continue
        
        return concepts


class EnhancedKnowledgeRetriever:
    """
    增强版知识检索器
    整合多种知识源：ASCCP指南、PubMed文献、UMLS术语、本地知识库
    """
    
    def __init__(
        self,
        local_kb_path: str,
        use_asccp: bool = True,
        use_pubmed: bool = False,
        use_umls: bool = False,
        pubmed_api_key: Optional[str] = None,
        umls_api_key: Optional[str] = None,
        asccp_cache_path: Optional[str] = None
    ):
        """
        Args:
            local_kb_path: 本地知识库JSON文件路径
            use_asccp: 是否使用ASCCP指南
            use_pubmed: 是否使用PubMed检索
            use_umls: 是否使用UMLS检索
            pubmed_api_key: PubMed API Key (可选)
            umls_api_key: UMLS API Key (需要注册)
            asccp_cache_path: ASCCP指南缓存路径
        """
        # 本地知识库（必需）
        self.local_retriever = self._load_local_kb(local_kb_path)
        
        # ASCCP指南检索器
        self.asccp_retriever = None
        if use_asccp:
            self.asccp_retriever = ASCCPGuidelinesRetriever(asccp_cache_path)
        
        # PubMed检索器
        self.pubmed_retriever = None
        if use_pubmed:
            self.pubmed_retriever = PubMedRetriever(api_key=pubmed_api_key)
        
        # UMLS检索器
        self.umls_retriever = None
        if use_umls:
            self.umls_retriever = UMLSRetriever(api_key=umls_api_key)
    
    def _load_local_kb(self, path: str) -> Dict[str, str]:
        """加载本地知识库"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def retrieve(
        self,
        clinical_data: Dict,
        top_k: int = 5,
        source_weights: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        从多个知识源检索相关知识
        
        Args:
            clinical_data: {'hpv': int, 'tct': str, 'age': int}
            top_k: 返回top-k条知识
            source_weights: 各知识源的权重 {'local': 0.5, 'asccp': 0.3, 'pubmed': 0.2}
        
        Returns:
            knowledge: 知识文本列表（去重后）
        """
        if source_weights is None:
            source_weights = {'local': 0.5, 'asccp': 0.3, 'pubmed': 0.15, 'umls': 0.05}
        
        all_knowledge = []
        
        # 1. 本地知识库（优先级最高）
        if 'local' in source_weights and source_weights['local'] > 0:
            local_k = self._retrieve_from_local(clinical_data, 
                                               int(top_k * source_weights['local']))
            all_knowledge.extend([(k, 'local') for k in local_k])
        
        # 2. ASCCP指南
        if self.asccp_retriever and 'asccp' in source_weights and source_weights['asccp'] > 0:
            asccp_k = self.asccp_retriever.retrieve(clinical_data,
                                                   int(top_k * source_weights['asccp']))
            all_knowledge.extend([(k, 'asccp') for k in asccp_k])
        
        # 3. PubMed文献
        if self.pubmed_retriever and 'pubmed' in source_weights and source_weights['pubmed'] > 0:
            pubmed_k = self.pubmed_retriever.retrieve(clinical_data,
                                                     int(top_k * source_weights['pubmed']))
            all_knowledge.extend([(k, 'pubmed') for k in pubmed_k])
        
        # 4. UMLS术语
        if self.umls_retriever and 'umls' in source_weights and source_weights['umls'] > 0:
            umls_k = self.umls_retriever.retrieve(clinical_data,
                                                 int(top_k * source_weights['umls']))
            all_knowledge.extend([(k, 'umls') for k in umls_k])
        
        # 去重并返回top-k
        seen = set()
        unique_knowledge = []
        for k, source in all_knowledge:
            if k not in seen:
                seen.add(k)
                unique_knowledge.append(k)
                if len(unique_knowledge) >= top_k:
                    break
        
        return unique_knowledge
    
    def _retrieve_from_local(self, clinical_data: Dict, top_k: int) -> List[str]:
        """从本地知识库检索（使用原有逻辑）"""
        hpv = clinical_data.get('hpv', 0)
        tct = clinical_data.get('tct', 'NILM')
        age = clinical_data.get('age', 50)
        
        query_keys = []
        
        if hpv == 1:
            query_keys.append("HPV_HR_POS_Age>30" if age > 30 else "HPV_High_Age_under_30")
        
        if isinstance(tct, str):
            tct_upper = tct.upper()
            if 'HSIL' in tct_upper:
                query_keys.append("TCT_HSIL")
            elif 'LSIL' in tct_upper:
                query_keys.append("TCT_LSIL_HPV_POS" if hpv == 1 else "TCT_LSIL")
        
        retrieved = []
        for key in query_keys:
            if key in self.local_retriever:
                retrieved.append(self.local_retriever[key])
                if len(retrieved) >= top_k:
                    break
        
        return retrieved


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 示例1: 仅使用本地知识库（默认）
    retriever = EnhancedKnowledgeRetriever(
        local_kb_path='knowledge_base/medical_guidelines.json',
        use_asccp=False,
        use_pubmed=False,
        use_umls=False
    )
    
    clinical_data = {'hpv': 1, 'tct': 'HSIL', 'age': 35}
    knowledge = retriever.retrieve(clinical_data, top_k=5)
    print("检索到的知识:")
    for i, k in enumerate(knowledge, 1):
        print(f"{i}. {k[:100]}...")
    
    # 示例2: 使用本地 + PubMed（需要网络）
    # retriever2 = EnhancedKnowledgeRetriever(
    #     local_kb_path='knowledge_base/medical_guidelines.json',
    #     use_asccp=False,
    #     use_pubmed=True,
    #     use_umls=False,
    #     pubmed_api_key=None  # 可选
    # )
    # knowledge2 = retriever2.retrieve(clinical_data, top_k=5)
    
    # 示例3: 使用所有知识源（需要API Keys）
    # retriever3 = EnhancedKnowledgeRetriever(
    #     local_kb_path='knowledge_base/medical_guidelines.json',
    #     use_asccp=True,
    #     use_pubmed=True,
    #     use_umls=True,
    #     pubmed_api_key='your_pubmed_key',  # 可选
    #     umls_api_key='your_umls_key'  # 需要注册
    # )

