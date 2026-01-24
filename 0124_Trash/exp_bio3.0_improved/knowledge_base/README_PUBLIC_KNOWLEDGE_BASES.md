# 公开医学知识库资源使用指南

## 📚 可直接使用的公开知识库

### 1. **ASCCP Guidelines（推荐）** ⭐⭐⭐⭐⭐

**来源**: 美国阴道镜和宫颈病理学会 (American Society for Colposcopy and Cervical Pathology)

**获取方式**:
- 官网: https://www.asccp.org/management-guidelines/
- 可以直接从官网下载指南PDF或文本
- 已经整理好的JSON格式指南（本项目已包含）

**优点**:
- ✅ 完全免费，公开可访问
- ✅ 与宫颈癌筛查任务直接相关
- ✅ 权威性强，临床认可度高
- ✅ 无需API Key，可直接使用

**使用方法**:
```python
from knowledge_base.enhanced_knowledge_retriever import ASCCPGuidelinesRetriever

retriever = ASCCPGuidelinesRetriever(
    local_cache_path='knowledge_base/asccp_guidelines.json'  # 可选，本地缓存
)
knowledge = retriever.retrieve({'hpv': 1, 'tct': 'HSIL', 'age': 35}, top_k=5)
```

---

### 2. **PubMed API（推荐用于文献检索）** ⭐⭐⭐⭐

**来源**: 美国国家医学图书馆 (National Library of Medicine)

**获取方式**:
- API文档: https://www.ncbi.nlm.nih.gov/books/NBK25497/
- 免费使用，无需API Key（但建议申请以提高请求限制）
- API Key申请: https://www.ncbi.nlm.nih.gov/account/settings/

**优点**:
- ✅ 完全免费
- ✅ 包含大量医学文献摘要
- ✅ 实时更新
- ✅ 无需API Key即可使用（有限制）

**限制**:
- ⚠️ 无API Key时：每秒最多3次请求
- ⚠️ 需要解析XML格式响应

**使用方法**:
```python
from knowledge_base.enhanced_knowledge_retriever import PubMedRetriever

retriever = PubMedRetriever(
    api_key=None,  # 可选，有key可以提高限制
    max_results=5
)
knowledge = retriever.retrieve({'hpv': 1, 'tct': 'HSIL', 'age': 35}, top_k=5)
```

---

### 3. **UMLS（医学术语标准化）** ⭐⭐⭐

**来源**: 美国国家医学图书馆统一医学语言系统

**获取方式**:
- 官网: https://www.nlm.nih.gov/research/umls/
- 需要注册账户: https://uts.nlm.nih.gov/uts/
- 免费用于研究用途

**优点**:
- ✅ 免费用于研究
- ✅ 包含大量医学术语和定义
- ✅ 标准化程度高
- ✅ 支持多种医学本体（SNOMED CT, MeSH等）

**限制**:
- ⚠️ 需要注册账户获取API Key
- ⚠️ 某些源（如SNOMED CT）在部分国家有使用限制

**使用方法**:
```python
from knowledge_base.enhanced_knowledge_retriever import UMLSRetriever

retriever = UMLSRetriever(
    api_key='your_umls_api_key'  # 需要从 https://uts.nlm.nih.gov/uts/ 注册获取
)
knowledge = retriever.retrieve({'hpv': 1, 'tct': 'HSIL', 'age': 35}, top_k=5)
```

---

### 4. **其他可用资源**

#### OpenClinical Library
- 网址: https://www.openclinical.net/library/
- 包含可执行的临床实践指南
- 适合做决策支持流程

#### MEDAKA（药物知识图谱）
- 论文: https://arxiv.org/abs/2509.26128
- 开源数据集
- 包含药品说明书、副作用、禁忌等信息

---

## 🚀 快速集成到Bio-COT 3.0

### 方法1: 使用增强版检索器（推荐）

修改 `models/knowledge_notes.py`，将 `KnowledgeRetriever` 替换为 `EnhancedKnowledgeRetriever`:

```python
# 原代码
from .knowledge_notes import KnowledgeRetriever

# 修改为
from knowledge_base.enhanced_knowledge_retriever import EnhancedKnowledgeRetriever

# 在 KnowledgeNotesModule.__init__ 中
self.retriever = EnhancedKnowledgeRetriever(
    local_kb_path=knowledge_base_path,
    use_asccp=True,  # 使用ASCCP指南
    use_pubmed=False,  # 可选：使用PubMed
    use_umls=False,  # 可选：使用UMLS
    pubmed_api_key=None,  # 可选
    umls_api_key=None  # 需要注册
)
```

### 方法2: 直接下载现成的知识库

#### 选项A: 使用GitHub上的整理好的知识库

搜索关键词：
- "ASCCP guidelines JSON"
- "cervical cancer screening knowledge base"
- "medical guidelines dataset"

#### 选项B: 从ASCCP官网下载并转换

1. 访问 https://www.asccp.org/management-guidelines/
2. 下载指南PDF或文本
3. 使用脚本转换为JSON格式

---

## 📥 下载和设置步骤

### 步骤1: 安装依赖

```bash
pip install requests  # 用于API调用
```

### 步骤2: 使用增强版检索器

```python
# 在 config.py 中配置
from knowledge_base.enhanced_knowledge_retriever import EnhancedKnowledgeRetriever

# 创建检索器
retriever = EnhancedKnowledgeRetriever(
    local_kb_path='knowledge_base/medical_guidelines.json',
    use_asccp=True,  # 启用ASCCP指南
    use_pubmed=False,  # 暂时关闭（需要网络）
    use_umls=False  # 暂时关闭（需要API Key）
)
```

### 步骤3: 测试检索

```python
clinical_data = {'hpv': 1, 'tct': 'HSIL', 'age': 35}
knowledge = retriever.retrieve(clinical_data, top_k=5)
print(f"检索到 {len(knowledge)} 条知识")
for i, k in enumerate(knowledge, 1):
    print(f"{i}. {k[:100]}...")
```

---

## 🔑 API Key获取指南

### PubMed API Key（可选）

1. 访问: https://www.ncbi.nlm.nih.gov/account/settings/
2. 注册NCBI账户
3. 在"API Key Management"中生成Key
4. 使用Key可以提高请求限制（每秒10次）

### UMLS API Key（可选）

1. 访问: https://uts.nlm.nih.gov/uts/
2. 注册UMLS账户
3. 同意使用协议
4. 在"Profile"中获取API Key

---

## 💡 推荐配置

### 最小配置（无需API Key）

```python
retriever = EnhancedKnowledgeRetriever(
    local_kb_path='knowledge_base/medical_guidelines.json',
    use_asccp=True,  # 使用本地ASCCP指南
    use_pubmed=False,
    use_umls=False
)
```

### 完整配置（需要API Keys）

```python
retriever = EnhancedKnowledgeRetriever(
    local_kb_path='knowledge_base/medical_guidelines.json',
    use_asccp=True,
    use_pubmed=True,
    use_umls=True,
    pubmed_api_key='your_pubmed_key',
    umls_api_key='your_umls_key'
)
```

---

## 📊 性能对比

| 知识源 | 响应速度 | 准确性 | 更新频率 | 推荐度 |
|--------|---------|--------|---------|--------|
| 本地知识库 | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 手动更新 | ⭐⭐⭐⭐⭐ |
| ASCCP指南 | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 定期更新 | ⭐⭐⭐⭐⭐ |
| PubMed | ⚡⚡ | ⭐⭐⭐⭐ | 实时 | ⭐⭐⭐⭐ |
| UMLS | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 定期更新 | ⭐⭐⭐ |

---

## ⚠️ 注意事项

1. **网络依赖**: PubMed和UMLS需要网络连接
2. **API限制**: 注意请求频率限制，避免被封禁
3. **数据质量**: 建议优先使用本地知识库和ASCCP指南
4. **版权问题**: 确保遵守各知识库的使用协议

---

## 🔗 相关链接

- ASCCP指南: https://www.asccp.org/management-guidelines/
- PubMed API: https://www.ncbi.nlm.nih.gov/books/NBK25497/
- UMLS: https://www.nlm.nih.gov/research/umls/
- OpenClinical: https://www.openclinical.net/

