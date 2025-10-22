"""
检索器基类和检索结果定义
提供统一的检索接口和结果格式
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """检索结果"""

    # 基本信息
    query: str
    documents: List[Dict[str, Any]]
    scores: List[float]

    # 元数据
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    retriever_type: str = ""
    retrieval_time: float = 0.0
    total_documents: int = 0

    # 额外信息
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if not self.total_documents and self.documents:
            self.total_documents = len(self.documents)

    def get_top_documents(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """获取前k个文档"""
        if not self.documents:
            return []

        # 按分数排序
        doc_scores = list(zip(self.documents, self.scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回前k个
        return [doc for doc, score in doc_scores[:top_k]]

    def get_document_texts(self, top_k: Optional[int] = None) -> List[str]:
        """获取文档文本内容"""
        documents = self.get_top_documents(top_k) if top_k else self.documents
        texts: List[str] = []
        for doc in documents:
            if not isinstance(doc, dict):
                texts.append(str(doc))
                continue
            text = doc.get('text')
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
                continue
            content = doc.get('content')
            if isinstance(content, str) and content.strip():
                texts.append(content.strip())
                continue
            if isinstance(content, list):
                combined = " ".join(str(item) for item in content if isinstance(item, str))
                texts.append(combined.strip())
                continue
            texts.append("")
        return texts

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'query': self.query,
            'documents': self.documents,
            'scores': self.scores,
            'timestamp': self.timestamp,
            'retriever_type': self.retriever_type,
            'retrieval_time': self.retrieval_time,
            'total_documents': self.total_documents,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalResult':
        """从字典创建检索结果"""
        return cls(**data)

class BaseRetriever(ABC):
    """检索器基类"""

    def __init__(self, retriever_type: str = "base"):
        self.retriever_type = retriever_type
        self._validate_setup()

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> RetrievalResult:
        """检索文档"""
        pass

    @abstractmethod
    def batch_retrieve(self, queries: List[str], top_k: int = 5, **kwargs) -> List[RetrievalResult]:
        """批量检索文档"""
        pass

    def _validate_setup(self):
        """验证检索器设置"""
        if not self.retriever_type:
            raise ValueError("检索器类型不能为空")

    def _validate_query(self, query: str):
        """验证查询"""
        if not query or not isinstance(query, str):
            raise ValueError("查询必须是有效的字符串")

    def _create_empty_result(self, query: str) -> RetrievalResult:
        """创建空结果"""
        return RetrievalResult(
            query=query,
            documents=[],
            scores=[],
            retriever_type=self.retriever_type,
            retrieval_time=0.0,
            total_documents=0
        )

    def _sort_by_score(self, documents: List[Dict[str, Any]], scores: List[float]) -> tuple:
        """按分数排序文档"""
        if not documents or not scores or len(documents) != len(scores):
            return documents, scores

        # 组合并排序
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # 分离
        sorted_docs, sorted_scores = zip(*doc_scores)
        return list(sorted_docs), list(sorted_scores)

    def get_supported_features(self) -> List[str]:
        """获取支持的功能列表"""
        return []

    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            'retriever_type': self.retriever_type,
            'supported_features': self.get_supported_features()
        }

class RetrievalError(Exception):
    """检索错误"""
    pass
