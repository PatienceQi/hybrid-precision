"""
检索器模块 - 提供各种文档检索实现
"""

from retrievers.base_retriever import BaseRetriever, RetrievalResult
from .embedding_retriever import EmbeddingRetriever
from .hybrid_retriever import HybridRetriever
from .retriever_factory import RetrieverFactory, create_retriever, create_embedding_retriever, create_hybrid_retriever

__all__ = [
    'BaseRetriever',
    'RetrievalResult',
    'EmbeddingRetriever',
    'HybridRetriever',
    'RetrieverFactory',
    'create_retriever',
    'create_embedding_retriever',
    'create_hybrid_retriever'
]