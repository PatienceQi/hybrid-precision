"""
检索器工厂类
提供统一的检索器创建接口
"""

import logging
from typing import Dict, Type, Optional

from .base_retriever import BaseRetriever
from .embedding_retriever import EmbeddingRetriever
from .hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

class RetrieverFactory:
    """检索器工厂类"""

    _retrievers = {
        'embedding': EmbeddingRetriever,
        'hybrid': HybridRetriever
    }

    @classmethod
    def create_retriever(cls, retriever_type: str = 'auto', **kwargs) -> BaseRetriever:
        """
        创建检索器

        Args:
            retriever_type: 检索器类型 ('embedding', 'hybrid', 'auto')
            **kwargs: 传递给检索器的参数

        Returns:
            检索器实例

        Raises:
            ValueError: 如果检索器类型不支持
        """
        if retriever_type == 'auto':
            # 自动选择检索器类型
            retriever_type = cls._auto_select_retriever(**kwargs)

        if retriever_type not in cls._retrievers:
            raise ValueError(f"不支持的检索器类型: {retriever_type}")

        retriever_class = cls._retrievers[retriever_type]
        logger.info(f"创建检索器: {retriever_type}")

        try:
            retriever = retriever_class(**kwargs)
            logger.info(f"✅ 检索器创建成功: {retriever_type}")
            return retriever
        except Exception as e:
            logger.error(f"创建检索器失败: {e}")
            raise

    @classmethod
    def _auto_select_retriever(cls, **kwargs) -> str:
        """
        自动选择检索器类型

        Returns:
            推荐的检索器类型
        """
        # 检查是否有混合检索需求
        if kwargs.get('use_hybrid_search') or kwargs.get('fusion_method'):
            return 'hybrid'

        # 检查是否有嵌入向量
        if kwargs.get('embeddings') or kwargs.get('embedding_model'):
            return 'embedding'

        # 默认使用混合检索
        return 'hybrid'

    @classmethod
    def register_retriever(cls, name: str, retriever_class: Type[BaseRetriever]):
        """
        注册新的检索器类型

        Args:
            name: 检索器名称
            retriever_class: 检索器类（必须继承自BaseRetriever）

        Raises:
            ValueError: 如果检索器类无效
        """
        if not issubclass(retriever_class, BaseRetriever):
            raise ValueError("检索器类必须继承自BaseRetriever")

        cls._retrievers[name] = retriever_class
        logger.info(f"注册新的检索器类型: {name}")

    @classmethod
    def get_supported_retrievers(cls) -> list:
        """
        获取支持的检索器类型列表

        Returns:
            检索器类型列表
        """
        return list(cls._retrievers.keys())

    @classmethod
    def get_retriever_info(cls, retriever_type: str) -> Dict[str, str]:
        """
        获取检索器信息

        Args:
            retriever_type: 检索器类型

        Returns:
            检索器信息字典
        """
        if retriever_type not in cls._retrievers:
            raise ValueError(f"不支持的检索器类型: {retriever_type}")

        retriever_class = cls._retrievers[retriever_type]

        # 创建临时实例来获取支持的功能
        try:
            temp_retriever = retriever_class()
            supported_features = temp_retriever.get_supported_features()
        except Exception:
            supported_features = []

        return {
            'type': retriever_type,
            'class': retriever_class.__name__,
            'description': retriever_class.__doc__ or '无描述',
            'supported_features': supported_features
        }

# 便捷的创建函数
def create_retriever(retriever_type: str = 'auto', **kwargs) -> BaseRetriever:
    """
    便捷函数：创建检索器

    Args:
        retriever_type: 检索器类型
        **kwargs: 传递给检索器的参数

    Returns:
        检索器实例
    """
    return RetrieverFactory.create_retriever(retriever_type, **kwargs)

def create_embedding_retriever(**kwargs) -> EmbeddingRetriever:
    """创建嵌入向量检索器"""
    return RetrieverFactory.create_retriever('embedding', **kwargs)

def create_hybrid_retriever(**kwargs) -> HybridRetriever:
    """创建混合检索器"""
    return RetrieverFactory.create_retriever('hybrid', **kwargs)

from pathlib import Path

# 全局检索器实例缓存
_global_retrievers = {}

def get_global_retriever(retriever_type: str = 'auto', **kwargs) -> BaseRetriever:
    """
    获取全局检索器实例（带缓存）

    Args:
        retriever_type: 检索器类型
        **kwargs: 创建参数

    Returns:
        检索器实例
    """
    global _global_retrievers

    # 创建缓存键
    cache_key = f"{retriever_type}_{str(sorted(kwargs.items()))}"

    if cache_key not in _global_retrievers:
        _global_retrievers[cache_key] = RetrieverFactory.create_retriever(retriever_type, **kwargs)

    return _global_retrievers[cache_key]

def reset_global_retrievers():
    """重置全局检索器缓存"""
    global _global_retrievers
    _global_retrievers = {}
    logger.info("全局检索器缓存已重置")