"""
评估器工厂类
提供统一的评估器创建接口
"""

import logging
from typing import Dict, Type, Optional

from core.evaluator import BaseEvaluator
from .ragas_evaluator import RagasEvaluator
from .hybrid_evaluator import HybridEvaluator
from .manual_evaluator import ManualEvaluator

logger = logging.getLogger(__name__)

class EvaluatorFactory:
    """评估器工厂类"""

    _evaluators = {
        'ragas': RagasEvaluator,
        'hybrid': HybridEvaluator,
        'manual': ManualEvaluator
    }

    @classmethod
    def create_evaluator(cls, evaluator_type: str = 'auto', **kwargs) -> BaseEvaluator:
        """
        创建评估器

        Args:
            evaluator_type: 评估器类型 ('ragas', 'hybrid', 'manual', 'auto')
            **kwargs: 传递给评估器的参数

        Returns:
            评估器实例

        Raises:
            ValueError: 如果评估器类型不支持
        """
        if evaluator_type == 'auto':
            # 自动选择评估器类型
            evaluator_type = cls._auto_select_evaluator(**kwargs)

        if evaluator_type not in cls._evaluators:
            raise ValueError(f"不支持的评估器类型: {evaluator_type}")

        evaluator_class = cls._evaluators[evaluator_type]
        logger.info(f"创建评估器: {evaluator_type}")

        try:
            evaluator = evaluator_class(**kwargs)
            logger.info(f"✅ 评估器创建成功: {evaluator_type}")
            return evaluator
        except Exception as e:
            logger.error(f"创建评估器失败: {e}")
            raise

    @classmethod
    def _auto_select_evaluator(cls, **kwargs) -> str:
        """
        自动选择评估器类型

        Returns:
            推荐的评估器类型
        """
        # 检查是否有特定的评估需求
        if kwargs.get('use_hybrid_metrics'):
            return 'hybrid'

        if kwargs.get('use_manual_only'):
            return 'manual'

        # 检查RAGAS是否可用
        try:
            from ragas import evaluate
            return 'ragas'
        except ImportError:
            logger.warning("RAGAS库不可用，回退到手动评估器")
            return 'manual'

    @classmethod
    def register_evaluator(cls, name: str, evaluator_class: Type[BaseEvaluator]):
        """
        注册新的评估器类型

        Args:
            name: 评估器名称
            evaluator_class: 评估器类（必须继承自BaseEvaluator）

        Raises:
            ValueError: 如果评估器类无效
        """
        if not issubclass(evaluator_class, BaseEvaluator):
            raise ValueError("评估器类必须继承自BaseEvaluator")

        cls._evaluators[name] = evaluator_class
        logger.info(f"注册新的评估器类型: {name}")

    @classmethod
    def get_supported_evaluators(cls) -> list:
        """
        获取支持的评估器类型列表

        Returns:
            评估器类型列表
        """
        return list(cls._evaluators.keys())

    @classmethod
    def get_evaluator_info(cls, evaluator_type: str) -> Dict[str, str]:
        """
        获取评估器信息

        Args:
            evaluator_type: 评估器类型

        Returns:
            评估器信息字典
        """
        if evaluator_type not in cls._evaluators:
            raise ValueError(f"不支持的评估器类型: {evaluator_type}")

        evaluator_class = cls._evaluators[evaluator_type]

        # 创建临时实例来获取支持的指标
        try:
            temp_evaluator = evaluator_class()
            supported_metrics = temp_evaluator.get_supported_metrics()
        except Exception:
            supported_metrics = []

        return {
            'type': evaluator_type,
            'class': evaluator_class.__name__,
            'description': evaluator_class.__doc__ or '无描述',
            'supported_metrics': supported_metrics
        }

# 便捷的创建函数
def create_evaluator(evaluator_type: str = 'auto', **kwargs) -> BaseEvaluator:
    """
    便捷函数：创建评估器

    Args:
        evaluator_type: 评估器类型
        **kwargs: 传递给评估器的参数

    Returns:
        评估器实例
    """
    return EvaluatorFactory.create_evaluator(evaluator_type, **kwargs)

def create_ragas_evaluator(**kwargs) -> RagasEvaluator:
    """创建RAGAS评估器"""
    return EvaluatorFactory.create_evaluator('ragas', **kwargs)

def create_hybrid_evaluator(**kwargs) -> HybridEvaluator:
    """创建混合检索评估器"""
    return EvaluatorFactory.create_evaluator('hybrid', **kwargs)

def create_manual_evaluator(**kwargs) -> ManualEvaluator:
    """创建手动评估器"""
    return EvaluatorFactory.create_evaluator('manual', **kwargs)

# 全局评估器实例缓存
_global_evaluators = {}

def get_global_evaluator(evaluator_type: str = 'auto', **kwargs) -> BaseEvaluator:
    """
    获取全局评估器实例（带缓存）

    Args:
        evaluator_type: 评估器类型
        **kwargs: 创建参数

    Returns:
        评估器实例
    """
    global _global_evaluators

    # 创建缓存键
    cache_key = f"{evaluator_type}_{str(sorted(kwargs.items()))}"

    if cache_key not in _global_evaluators:
        _global_evaluators[cache_key] = EvaluatorFactory.create_evaluator(evaluator_type, **kwargs)

    return _global_evaluators[cache_key]

def reset_global_evaluators():
    """重置全局评估器缓存"""
    global _global_evaluators
    _global_evaluators = {}
    logger.info("全局评估器缓存已重置")