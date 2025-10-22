"""
评估器基类和评估结果定义
提供统一的评估接口和结果格式
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from numbers import Real
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """评估结果"""

    # 基本信息
    question: str
    answer: str
    contexts: List[str]
    reference: str

    # 评估指标
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 元数据
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    evaluator_type: str = ""
    sample_id: Optional[int] = None

    # 错误信息
    error: Optional[str] = None
    error_context: Optional[str] = None

    def add_metric(self, name: str, value: float):
        """添加评估指标"""
        self.metrics[name] = value

    def get_metric(self, name: str, default: float = 0.0) -> float:
        """获取评估指标"""
        return self.metrics.get(name, default)

    def has_error(self) -> bool:
        """检查是否有错误"""
        return self.error is not None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'question': self.question,
            'answer': self.answer,
            'contexts': self.contexts,
            'reference': self.reference,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'evaluator_type': self.evaluator_type,
            'sample_id': self.sample_id,
            'error': self.error,
            'error_context': self.error_context
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """从字典创建评估结果"""
        return cls(
            question=data['question'],
            answer=data['answer'],
            contexts=data['contexts'],
            reference=data['reference'],
            metrics=data.get('metrics', {}),
            metadata=data.get('metadata', {}),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            evaluator_type=data.get('evaluator_type', ''),
            sample_id=data.get('sample_id'),
            error=data.get('error'),
            error_context=data.get('error_context'),
        )

class BaseEvaluator(ABC):
    """评估器基类"""

    def __init__(self, evaluator_type: str = "base"):
        self.evaluator_type = evaluator_type
        self.supported_metrics = []
        self._validate_setup()

    @abstractmethod
    def evaluate_single_sample(self,
                             question: str,
                             answer: str,
                             contexts: List[str],
                             reference: str,
                             **kwargs) -> EvaluationResult:
        """评估单个样本"""
        pass

    @abstractmethod
    def evaluate_batch(self,
                      questions: List[str],
                      answers: List[str],
                      contexts: List[List[str]],
                      references: List[str],
                      **kwargs) -> List[EvaluationResult]:
        """评估批量样本"""
        pass

    def _validate_setup(self):
        """验证评估器设置"""
        if not self.evaluator_type:
            raise ValueError("评估器类型不能为空")

    def _validate_input_data(self, question: str, answer: str, contexts: List[str], reference: str):
        """验证输入数据"""
        if not question or not isinstance(question, str):
            raise ValueError("问题必须是有效的字符串")

        if not answer or not isinstance(answer, str):
            raise ValueError("答案必须是有效的字符串")

        if not contexts or not isinstance(contexts, list):
            raise ValueError("上下文必须是有效的列表")

        if not reference or not isinstance(reference, str):
            raise ValueError("参考必须是有效的字符串")

    def _create_error_result(self, question: str, answer: str, contexts: List[str],
                           reference: str, error: str, error_context: str = "") -> EvaluationResult:
        """创建错误结果"""
        return EvaluationResult(
            question=question,
            answer=answer,
            contexts=contexts,
            reference=reference,
            evaluator_type=self.evaluator_type,
            error=error,
            error_context=error_context
        )

    def _calculate_average_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """计算平均指标"""
        if not results:
            return {}

        # 收集所有指标名称
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())

        # 计算平均值
        avg_metrics = {}
        for metric in all_metrics:
            values: List[float] = []
            for result in results:
                if result.has_error():
                    continue
                value = result.get_metric(metric)
                if isinstance(value, Real):
                    values.append(float(value))
            if values:
                avg_metrics[metric] = float(np.mean(values))
            else:
                avg_metrics[metric] = 0.0

        return avg_metrics

    def get_supported_metrics(self) -> List[str]:
        """获取支持的评估指标"""
        return self.supported_metrics.copy()

    def validate_metrics(self, metrics: List[str]) -> List[str]:
        """验证评估指标"""
        if not metrics:
            return self.supported_metrics

        invalid_metrics = [m for m in metrics if m not in self.supported_metrics]
        if invalid_metrics:
            logger.warning(f"不支持的评估指标: {invalid_metrics}")
            metrics = [m for m in metrics if m in self.supported_metrics]

        return metrics

class EvaluationError(Exception):
    """评估错误"""
    pass

class EvaluationMetrics:
    """评估指标工具类"""

    @staticmethod
    def calculate_context_precision(contexts: List[str], reference: str) -> float:
        """计算上下文精确度"""
        if not contexts or not reference:
            return 0.0

        total_score = 0.0
        reference_lower = reference.lower()

        for i, context in enumerate(contexts):
            context_lower = context.lower()

            # 计算共同关键词
            ref_words = set(reference_lower.split())
            ctx_words = set(context_lower.split())

            if ref_words:
                overlap = len(ref_words.intersection(ctx_words))
                relevance = overlap / len(ref_words)
                # 位置权重（越靠前越重要）
                position_weight = 1.0 / (i + 1)
                total_score += relevance * position_weight

        # 标准化分数
        max_possible = sum(1.0 / (i + 1) for i in range(len(contexts)))
        return total_score / max_possible if max_possible > 0 else 0.0

    @staticmethod
    def calculate_faithfulness(answer: str, contexts: List[str]) -> float:
        """计算忠实度"""
        if not answer or not contexts:
            return 0.0

        answer_lower = answer.lower()
        all_context_text = " ".join(contexts).lower()

        # 计算答案中能在上下文中找到的内容比例
        answer_words = answer_lower.split()
        context_words = set(all_context_text.split())

        if not answer_words:
            return 0.0

        faithful_words = sum(1 for word in answer_words if word in context_words)
        return faithful_words / len(answer_words)

    @staticmethod
    def calculate_answer_relevancy(question: str, answer: str) -> float:
        """计算答案相关性"""
        if not question or not answer:
            return 0.0

        # 基于问题和答案的关键词重叠
        question_lower = question.lower()
        answer_lower = answer.lower()

        question_words = set(question_lower.split())
        answer_words = set(answer_lower.split())

        if not question_words:
            return 0.0

        overlap = len(question_words.intersection(answer_words))
        return overlap / len(question_words)

    @staticmethod
    def calculate_context_recall(contexts: List[str], reference: str) -> float:
        """计算上下文召回率"""
        if not contexts or not reference:
            return 0.0

        # 检查上下文是否覆盖了参考内容的关键部分
        reference_lower = reference.lower()
        ref_words = set(reference_lower.split())

        all_context_text = " ".join(contexts).lower()
        context_words = set(all_context_text.split())

        if not ref_words:
            return 0.0

        overlap = len(ref_words.intersection(context_words))
        return overlap / len(ref_words)

    @staticmethod
    def calculate_hybrid_precision(contexts: List[str], reference: str,
                                 fusion_method: str = "unknown") -> Dict[str, float]:
        """计算混合检索精确度"""
        base_precision = EvaluationMetrics.calculate_context_precision(contexts, reference)

        # 根据融合方法调整权重
        fusion_weights = {
            "cascading": 1.2,
            "weighted_rrf": 1.1,
            "linear_weighted": 1.15,
            "unknown": 1.0
        }

        weight = fusion_weights.get(fusion_method, 1.0)
        hybrid_precision = min(base_precision * weight, 1.0)

        return {
            "hybrid_context_precision": hybrid_precision,
            "base_context_precision": base_precision,
            "fusion_method": fusion_method,
            "fusion_weight": weight
        }
