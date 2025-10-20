"""
RAGAS Hybrid Precision Extension
扩展RAGAS框架以支持混合检索评估
基于InnovativeVersion.md的实现
"""

import logging
import typing as t
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import numpy as np
from pydantic import BaseModel, Field

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
    ensembler,
)
from ragas.prompt import PydanticPrompt
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
else:
    Callbacks = t.Any

logger = logging.getLogger(__name__)


class QACWithScore(BaseModel):
    """扩展的QAC模型，包含混合检索分数"""
    question: str = Field(..., description="问题")
    context: str = Field(..., description="上下文")
    answer: str = Field(..., description="答案")
    hybrid_score: float = Field(..., description="混合检索分数 (0-1)")
    rank_position: int = Field(..., description="排名位置 (从1开始)")


class HybridVerification(BaseModel):
    """混合验证结果"""
    reason: str = Field(..., description="验证原因")
    verdict: int = Field(..., description="二元验证结果 (0/1)")
    hybrid_score: float = Field(..., description="混合检索分数")


class HybridContextPrecisionPrompt(PydanticPrompt[QACWithScore, HybridVerification]):
    """混合上下文精度提示"""
    name: str = "hybrid_context_precision"
    instruction: str = '''给定问题、答案、上下文和混合检索分数，验证上下文是否对得出给定答案有用。
    考虑混合检索分数：高分数表示更好的检索质量。
    给出判决：如果有用则为"1"，如果无用则为"0"。
    输出JSON格式，包含reason、verdict和hybrid_score。'''
    input_model = QACWithScore
    output_model = HybridVerification


@dataclass
class HybridContextPrecision(MetricWithLLM, SingleTurnMetric):
    """
    混合上下文精确度指标

    创新点：
    1. 融入混合检索分数作为权重
    2. 让precision更准确地反映混合检索的融合效果
    3. 高hybrid_score的上下文会放大权重，避免原版对噪声的均匀对待
    """
    name: str = "hybrid_context_precision"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "retrieved_contexts",
                "reference",
                "context_scores",  # 新增：上下文分数
            }
        }
    )
    output_type = MetricOutputType.CONTINUOUS
    hybrid_context_precision_prompt: PydanticPrompt = field(
        default_factory=HybridContextPrecisionPrompt
    )
    max_retries: int = 1

    def _get_row_attributes(self, row: t.Dict) -> t.Tuple[str, t.List[str], t.Any, t.List[float]]:
        """获取行属性，包括混合分数"""
        return (
            row["user_input"],
            row["retrieved_contexts"],
            row["reference"],
            row.get("context_scores", [1.0] * len(row["retrieved_contexts"]))  # 默认分数为1.0
        )

    def _calculate_hybrid_average_precision(
        self, verifications: t.List[HybridVerification]
    ) -> float:
        """计算混合平均精确度"""
        if not verifications:
            return 0.0

        # 提取判决和混合分数
        verdict_list = [1 if ver.verdict else 0 for ver in verifications]
        hybrid_scores = [ver.hybrid_score for ver in verifications]

        # 计算混合权重：hybrid_score * rank_weight
        hybrid_weights = []
        for i, (verdict, hybrid_score) in enumerate(zip(verdict_list, hybrid_scores)):
            rank_weight = 1.0 / (i + 1)  # 排名权重（MRR风格）
            hybrid_weight = hybrid_score * rank_weight  # 创新：融入混合分数
            hybrid_weights.append(hybrid_weight)

        # 计算分子：relevance_i * hybrid_weight_i
        numerator = sum(relevance * weight for relevance, weight in zip(verdict_list, hybrid_weights))

        # 计算分母：sum(hybrid_weight_i)
        denominator = sum(hybrid_weights) + 1e-10  # 避免除零

        score = numerator / denominator

        if np.isnan(score):
            logger.warning("混合上下文精确度计算结果为NaN，返回0.0")
            return 0.0

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        """单轮评分接口"""
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(
        self,
        row: t.Dict,
        callbacks: Callbacks,
    ) -> float:
        """异步评分核心逻辑"""
        assert self.llm is not None, "LLM未设置"

        user_input, retrieved_contexts, reference, context_scores = self._get_row_attributes(row)

        # 确保分数列表长度匹配
        if len(context_scores) != len(retrieved_contexts):
            logger.warning(f"上下文分数数量({len(context_scores)})与上下文数量({len(retrieved_contexts)})不匹配，使用默认分数")
            context_scores = [1.0] * len(retrieved_contexts)

        responses = []
        for i, (context, hybrid_score) in enumerate(zip(retrieved_contexts, context_scores)):
            # 生成混合验证
            verdicts: t.List[HybridVerification] = await self.hybrid_context_precision_prompt.generate_multiple(
                data=QACWithScore(
                    question=user_input,
                    context=context,
                    answer=reference,
                    hybrid_score=hybrid_score,
                    rank_position=i + 1,
                ),
                llm=self.llm,
                callbacks=callbacks,
            )
            responses.append([result.model_dump() for result in verdicts])

        # 聚合回答
        answers = []
        for response in responses:
            agg_answer = ensembler.from_discrete(response, "verdict")
            hybrid_verification = HybridVerification(**agg_answer[0])
            # 确保混合分数被正确传递
            if len(agg_answer[0]) >= 3 and 'hybrid_score' in agg_answer[0]:
                hybrid_verification.hybrid_score = agg_answer[0]['hybrid_score']
            else:
                # 如果没有混合分数，使用默认排名分数
                hybrid_verification.hybrid_score = 1.0 / (len(answers) + 1)
            answers.append(hybrid_verification)

        # 计算混合平均精确度
        score = self._calculate_hybrid_average_precision(answers)
        return score


def create_hybrid_dataset(
    questions: List[str],
    retrieved_contexts: List[List[str]],
    context_scores: List[List[float]],
    answers: List[str],
    references: List[str]
) -> List[Dict[str, Any]]:
    """
    创建混合检索数据集

    Args:
        questions: 问题列表
        retrieved_contexts: 检索上下文列表（每个问题对应一个上下文列表）
        context_scores: 上下文分数列表（与上下文对应）
        answers: 生成答案列表
        references: 参考答案列表

    Returns:
        适用于混合评估的数据集
    """
    dataset = []

    for i, (question, contexts, scores, answer, reference) in enumerate(
        zip(questions, retrieved_contexts, context_scores, answers, references)
    ):
        # 确保分数列表长度匹配
        if len(scores) != len(contexts):
            logger.warning(f"问题 {i}: 分数数量不匹配，使用默认分数")
            scores = [1.0] * len(contexts)

        dataset.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "context_scores": scores,
            "response": answer,
            "reference": reference
        })

    return dataset


def evaluate_hybrid_retrieval(
    questions: List[str],
    retrieved_contexts: List[List[str]],
    context_scores: List[List[float]],
    answers: List[str],
    references: List[str],
    llm_model: str = "gpt-3.5-turbo"
) -> Dict[str, float]:
    """
    评估混合检索系统

    Args:
        questions: 问题列表
        retrieved_contexts: 检索上下文列表
        context_scores: 上下文分数列表（混合检索分数）
        answers: 生成答案列表
        references: 参考答案列表
        llm_model: LLM模型名称

    Returns:
        评估结果字典
    """
    try:
        from ragas import evaluate
        from datasets import Dataset

        # 创建混合数据集
        hybrid_data = create_hybrid_dataset(
            questions=questions,
            retrieved_contexts=retrieved_contexts,
            context_scores=context_scores,
            answers=answers,
            references=references
        )

        # 转换为Dataset格式
        dataset = Dataset.from_list(hybrid_data)

        # 创建混合评估指标
        hybrid_precision = HybridContextPrecision()

        # 运行评估
        results = evaluate(
            dataset=dataset,
            metrics=[hybrid_precision],
            llm_model=llm_model
        )

        return {
            "hybrid_context_precision": float(results["hybrid_context_precision"]),
            "avg_context_scores": np.mean([score for scores in context_scores for score in scores]),
            "total_samples": len(questions)
        }

    except Exception as e:
        logger.error(f"混合检索评估失败: {e}")
        return {
            "hybrid_context_precision": 0.0,
            "avg_context_scores": 0.0,
            "total_samples": len(questions),
            "error": str(e)
        }


# 向后兼容的导出
__all__ = [
    "HybridContextPrecision",
    "create_hybrid_dataset",
    "evaluate_hybrid_retrieval",
    "QACWithScore",
    "HybridVerification"
]