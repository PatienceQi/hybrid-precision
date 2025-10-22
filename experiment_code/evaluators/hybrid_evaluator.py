"""
混合检索评估器实现
基于信息论的混合检索评估方法
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats

from ..core.api_client import BaseAPIClient, APIClientFactory
from ..core.config import get_config
from ..core.evaluator import BaseEvaluator, EvaluationResult, EvaluationMetrics
from ..core.utils import safe_divide, calculate_similarity

logger = logging.getLogger(__name__)

class HybridEvaluator(BaseEvaluator):
    """混合检索评估器 - 基于信息论的评估方法"""

    def __init__(self, api_client: Optional[BaseAPIClient] = None):
        super().__init__("hybrid")
        self.config = get_config()
        self.api_client = api_client or APIClientFactory.create_client()
        self.supported_metrics = [
            "hybrid_context_precision",
            "hybrid_faithfulness",
            "hybrid_answer_relevancy",
            "hybrid_context_recall",
            "information_entropy",
            "mutual_information",
            "statistical_significance",
            "avg_hybrid_score"
        ]
        self._precision_warning_emitted = False

    def evaluate_single_sample(self,
                             question: str,
                             answer: str,
                             contexts: List[str],
                             reference: str,
                             fusion_method: str = "unknown",
                             **kwargs) -> EvaluationResult:
        """评估单个样本"""
        try:
            self._validate_input_data(question, answer, contexts, reference)

            # 计算基础RAGAS指标
            base_metrics = self._calculate_base_metrics(question, answer, contexts, reference)

            # 计算信息论指标
            information_metrics = self._calculate_information_metrics(contexts, reference, fusion_method)
            fusion_metadata = {
                "fusion_method": information_metrics.pop("fusion_method", fusion_method),
                "fusion_weight": information_metrics.pop("fusion_weight", 1.0),
            }

            # 计算统计显著性
            significance_metrics = self._calculate_statistical_significance(contexts, reference)

            # 合并所有指标
            all_metrics = {**base_metrics, **information_metrics, **significance_metrics}

            # 计算平均混合分数
            all_metrics["avg_hybrid_score"] = self._calculate_avg_hybrid_score(all_metrics)

            evaluation_result = EvaluationResult(
                question=question,
                answer=answer,
                contexts=contexts,
                reference=reference,
                metrics=all_metrics,
                evaluator_type=self.evaluator_type
            )
            evaluation_result.metadata.update(fusion_metadata)
            return evaluation_result

        except Exception as e:
            logger.error(f"混合检索评估失败: {e}")
            return self._create_error_result(question, answer, contexts, reference,
                                           f"混合检索评估失败: {str(e)}")

    def _calculate_base_metrics(self, question: str, answer: str,
                              contexts: List[str], reference: str) -> Dict[str, float]:
        """计算基础指标（混合版本）"""
        return {
            "hybrid_context_precision": EvaluationMetrics.calculate_context_precision(contexts, reference),
            "hybrid_faithfulness": EvaluationMetrics.calculate_faithfulness(answer, contexts),
            "hybrid_answer_relevancy": EvaluationMetrics.calculate_answer_relevancy(question, answer),
            "hybrid_context_recall": EvaluationMetrics.calculate_context_recall(contexts, reference)
        }

    def _calculate_information_metrics(self, contexts: List[str], reference: str,
                                     fusion_method: str) -> Dict[str, float]:
        """计算信息论指标"""
        try:
            # 计算信息熵
            entropy = self._calculate_information_entropy(contexts, reference)

            # 计算互信息
            mutual_info = self._calculate_mutual_information(contexts, reference)

            # 根据融合方法调整权重
            fusion_weights = {
                "cascading": 1.2,
                "weighted_rrf": 1.1,
                "linear_weighted": 1.15,
                "unknown": 1.0
            }

            weight = fusion_weights.get(fusion_method, 1.0)

            return {
                "information_entropy": entropy,
                "mutual_information": mutual_info * weight,
                "fusion_method": fusion_method,
                "fusion_weight": weight
            }

        except Exception as e:
            logger.warning(f"信息论指标计算失败: {e}")
            return {
                "information_entropy": 0.0,
                "mutual_information": 0.0,
                "fusion_method": fusion_method,
                "fusion_weight": 1.0
            }

    def _calculate_information_entropy(self, contexts: List[str], reference: str) -> float:
        """计算信息熵"""
        try:
            # 合并所有上下文
            all_context_text = " ".join(contexts).lower()
            reference_lower = reference.lower()

            # 提取词汇
            context_words = set(all_context_text.split())
            reference_words = set(reference_lower.split())

            if not context_words or not reference_words:
                return 0.0

            # 计算词汇分布
            total_words = len(context_words.union(reference_words))
            context_prob = len(context_words) / total_words
            reference_prob = len(reference_words) / total_words

            # 计算熵
            entropy = 0.0
            if context_prob > 0:
                entropy -= context_prob * np.log2(context_prob)
            if reference_prob > 0:
                entropy -= reference_prob * np.log2(reference_prob)

            return entropy

        except Exception as e:
            logger.warning(f"信息熵计算失败: {e}")
            return 0.0

    def _calculate_mutual_information(self, contexts: List[str], reference: str) -> float:
        """计算互信息"""
        try:
            # 合并所有上下文
            all_context_text = " ".join(contexts).lower()
            reference_lower = reference.lower()

            # 提取词汇
            context_words = set(all_context_text.split())
            reference_words = set(reference_lower.split())

            if not context_words or not reference_words:
                return 0.0

            # 计算交集
            intersection = context_words.intersection(reference_words)

            if not intersection:
                return 0.0

            # 计算互信息（简化版本）
            total_words = len(context_words.union(reference_words))
            joint_prob = len(intersection) / total_words
            context_prob = len(context_words) / total_words
            reference_prob = len(reference_words) / total_words

            # 互信息计算
            if joint_prob > 0 and context_prob > 0 and reference_prob > 0:
                mutual_info = joint_prob * np.log2(joint_prob / (context_prob * reference_prob))
                return max(mutual_info, 0.0)  # 确保非负

            return 0.0

        except Exception as e:
            logger.warning(f"互信息计算失败: {e}")
            return 0.0

    def _calculate_statistical_significance(self, contexts: List[str], reference: str) -> Dict[str, float]:
        """计算统计显著性"""
        try:
            # 合并所有上下文
            all_context_text = " ".join(contexts).lower()
            reference_lower = reference.lower()

            # 提取词汇
            context_words = all_context_text.split()
            reference_words = reference_lower.split()

            if not context_words or not reference_words:
                return {"statistical_significance": 0.0}

            # 计算词频
            from collections import Counter
            context_freq = Counter(context_words)
            reference_freq = Counter(reference_words)

            # 获取共同词汇
            common_words = set(context_freq.keys()).intersection(set(reference_freq.keys()))

            if len(common_words) < 2:
                return {"statistical_significance": 0.0}

            # 计算相关性（简化版卡方检验）
            context_counts = [float(context_freq[word]) for word in common_words]
            reference_counts = [float(reference_freq[word]) for word in common_words]

            # 计算皮尔逊相关系数
            if len(context_counts) >= 2 and len(reference_counts) >= 2:
                correlation, p_value = stats.pearsonr(context_counts, reference_counts)
                significance = max(0.0, 1.0 - p_value) if not np.isnan(p_value) else 0.0
            else:
                significance = 0.0

            return {"statistical_significance": significance}

        except Exception as e:
            logger.warning(f"统计显著性计算失败: {e}")
            return {"statistical_significance": 0.0}

    def _calculate_avg_hybrid_score(self, metrics: Dict[str, float]) -> float:
        """计算平均混合分数"""
        try:
            # 选择关键指标进行平均
            key_metrics = [
                metrics.get("hybrid_context_precision", 0.0),
                metrics.get("hybrid_faithfulness", 0.0),
                metrics.get("hybrid_answer_relevancy", 0.0),
                metrics.get("hybrid_context_recall", 0.0),
                metrics.get("mutual_information", 0.0),
                metrics.get("statistical_significance", 0.0)
            ]

            # 标准化互信息和统计显著性（它们可能大于1）
            if len(key_metrics) >= 6:
                key_metrics[4] = min(key_metrics[4], 1.0)  # 互信息
                key_metrics[5] = min(key_metrics[5], 1.0)  # 统计显著性

            if not key_metrics:
                return 0.0

            return float(np.mean(key_metrics))

        except Exception as e:
            logger.warning(f"平均混合分数计算失败: {e}")
            return 0.0

    def evaluate_batch(self,
                      questions: List[str],
                      answers: List[str],
                      contexts: List[List[str]],
                      references: List[str],
                      fusion_method: str = "unknown",
                      **kwargs) -> List[EvaluationResult]:
        """评估批量样本"""
        if not (len(questions) == len(answers) == len(contexts) == len(references)):
            raise ValueError("输入数据长度不匹配")

        results = []
        for i, (question, answer, context, reference) in enumerate(zip(questions, answers, contexts, references)):
            try:
                result = self.evaluate_single_sample(
                    question, answer, context, reference,
                    fusion_method=fusion_method, **kwargs
                )
                result.sample_id = i
                results.append(result)
            except Exception as e:
                logger.error(f"批量评估中样本 {i} 失败: {e}")
                error_result = self._create_error_result(
                    question, answer, context, reference,
                    f"批量评估失败: {str(e)}", f"样本索引: {i}"
                )
                error_result.sample_id = i
                results.append(error_result)

        return results

    def evaluate_with_hybrid_precision(self,
                                     questions: List[str],
                                     answers: List[str],
                                     contexts: List[List[str]],
                                     references: List[str],
                                     fusion_method: str = "unknown") -> Dict[str, float]:
        """使用混合精确度评估（向后兼容）"""
        try:
            results = self.evaluate_batch(
                questions, answers, contexts, references,
                fusion_method=fusion_method
            )

            # 计算平均指标
            avg_metrics = self._calculate_average_metrics(results)

            # 添加统计信息
            total_samples = len(results)
            error_count = sum(1 for r in results if r.has_error())

            avg_metrics.update({
                'total_samples': total_samples,
                'error_count': error_count,
                'success_rate': (total_samples - error_count) / total_samples if total_samples > 0 else 0.0,
                'fusion_method': fusion_method
            })

            return avg_metrics

        except Exception as e:
            message = f"混合精确度评估失败: {e}，已回退到默认统计结果"
            if not self._precision_warning_emitted:
                logger.warning(message)
                self._precision_warning_emitted = True
            else:
                logger.debug(message)
            return self._error_result(f"评估失败: {str(e)}")

    def _error_result(self, error_msg: str) -> Dict[str, float]:
        """返回错误时的默认结果"""
        return {
            "hybrid_context_precision": 0.0,
            "avg_hybrid_score": 0.0,
            "information_entropy": 0.0,
            "mutual_information": 0.0,
            "statistical_significance": 0.0,
            "error": error_msg
        }

# 向后兼容的函数
def test_hybrid_precision():
    """测试混合精确度评估器（向后兼容）"""
    print("🔧 测试混合检索评估器...")

    evaluator = HybridEvaluator()

    # 测试数据
    test_questions = ["什么是机器学习？", "深度学习的特点是什么？"]
    test_answers = ["机器学习是人工智能的一个分支。", "深度学习使用多层神经网络。"]
    test_contexts = [
        ["机器学习是人工智能的一个分支，通过数据学习模式。"],
        ["深度学习是机器学习的一个子集，使用多层神经网络进行特征提取。"]
    ]
    test_references = ["机器学习是AI的一个分支。", "深度学习使用神经网络。"]

    results = evaluator.evaluate_with_hybrid_precision(
        questions=test_questions,
        answers=test_answers,
        contexts=test_contexts,
        references=test_references,
        fusion_method="weighted_rrf"
    )

    print("✅ 混合检索评估结果:")
    for metric, score in results.items():
        if metric != "error":
            print(f"   {metric}: {score:.4f}")

    if "error" in results:
        print(f"❌ 错误: {results['error']}")

    return results

if __name__ == "__main__":
    test_hybrid_precision()
    print("\n✅ 混合检索评估器测试完成")
