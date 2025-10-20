"""
手动评估器实现
不依赖外部库的纯Python评估器
"""

import logging
from typing import Dict, List, Any, Optional

from core.evaluator import BaseEvaluator, EvaluationResult, EvaluationMetrics
from core.api_client import BaseAPIClient, APIClientFactory
from core.config import get_config

logger = logging.getLogger(__name__)

class ManualEvaluator(BaseEvaluator):
    """手动评估器 - 纯Python实现，无外部依赖"""

    def __init__(self, api_client: Optional[BaseAPIClient] = None):
        super().__init__("manual")
        self.config = get_config()
        self.api_client = api_client or APIClientFactory.create_client()
        self.supported_metrics = [
            "context_precision",
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "avg_manual_score"
        ]

    def evaluate_single_sample(self,
                             question: str,
                             answer: str,
                             contexts: List[str],
                             reference: str,
                             **kwargs) -> EvaluationResult:
        """评估单个样本"""
        try:
            self._validate_input_data(question, answer, contexts, reference)

            # 计算各项指标
            metrics = {}

            # 1. 上下文精确度
            metrics['context_precision'] = EvaluationMetrics.calculate_context_precision(contexts, reference)

            # 2. 忠实度
            metrics['faithfulness'] = EvaluationMetrics.calculate_faithfulness(answer, contexts)

            # 3. 答案相关性
            metrics['answer_relevancy'] = EvaluationMetrics.calculate_answer_relevancy(question, answer)

            # 4. 上下文召回率
            metrics['context_recall'] = EvaluationMetrics.calculate_context_recall(contexts, reference)

            # 5. 平均分数
            metrics['avg_manual_score'] = self._calculate_avg_score(metrics)

            return EvaluationResult(
                question=question,
                answer=answer,
                contexts=contexts,
                reference=reference,
                metrics=metrics,
                evaluator_type=self.evaluator_type
            )

        except Exception as e:
            logger.error(f"手动评估失败: {e}")
            return self._create_error_result(question, answer, contexts, reference,
                                           f"手动评估失败: {str(e)}")

    def _calculate_avg_score(self, metrics: Dict[str, float]) -> float:
        """计算平均分数"""
        try:
            # 选择基础指标进行平均
            base_metrics = [
                metrics.get("context_precision", 0.0),
                metrics.get("faithfulness", 0.0),
                metrics.get("answer_relevancy", 0.0),
                metrics.get("context_recall", 0.0)
            ]

            if not base_metrics:
                return 0.0

            return sum(base_metrics) / len(base_metrics)

        except Exception as e:
            logger.warning(f"平均分数计算失败: {e}")
            return 0.0

    def evaluate_batch(self,
                      questions: List[str],
                      answers: List[str],
                      contexts: List[List[str]],
                      references: List[str],
                      **kwargs) -> List[EvaluationResult]:
        """评估批量样本"""
        if not (len(questions) == len(answers) == len(contexts) == len(references)):
            raise ValueError("输入数据长度不匹配")

        results = []
        for i, (question, answer, context, reference) in enumerate(zip(questions, answers, contexts, references)):
            try:
                result = self.evaluate_single_sample(question, answer, context, reference, **kwargs)
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

    def evaluate_simple(self,
                       questions: List[str],
                       answers: List[str],
                       contexts: List[List[str]],
                       references: List[str]) -> Dict[str, float]:
        """简单的手动评估（向后兼容）"""
        try:
            results = self.evaluate_batch(questions, answers, contexts, references)

            # 计算平均指标
            avg_metrics = self._calculate_average_metrics(results)

            # 添加统计信息
            total_samples = len(results)
            error_count = sum(1 for r in results if r.has_error())

            avg_metrics.update({
                'total_samples': total_samples,
                'error_count': error_count,
                'success_rate': (total_samples - error_count) / total_samples if total_samples > 0 else 0.0
            })

            return avg_metrics

        except Exception as e:
            logger.error(f"简单评估失败: {e}")
            return self._error_result(f"评估失败: {str(e)}")

    def _error_result(self, error_msg: str) -> Dict[str, float]:
        """返回错误时的默认结果"""
        return {
            "context_precision": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "avg_manual_score": 0.0,
            "error": error_msg
        }

# 向后兼容的函数
def test_manual_ragas():
    """测试手动RAGAS评估器（向后兼容）"""
    print("🔧 测试手动RAGAS评估器...")

    evaluator = ManualEvaluator()

    # 测试数据
    test_questions = ["什么是机器学习？", "深度学习的特点是什么？"]
    test_answers = ["机器学习是人工智能的一个分支。", "深度学习使用多层神经网络。"]
    test_contexts = [
        ["机器学习是人工智能的一个分支，通过数据学习模式。"],
        ["深度学习是机器学习的一个子集，使用多层神经网络进行特征提取。"]
    ]
    test_references = ["机器学习是AI的一个分支。", "深度学习使用神经网络。"]

    results = evaluator.evaluate_simple(
        questions=test_questions,
        answers=test_answers,
        contexts=test_contexts,
        references=test_references
    )

    print("✅ 手动RAGAS测试结果:")
    for metric, score in results.items():
        if metric != "error":
            print(f"   {metric}: {score:.4f}")

    if "error" in results:
        print(f"❌ 错误: {results['error']}")

    return results

if __name__ == "__main__":
    test_manual_ragas()
    print("\n✅ 手动RAGAS评估器测试完成")