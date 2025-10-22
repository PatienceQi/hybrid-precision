"""
RAGAS评估器实现
基于RAGAS框架的评估器，支持标准RAG评估指标
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
import numpy as np

# 尝试导入RAGAS
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall
    )
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.evaluation import EvaluationResult
    from ragas.llms import BaseRagasLLM
    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.evaluation import EvaluationResult
    from ragas.run_config import RunConfig
    from langchain_core.outputs import Generation, LLMResult
    from langchain_core.prompt_values import PromptValue
    from openai import OpenAI
    import requests
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAS库导入失败，将使用手动实现")

from ..core.api_client import BaseAPIClient, APIClientFactory
from ..core.config import get_config
from ..core.evaluator import BaseEvaluator, EvaluationResult, EvaluationMetrics

logger = logging.getLogger(__name__)

if RAGAS_AVAILABLE:

    class RagasLLMWrapper(BaseRagasLLM):
        """使用 OpenAI 兼容接口（例如 OpenRouter）执行 RAGAS 评估所需的 LLM 调用"""

        def __init__(self, config):
            super().__init__()
            if not config.api.api_key:
                raise ValueError("RAGAS 需要有效的 LLM API key")

            self.config = config
            self.client = OpenAI(
                api_key=config.api.api_key,
                base_url=config.api.base_url,
                default_headers=config.api.default_headers or None,
                timeout=config.api.timeout,
            )
            self.model = config.api.model

        def is_finished(self, response: LLMResult) -> bool:
            return True

        def _chat_completion(
            self,
            prompt_text: str,
            n: int = 1,
            temperature: float = 0.01,
        ) -> LLMResult:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temperature,
                n=n,
                max_tokens=512,
            )

            generations: List[List[Generation]] = []
            for choice in completion.choices:
                content = (choice.message.content or "").strip()
                generations.append([Generation(text=content)])

            return LLMResult(generations=generations)

        def generate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 0.01,
            stop: Optional[List[str]] = None,
            callbacks: Optional[Any] = None,
        ) -> LLMResult:
            prompt_text = prompt.to_string()
            return self._chat_completion(prompt_text, n=n, temperature=temperature)

        async def agenerate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: Optional[float] = 0.01,
            stop: Optional[List[str]] = None,
            callbacks: Optional[Any] = None,
        ) -> LLMResult:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                self.generate_text,
                prompt,
                n,
                temperature,
                stop,
                callbacks,
            )


    class RagasEmbeddingsWrapper(BaseRagasEmbeddings):
        """使用嵌入服务适配 RAGAS embeddings 接口"""

        def __init__(self, service_url: str, model_name: str):
            super().__init__()
            self.service_url = service_url
            self.model_name = model_name
            self.session = requests.Session()

        def _embed(self, text: str) -> List[float]:
            payload = {"model": self.model_name, "input": [text]}
            response = self.session.post(self.service_url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            embedding: Optional[List[float]] = None

            if isinstance(data, dict):
                if "embedding" in data:
                    embedding = data["embedding"]
                elif "embeddings" in data:
                    embeddings_field = data["embeddings"]
                    if isinstance(embeddings_field, list) and embeddings_field:
                        embedding = embeddings_field[0]
                elif "data" in data and isinstance(data["data"], list) and data["data"]:
                    first_item = data["data"][0]
                    if isinstance(first_item, dict) and "embedding" in first_item:
                        embedding = first_item["embedding"]
                    elif isinstance(first_item, list):
                        embedding = first_item
            elif isinstance(data, list) and data:
                embedding = data[0]

            if embedding is None:
                raise ValueError(f"嵌入服务返回异常结构: {data}")

            return [float(value) for value in embedding]

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [self._embed(text) for text in texts]

        def embed_query(self, text: str) -> List[float]:
            return self._embed(text)

        async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.embed_documents, texts)

        async def aembed_query(self, text: str) -> List[float]:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.embed_query, text)

else:

    class RagasLLMWrapper:
        """RAGAS LLM包装器（降级版本）"""

        def __init__(self, api_client: BaseAPIClient):
            self.api_client = api_client
            self.model = api_client.model

        def generate(self, prompts: List[str]) -> List[str]:
            results = []
            for prompt in prompts:
                try:
                    answer = self.api_client.generate_answer(
                        prompt,
                        [{"title": "doc1", "text": prompt}],
                    )
                    results.append(answer)
                except Exception as e:
                    logger.error(f"LLM生成失败: {e}")
                    results.append("基于文档内容，可以确认相关信息存在。")
            return results

        async def agenerate(self, prompts: List[str]) -> List[str]:
            return self.generate(prompts)

class MockRagasLLM:
    """模拟RAGAS LLM"""

    def __init__(self):
        self.model = "gpt-3.5-turbo"

    async def agenerate(self, prompts: List[str]) -> List[str]:
        """异步生成，返回模拟结果"""
        return [self._mock_response(prompt) for prompt in prompts]

    def generate(self, prompts: List[str]) -> List[str]:
        """同步生成，返回模拟结果"""
        return [self._mock_response(prompt) for prompt in prompts]

    def _mock_response(self, prompt: str) -> str:
        """根据提示生成模拟响应"""
        prompt_lower = prompt.lower()

        # 相关性判断
        if "judge relevance" in prompt_lower or "relevance of context" in prompt_lower:
            # 简单启发式：如果上下文较长且包含关键词，认为相关
            if len(prompt) > 200 and any(word in prompt_lower for word in ["document", "context", "information"]):
                return "1"  # 相关
            else:
                return "0"  # 不相关

        # 其他类型的判断
        if "faithfulness" in prompt_lower:
            return "1"  # 默认认为忠实

        if "answer relevancy" in prompt_lower:
            return "1"  # 默认认为相关

        if "context recall" in prompt_lower:
            return "1"  # 默认认为召回

        return "0.5"  # 默认中性回答

class RagasEvaluator(BaseEvaluator):
    """RAGAS评估器"""

    def __init__(self, api_client: Optional[BaseAPIClient] = None, use_mock: bool = False):
        super().__init__("ragas")
        self.config = get_config()
        self.use_mock = use_mock or self.config.evaluation.use_mock_api
        self.api_client = api_client or APIClientFactory.create_client()
        self.supported_metrics = ["context_precision", "faithfulness", "answer_relevancy", "context_recall"]
        self._ragas_warning_emitted = False
        self.ragas_embeddings = None

        # 设置RAGAS LLM
        if RAGAS_AVAILABLE and not self.use_mock:
            try:
                self.ragas_llm = RagasLLMWrapper(self.config)
            except Exception as llm_error:
                logger.warning(f"初始化RAGAS LLM失败，将使用手动指标: {llm_error}")
                self.ragas_llm = None
                self.use_mock = True
            else:
                try:
                    self.ragas_embeddings = RagasEmbeddingsWrapper(
                        service_url=self.config.retrieval.embedding_service_url,
                        model_name=self.config.retrieval.embedding_model,
                    )
                except Exception as embed_error:
                    logger.warning(f"初始化RAGAS嵌入适配器失败，将回退到默认行为: {embed_error}")
                    self.ragas_embeddings = None
        else:
            self.ragas_llm = MockRagasLLM()
            logger.info("使用模拟RAGAS LLM")

    def evaluate_single_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        reference: str,
        **kwargs,
    ) -> EvaluationResult:
        """评估单个样本"""
        try:
            self._validate_input_data(question, answer, contexts, reference)
            custom_metrics = kwargs.get("metrics")

            # 使用RAGAS进行评估
            if RAGAS_AVAILABLE and not self.use_mock:
                return self._evaluate_with_ragas(
                    question,
                    answer,
                    contexts,
                    reference,
                    metrics=custom_metrics,
                )
            else:
                return self._evaluate_manually(question, answer, contexts, reference)

        except Exception as e:
            logger.error(f"RAGAS评估失败: {e}")
            return self._create_error_result(question, answer, contexts, reference,
                                           f"RAGAS评估失败: {str(e)}")

    def _evaluate_with_ragas(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        reference: str,
        metrics: Optional[List] = None,
    ) -> EvaluationResult:
        """使用RAGAS框架进行评估"""
        try:
            if not self.ragas_llm:
                logger.warning("RAGAS LLM 未初始化，回退到手动指标")
                return self._evaluate_manually(question, answer, contexts, reference)

            # RAGAS 当前实现要求每条样本至少包含一个上下文和参考
            if not contexts:
                logger.warning("RAGAS评估跳过：contexts 为空，回退到手动指标")
                return self._evaluate_manually(question, answer, contexts, reference)

            sanitized_contexts: List[str] = []
            for ctx in contexts:
                text_candidate: Optional[str] = None
                if isinstance(ctx, str):
                    text_candidate = ctx
                elif isinstance(ctx, bytes):
                    text_candidate = ctx.decode("utf-8", errors="ignore")
                elif isinstance(ctx, dict):
                    text_candidate = ctx.get("text") or ctx.get("content") or ctx.get("body")
                    if isinstance(text_candidate, list):
                        text_candidate = " ".join(str(part) for part in text_candidate)
                elif isinstance(ctx, (list, tuple)):
                    text_candidate = " ".join(str(part) for part in ctx if isinstance(part, (str, bytes)))

                if text_candidate:
                    sanitized_contexts.append(text_candidate.strip())

            sanitized_contexts = [text for text in sanitized_contexts if text]
            if not sanitized_contexts:
                logger.warning(
                    "RAGAS评估跳过：contexts 无有效文本，原始上下文=%s",
                    contexts,
                )
                return self._evaluate_manually(question, answer, contexts, reference)

            if isinstance(reference, str):
                sanitized_reference = reference
            elif isinstance(reference, bytes):
                sanitized_reference = reference.decode("utf-8", errors="ignore")
            elif isinstance(reference, (list, tuple)):
                sanitized_reference = " ".join(str(part) for part in reference if part)
            else:
                sanitized_reference = str(reference)

            sanitized_reference = sanitized_reference.strip()
            if not sanitized_reference:
                logger.warning(
                    "RAGAS评估跳过：reference 为空或无效，原始reference=%s",
                    reference,
                )
                return self._evaluate_manually(question, answer, contexts, reference)

            # 准备数据
            sample = SingleTurnSample(
                question=question,
                answer=answer,
                contexts=sanitized_contexts,
                ground_truth=sanitized_reference,
            )
            dataset = EvaluationDataset(samples=[sample])

            # 使用RAGAS进行评估
            selected_metrics = metrics or [
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ]

            metrics_dict: Dict[str, float] = {}
            ragas_run_config = RunConfig(max_workers=1)

            for metric_obj in selected_metrics:
                metric_name = getattr(metric_obj, "name", None)
                if not metric_name:
                    continue

                single_kwargs: Dict[str, Any] = {
                    "dataset": dataset,
                    "metrics": [metric_obj],
                    "llm": self.ragas_llm,
                    "show_progress": False,
                    "run_config": ragas_run_config,
                    "raise_exceptions": True,
                }
                if self.ragas_embeddings is not None:
                    single_kwargs["embeddings"] = self.ragas_embeddings

                try:
                    result = evaluate(**single_kwargs)
                    metric_value = self._extract_metric_value(result, metric_name)
                    if metric_value is not None:
                        metrics_dict[metric_name] = metric_value
                except Exception as metric_error:
                    logger.warning(
                        "RAGAS指标 %s 计算失败: %s，已跳过",
                        metric_name,
                        metric_error,
                    )

            if not metrics_dict:
                logger.warning("RAGAS未返回任何指标，回退到手动评估")
                return self._evaluate_manually(question, answer, contexts, reference)

            return EvaluationResult(
                question=question,
                answer=answer,
                contexts=contexts,
                reference=reference,
                metrics=metrics_dict,
                evaluator_type=self.evaluator_type
            )

        except (IndexError, ValueError, TypeError) as e:
            message = f"RAGAS框架评估失败: {e}，已自动回退到手动指标"
            if not self._ragas_warning_emitted:
                logger.exception(message)
                self._ragas_warning_emitted = True
            else:
                logger.exception(message)
            # 回退到手动评估
            return self._evaluate_manually(question, answer, contexts, reference)
        except Exception as e:
            message = f"RAGAS评估发生未处理异常: {e}"
            logger.error(message)
            return self._evaluate_manually(question, answer, contexts, reference)

    def _extract_metric_value(
        self,
        ragas_result: Any,
        metric_name: str,
    ) -> Optional[float]:
        """从RAGAS返回值中提取指标"""
        if isinstance(ragas_result, EvaluationResult):
            scores_df = ragas_result.scores.to_pandas()
            if not scores_df.empty and metric_name in scores_df.columns:
                try:
                    return float(scores_df.iloc[0][metric_name])
                except (TypeError, ValueError):
                    return None
        elif isinstance(ragas_result, dict):
            metric_values = ragas_result.get(metric_name)
            if isinstance(metric_values, list) and metric_values:
                try:
                    return float(metric_values[0])
                except (TypeError, ValueError):
                    return None
        else:
            logger.debug("未识别的RAGAS返回类型: %s", type(ragas_result))
        return None

    def _evaluate_manually(self, question: str, answer: str,
                         contexts: List[str], reference: str) -> EvaluationResult:
        """手动计算RAGAS指标"""
        logger.info("使用手动RAGAS评估")

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

        return EvaluationResult(
            question=question,
            answer=answer,
            contexts=contexts,
            reference=reference,
            metrics=metrics,
            evaluator_type=self.evaluator_type
        )

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

    def safe_evaluate(self,
                     questions: List[str],
                     answers: List[str],
                     contexts: List[List[str]],
                     references: List[str],
                     metrics: Optional[List] = None) -> Dict[str, float]:
        """安全的RAGAS评估（向后兼容）"""
        try:
            results = self.evaluate_batch(questions, answers, contexts, references, metrics=metrics)

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
            logger.error(f"安全评估失败: {e}")
            return self._error_result(f"评估失败: {str(e)}")

    def _error_result(self, error_msg: str) -> Dict[str, float]:
        """返回错误时的默认结果（向后兼容）"""
        return {
            "context_precision": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "error": error_msg
        }

# 向后兼容的函数
def test_fixed_ragas():
    """测试修复版RAGAS评估器（向后兼容）"""
    print("🔧 测试修复版RAGAS评估器...")

    evaluator = RagasEvaluator(use_mock=True)

    # 测试数据
    test_questions = ["什么是机器学习？", "深度学习的特点是什么？"]
    test_answers = ["机器学习是人工智能的一个分支。", "深度学习使用多层神经网络。"]
    test_contexts = [
        ["机器学习是人工智能的一个分支，通过数据学习模式。"],
        ["深度学习是机器学习的一个子集，使用多层神经网络进行特征提取。"]
    ]
    test_references = ["机器学习是AI的一个分支。", "深度学习使用神经网络。"]

    results = evaluator.safe_evaluate(
        questions=test_questions,
        answers=test_answers,
        contexts=test_contexts,
        references=test_references
    )

    print("✅ 修复版RAGAS测试结果:")
    for metric, score in results.items():
        if metric != "error":
            print(f"   {metric}: {score:.4f}")

    if "error" in results:
        print(f"❌ 错误: {results['error']}")

    return results

if __name__ == "__main__":
    test_fixed_ragas()
    print("\n✅ 修复版RAGAS评估器测试完成")
