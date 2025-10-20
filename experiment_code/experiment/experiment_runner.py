"""
实验运行器 - 统一的实验执行接口
支持多种实验类型的运行和管理
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from datetime import datetime

from core.config import get_config
from core.utils import setup_logging, save_json_file, load_json_file, format_time, create_progress_bar
from evaluators import create_evaluator, BaseEvaluator
from retrievers import create_retriever, BaseRetriever
from generators import ResponseGenerator
from .batch_manager import BatchExperimentManager

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """实验运行器 - 统一的实验执行接口"""

    def __init__(self, experiment_type: str = "baseline"):
        self.config = get_config()
        self.experiment_type = experiment_type
        self.start_time = time.time()
        self.results = []
        self.errors = []

        # 设置日志
        self.logger = setup_logging(level=self.config.experiment.log_level)

        # 初始化组件
        self.evaluator = None
        self.retriever = None
        self.response_generator = None

        self._initialize_components()

    def _initialize_components(self):
        """初始化实验组件"""
        try:
            # 根据实验类型选择评估器
            if self.experiment_type == "baseline":
                self.evaluator = create_evaluator("ragas")
            elif self.experiment_type == "hybrid_standard":
                self.evaluator = create_evaluator("hybrid")
            elif self.experiment_type == "hybrid_precision":
                self.evaluator = create_evaluator("hybrid")
            else:
                raise ValueError(f"不支持的实验类型: {self.experiment_type}")

            # 创建检索器
            self.retriever = create_retriever("hybrid")

            # 创建响应生成器
            fusion_method = self._get_fusion_method()
            self.response_generator = ResponseGenerator(
                retriever=self.retriever,
                fusion_method=fusion_method
            )

            self.logger.info(f"✅ 实验组件初始化完成 - 类型: {self.experiment_type}")

        except Exception as e:
            self.logger.error(f"实验组件初始化失败: {e}")
            raise

    def _get_fusion_method(self) -> str:
        """获取融合方法"""
        fusion_methods = {
            "baseline": "unknown",
            "hybrid_standard": "weighted_rrf",
            "hybrid_precision": "weighted_rrf"
        }
        return fusion_methods.get(self.experiment_type, "unknown")

    def run_single_experiment(self, question: str, contexts: List[str], reference: str) -> Dict[str, Any]:
        """
        运行单个实验

        Args:
            question: 问题
            contexts: 上下文列表
            reference: 参考答案

        Returns:
            实验结果
        """
        try:
            # 生成回答
            response_data = self.response_generator.generate_response(question)
            answer = response_data.get('answer', '')

            # 评估结果
            if self.experiment_type == "hybrid_precision":
                # 混合精确度评估
                evaluation_result = self.evaluator.evaluate_single_sample(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    reference=reference,
                    fusion_method=self._get_fusion_method()
                )
            else:
                # 标准评估
                evaluation_result = self.evaluator.evaluate_single_sample(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    reference=reference
                )

            return {
                'question': question,
                'answer': answer,
                'contexts': contexts,
                'reference': reference,
                'evaluation': evaluation_result.to_dict(),
                'retrieval_metadata': response_data.get('retrieval_metadata', {}),
                'fusion_method': response_data.get('fusion_method', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"单个实验运行失败: {e}")
            error_result = {
                'question': question,
                'answer': '',
                'contexts': contexts,
                'reference': reference,
                'evaluation': {
                    'question': question,
                    'answer': '',
                    'contexts': contexts,
                    'reference': reference,
                    'metrics': self._get_default_error_metrics(),
                    'error': str(e),
                    'evaluator_type': self.evaluator.evaluator_type
                },
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return error_result

    def run_batch_experiment(self, questions: List[str], contexts_list: List[List[str]],
                           references: List[str], batch_size: int = None) -> List[Dict[str, Any]]:
        """
        运行批量实验

        Args:
            questions: 问题列表
            contexts_list: 上下文列表的列表
            references: 参考答案列表
            batch_size: 批次大小

        Returns:
            实验结果列表
        """
        if not (len(questions) == len(contexts_list) == len(references)):
            raise ValueError("输入数据长度不匹配")

        batch_size = batch_size or self.config.evaluation.batch_size
        total_samples = len(questions)

        self.logger.info(f"开始批量实验 - 总样本数: {total_samples}, 批次大小: {batch_size}")

        results = []
        for i in range(0, total_samples, batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_contexts = contexts_list[i:i + batch_size]
            batch_references = references[i:i + batch_size]

            self.logger.info(f"处理批次 {i//batch_size + 1}: 样本 {i+1}-{min(i+batch_size, total_samples)}")

            # 处理当前批次
            for j, (question, contexts, reference) in enumerate(zip(batch_questions, batch_contexts, batch_references)):
                try:
                    sample_idx = i + j
                    result = self.run_single_experiment(question, contexts, reference)
                    results.append(result)

                    # 显示进度
                    if (sample_idx + 1) % 10 == 0:
                        progress = (sample_idx + 1) / total_samples * 100
                        self.logger.info(f"进度: {progress:.1f}% ({sample_idx + 1}/{total_samples})")

                except Exception as e:
                    self.logger.error(f"批量实验样本 {i + j} 失败: {e}")
                    error_result = {
                        'question': question,
                        'answer': '',
                        'contexts': contexts,
                        'reference': reference,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(error_result)

        self.logger.info(f"✅ 批量实验完成 - 处理样本数: {len(results)}")
        return results

    def run_experiment_with_manager(self, batch_id: int, data_file: str = None) -> Optional[Dict[str, Any]]:
        """
        使用批次管理器运行实验

        Args:
            batch_id: 批次ID
            data_file: 数据文件路径（可选）

        Returns:
            实验结果
        """
        try:
            # 创建批次管理器
            manager = BatchExperimentManager(batch_id, self.experiment_type)

            # 加载之前的进度
            if manager.load_previous_progress():
                if manager.batch_stats.status == 'completed':
                    self.logger.info(f"批次 {batch_id} 已完成，跳过处理")
                    return manager.finalize_batch({})

            # 加载数据
            if data_file:
                samples = load_json_file(data_file)
            else:
                samples = manager.load_batch_dataset()

            if not samples:
                self.logger.error(f"无法加载批次 {batch_id} 的数据")
                return None

            # 运行实验
            self.logger.info(f"开始处理批次 {batch_id} - 样本数: {len(samples)}")

            for i, sample in enumerate(samples[manager.current_sample_idx:], manager.current_sample_idx):
                try:
                    question = sample.get('question', '')
                    contexts = sample.get('contexts', [])
                    reference = sample.get('reference', '')

                    if not all([question, contexts, reference]):
                        manager.add_error_result(i, "缺少必要数据字段")
                        continue

                    # 运行实验
                    result = self.run_single_experiment(question, contexts, reference)

                    # 提取评估指标
                    metrics = result['evaluation'].get('metrics', {})

                    # 记录结果
                    manager.add_sample_result(i, metrics, {
                        'sample_id': i + 1,
                        'question': question,
                        'answer': result.get('answer', ''),
                        'contexts_count': len(contexts),
                        'metrics': metrics,
                        'error': result.get('error'),
                        'timestamp': result.get('timestamp')
                    })

                    # 显示进度
                    if (i + 1) % 10 == 0:
                        manager.print_progress()

                except Exception as e:
                    self.logger.error(f"处理样本 {i} 失败: {e}")
                    manager.add_error_result(i, str(e))

            # 计算汇总统计
            summary_stats = self._calculate_summary_stats(manager.results)

            # 完成批次
            final_results = manager.finalize_batch(summary_stats)

            self.logger.info(f"✅ 批次 {batch_id} 实验完成")
            return final_results

        except Exception as e:
            self.logger.error(f"批次实验运行失败: {e}")
            return None

    def _calculate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算汇总统计"""
        if not results:
            return {}

        # 收集所有指标
        all_metrics = {}
        for result in results:
            metrics = result.get('metrics', {})
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        # 计算统计信息
        summary_stats = {}
        for metric_name, values in all_metrics.items():
            if values:
                summary_stats[metric_name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }

        # 添加总体统计
        total_samples = len(results)
        error_count = sum(1 for r in results if r.get('error'))

        summary_stats.update({
            'total_samples': total_samples,
            'error_count': error_count,
            'success_rate': (total_samples - error_count) / total_samples if total_samples > 0 else 0.0,
            'experiment_type': self.experiment_type
        })

        return summary_stats

    def _get_default_error_metrics(self) -> Dict[str, float]:
        """获取默认错误指标"""
        if self.experiment_type == 'hybrid_precision':
            return {
                'hybrid_context_precision': 0.0,
                'avg_hybrid_score': 0.0
            }
        else:
            return {
                'context_precision': 0.0,
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_recall': 0.0
            }

    def save_results(self, results: List[Dict[str, Any]], output_file: str) -> None:
        """保存实验结果"""
        try:
            output_data = {
                'experiment_type': self.experiment_type,
                'total_samples': len(results),
                'timestamp': datetime.now().isoformat(),
                'config': self.config.to_dict(),
                'results': results,
                'summary': self._calculate_summary_stats(results)
            }

            save_json_file(output_data, output_file)
            self.logger.info(f"✅ 实验结果已保存: {output_file}")

        except Exception as e:
            self.logger.error(f"保存实验结果失败: {e}")
            raise

    def get_experiment_info(self) -> Dict[str, Any]:
        """获取实验信息"""
        return {
            'experiment_type': self.experiment_type,
            'evaluator_type': self.evaluator.evaluator_type if self.evaluator else None,
            'retriever_type': self.retriever.retriever_type if self.retriever else None,
            'supported_metrics': self.evaluator.get_supported_metrics() if self.evaluator else [],
            'fusion_method': self._get_fusion_method(),
            'config': self.config.get_config_info() if hasattr(self.config, 'get_config_info') else {}
        }

    def test_setup(self) -> bool:
        """测试实验设置"""
        try:
            self.logger.info("测试实验设置...")

            # 测试评估器
            if self.evaluator:
                self.logger.info(f"✅ 评估器测试通过: {self.evaluator.evaluator_type}")

            # 测试检索器
            if self.retriever:
                self.logger.info(f"✅ 检索器测试通过: {self.retriever.retriever_type}")

            # 测试响应生成器
            if self.response_generator:
                self.logger.info(f"✅ 响应生成器测试通过")

            # 测试API连接
            if hasattr(self.response_generator, 'api_client'):
                if self.response_generator.api_client.test_connection():
                    self.logger.info("✅ API连接测试通过")
                else:
                    self.logger.warning("⚠️  API连接测试失败")

            self.logger.info("✅ 实验设置测试完成")
            return True

        except Exception as e:
            self.logger.error(f"实验设置测试失败: {e}")
            return False

# 便捷的创建函数
def create_experiment_runner(experiment_type: str = "baseline") -> ExperimentRunner:
    """创建实验运行器"""
    return ExperimentRunner(experiment_type)

def run_single_experiment(question: str, contexts: List[str], reference: str,
                         experiment_type: str = "baseline") -> Dict[str, Any]:
    """运行单个实验（便捷函数）"""
    runner = ExperimentRunner(experiment_type)
    return runner.run_single_experiment(question, contexts, reference)

def run_batch_experiment(questions: List[str], contexts_list: List[List[str]],
                        references: List[str], experiment_type: str = "baseline",
                        batch_size: int = None) -> List[Dict[str, Any]]:
    """运行批量实验（便捷函数）"""
    runner = ExperimentRunner(experiment_type)
    return runner.run_batch_experiment(questions, contexts_list, references, batch_size)

if __name__ == "__main__":
    # 测试实验运行器
    print("🔧 测试实验运行器...")

    # 创建运行器
    runner = ExperimentRunner("baseline")

    print(f"✅ 实验运行器创建成功")
    print(f"✅ 实验类型: {runner.experiment_type}")
    print(f"✅ 实验信息: {runner.get_experiment_info()}")

    # 测试设置
    if runner.test_setup():
        print("✅ 实验设置测试通过")
    else:
        print("❌ 实验设置测试失败")

    print("\n✅ 实验运行器测试完成")