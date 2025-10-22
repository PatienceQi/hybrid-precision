"""
对比实验工具。

提供基线与混合检索实验的快速对比流程，便于观察不同实验类型的指标差异。
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..experiment import ExperimentRunner


def run_comparison_experiment() -> Dict[str, Any]:
    """
    运行对比实验，比较基线与混合检索表现。

    Returns:
        包含各实验类型汇总指标的字典。
    """
    print("🚀 运行对比实验 - 基线 vs 混合检索")

    test_questions = [
        "什么是机器学习？",
        "深度学习和机器学习有什么区别？",
        "自然语言处理的主要应用有哪些？",
    ]

    test_contexts = [
        ["机器学习是人工智能的一个分支，通过数据学习模式。"],
        [
            "深度学习是机器学习的一个子集，使用多层神经网络。",
            "机器学习包括多种算法和技术。",
        ],
        [
            "自然语言处理用于文本分析、机器翻译等应用。",
            "NLP技术包括分词、命名实体识别等。",
        ],
    ]

    test_references = [
        "机器学习是AI的分支，通过数据学习模式。",
        "深度学习使用神经网络，是机器学习的子集。",
        "自然语言处理用于文本分析和机器翻译。",
    ]

    experiment_types: List[str] = ["baseline", "hybrid_standard"]
    all_results: Dict[str, Any] = {}

    for exp_type in experiment_types:
        print(f"\n{'=' * 60}")
        print(f"运行实验类型: {exp_type}")
        print("=" * 60)

        try:
            runner = ExperimentRunner(exp_type)
            results = runner.run_batch_experiment(
                questions=test_questions,
                contexts_list=test_contexts,
                references=test_references,
            )

            summary = runner._calculate_summary_stats(results)
            all_results[exp_type] = summary

            print(f"✅ {exp_type} 实验完成")
            print(f"   总样本数: {summary.get('total_samples', 0)}")
            print(f"   成功率: {summary.get('success_rate', 0):.2%}")

            if exp_type == "baseline":
                key_metrics = ["context_precision", "faithfulness", "answer_relevancy"]
            else:
                key_metrics = ["hybrid_context_precision", "avg_hybrid_score"]

            for metric in key_metrics:
                if metric in summary:
                    metric_summary = summary[metric]
                    if isinstance(metric_summary, dict):
                        mean_value = metric_summary.get("mean", 0.0)
                    else:
                        mean_value = float(metric_summary)
                    print(f"   {metric}: {mean_value:.4f}")

        except Exception as err:  # noqa: BLE001
            print(f"❌ {exp_type} 实验失败: {err}")
            all_results[exp_type] = {"error": str(err)}

    print(f"\n{'=' * 60}")
    print("对比实验结果总结")
    print("=" * 60)

    for exp_type, results in all_results.items():
        if "error" in results:
            print(f"{exp_type}: 实验失败 - {results['error']}")
        else:
            print(f"{exp_type}: 成功率 {results.get('success_rate', 0):.2%}")

    return all_results

