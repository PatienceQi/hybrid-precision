"""
集成测试与快速自检工具。

提供一组可在命令行触发的测试函数，用于验证配置、检索器、评估器以及
实验运行器等核心组件是否正常工作。
"""

from __future__ import annotations

from typing import Callable, List, Tuple

from ..core.config import get_config
from ..evaluators import create_evaluator
from ..experiment import ExperimentRunner
from ..generators import LLMClient, ResponseGenerator
from ..retrievers import create_retriever


def test_basic_functionality() -> bool:
    """测试核心组件是否能够正常创建。"""
    print("🔧 测试基础功能...")

    try:
        config = get_config()
        print(f"✅ 配置加载成功 - API模型: {config.api.model}")

        evaluator = create_evaluator("manual")
        print(f"✅ 评估器创建成功 - 类型: {evaluator.evaluator_type}")

        retriever = create_retriever("embedding")
        print(f"✅ 检索器创建成功 - 类型: {retriever.retriever_type}")

        generator = ResponseGenerator()
        print("✅ 响应生成器创建成功")

        llm_client = LLMClient()
        print(f"✅ LLM客户端创建成功 - 模型: {llm_client.model}")

        print("✅ 基础功能测试通过")
        return True

    except Exception as err:  # noqa: BLE001 - 捕获所有异常以便反馈
        print(f"❌ 基础功能测试失败: {err}")
        return False


def test_evaluation_functionality() -> bool:
    """测试手动与混合评估流程。"""
    print("\n🔧 测试评估功能...")

    try:
        test_questions = ["什么是机器学习？", "深度学习的特点是什么？"]
        test_answers = ["机器学习是人工智能的一个分支。", "深度学习使用多层神经网络。"]
        test_contexts = [
            ["机器学习是人工智能的一个分支，通过数据学习模式。"],
            ["深度学习是机器学习的一个子集，使用多层神经网络进行特征提取。"],
        ]
        test_references = ["机器学习是AI的一个分支。", "深度学习使用神经网络。"]

        print("测试RAGAS评估器...")
        ragas_evaluator = create_evaluator("manual")
        ragas_results = ragas_evaluator.evaluate_simple(
            questions=test_questions,
            answers=test_answers,
            contexts=test_contexts,
            references=test_references,
        )
        print(f"✅ RAGAS评估完成 - 平均分数: {ragas_results.get('avg_manual_score', 0):.4f}")

        print("测试混合评估器...")
        hybrid_evaluator = create_evaluator("hybrid")
        hybrid_results = hybrid_evaluator.evaluate_with_hybrid_precision(
            questions=test_questions,
            answers=test_answers,
            contexts=test_contexts,
            references=test_references,
            fusion_method="weighted_rrf",
        )
        print(f"✅ 混合评估完成 - 平均混合分数: {hybrid_results.get('avg_hybrid_score', 0):.4f}")

        print("✅ 评估功能测试通过")
        return True

    except Exception as err:  # noqa: BLE001
        print(f"❌ 评估功能测试失败: {err}")
        return False


def test_experiment_runner() -> bool:
    """测试实验运行器的单实验流程。"""
    print("\n🔧 测试实验运行器...")

    try:
        runner = ExperimentRunner("baseline")
        print(f"✅ 实验运行器创建成功 - 类型: {runner.experiment_type}")

        if runner.test_setup():
            print("✅ 实验设置测试通过")
        else:
            print("⚠️  实验设置测试失败（可能缺少API配置）")

        print("测试单个实验...")
        test_result = runner.run_single_experiment(
            question="什么是机器学习？",
            contexts=["机器学习是人工智能的一个分支，通过数据学习模式。"],
            reference="机器学习是AI的一个分支。",
        )
        print(f"✅ 单个实验完成 - 评估器: {test_result['evaluation']['evaluator_type']}")

        print("✅ 实验运行器测试通过")
        return True

    except Exception as err:  # noqa: BLE001
        print(f"❌ 实验运行器测试失败: {err}")
        return False


def run_smoke_tests() -> None:
    """运行基础冒烟测试（基础功能 + 评估流程）。"""
    test_basic_functionality()
    test_evaluation_functionality()


def run_full_integration_test() -> bool:
    """运行完整的集成测试套件。"""
    print("🚀 运行完整的集成测试...")

    tests: List[Tuple[str, Callable[[], bool]]] = [
        ("基础功能", test_basic_functionality),
        ("评估功能", test_evaluation_functionality),
        ("实验运行器", test_experiment_runner),
    ]

    passed = 0
    total = len(tests)

    for name, func in tests:
        print(f"\n{'=' * 50}")
        print(f"运行测试: {name}")
        print("=" * 50)

        if func():
            passed += 1
            print(f"✅ {name} 测试通过")
        else:
            print(f"❌ {name} 测试失败")

    print(f"\n{'=' * 50}")
    print(f"集成测试完成: {passed}/{total} 测试通过")
    print("=" * 50)

    return passed == total
