"""
混合检索评估系统 - 主入口文件
提供统一的命令行接口和实验运行功能
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import get_config, reload_config
from core.utils import setup_logging, format_time
from experiment import ExperimentRunner, BatchExperimentManager
from evaluators import create_evaluator
from retrievers import create_retriever
from generators import ResponseGenerator, LLMClient
from knowledge_base import SimpleKnowledgeBuilder, run_simple_setup

def test_basic_functionality():
    """测试基础功能"""
    print("🔧 测试基础功能...")

    try:
        # 测试配置
        config = get_config()
        print(f"✅ 配置加载成功 - API模型: {config.api.model}")

        # 测试评估器
        evaluator = create_evaluator("manual")
        print(f"✅ 评估器创建成功 - 类型: {evaluator.evaluator_type}")

        # 测试检索器
        retriever = create_retriever("embedding")
        print(f"✅ 检索器创建成功 - 类型: {retriever.retriever_type}")

        # 测试响应生成器
        generator = ResponseGenerator()
        print(f"✅ 响应生成器创建成功")

        # 测试LLM客户端
        llm_client = LLMClient()
        print(f"✅ LLM客户端创建成功 - 模型: {llm_client.model}")

        print("✅ 基础功能测试通过")
        return True

    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        return False

def test_evaluation_functionality():
    """测试评估功能"""
    print("\n🔧 测试评估功能...")

    try:
        # 测试数据
        test_questions = ["什么是机器学习？", "深度学习的特点是什么？"]
        test_answers = ["机器学习是人工智能的一个分支。", "深度学习使用多层神经网络。"]
        test_contexts = [
            ["机器学习是人工智能的一个分支，通过数据学习模式。"],
            ["深度学习是机器学习的一个子集，使用多层神经网络进行特征提取。"]
        ]
        test_references = ["机器学习是AI的一个分支。", "深度学习使用神经网络。"]

        # 测试RAGAS评估器
        print("测试RAGAS评估器...")
        ragas_evaluator = create_evaluator("manual")  # 使用手动评估器作为回退
        ragas_results = ragas_evaluator.evaluate_simple(
            questions=test_questions,
            answers=test_answers,
            contexts=test_contexts,
            references=test_references
        )
        print(f"✅ RAGAS评估完成 - 平均分数: {ragas_results.get('avg_manual_score', 0):.4f}")

        # 测试混合评估器
        print("测试混合评估器...")
        hybrid_evaluator = create_evaluator("hybrid")
        hybrid_results = hybrid_evaluator.evaluate_with_hybrid_precision(
            questions=test_questions,
            answers=test_answers,
            contexts=test_contexts,
            references=test_references,
            fusion_method="weighted_rrf"
        )
        print(f"✅ 混合评估完成 - 平均混合分数: {hybrid_results.get('avg_hybrid_score', 0):.4f}")

        print("✅ 评估功能测试通过")
        return True

    except Exception as e:
        print(f"❌ 评估功能测试失败: {e}")
        return False

def test_experiment_runner():
    """测试实验运行器"""
    print("\n🔧 测试实验运行器...")

    try:
        # 创建实验运行器
        runner = ExperimentRunner("baseline")
        print(f"✅ 实验运行器创建成功 - 类型: {runner.experiment_type}")

        # 测试设置
        if runner.test_setup():
            print("✅ 实验设置测试通过")
        else:
            print("⚠️  实验设置测试失败（可能缺少API配置）")

        # 测试单个实验
        print("测试单个实验...")
        test_result = runner.run_single_experiment(
            question="什么是机器学习？",
            contexts=["机器学习是人工智能的一个分支，通过数据学习模式。"],
            reference="机器学习是AI的一个分支。"
        )
        print(f"✅ 单个实验完成 - 评估器: {test_result['evaluation']['evaluator_type']}")

        print("✅ 实验运行器测试通过")
        return True

    except Exception as e:
        print(f"❌ 实验运行器测试失败: {e}")
        return False

def run_full_integration_test():
    """运行完整的集成测试"""
    print("🚀 运行完整的集成测试...")

    tests = [
        ("基础功能", test_basic_functionality),
        ("评估功能", test_evaluation_functionality),
        ("实验运行器", test_experiment_runner)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"运行测试: {test_name}")
        print('='*50)

        if test_func():
            passed += 1
            print(f"✅ {test_name} 测试通过")
        else:
            print(f"❌ {test_name} 测试失败")

    print(f"\n{'='*50}")
    print(f"集成测试完成: {passed}/{total} 测试通过")
    print('='*50)

    return passed == total

def check_and_build_knowledge_base():
    """检查并构建知识库"""
    print("🔍 检查知识库状态...")

    builder = SimpleKnowledgeBuilder()
    kb_info = builder.get_knowledge_info()

    if kb_info['status'] == 'ready':
        print(f"✅ 知识库已就绪，文档数量: {kb_info['total_documents']}")
        return True

    print("⚠️  知识库为空，需要构建")
    print("\n💡 您可以选择：")
    print("   1. 运行知识库设置向导")
    print("   2. 使用默认的HotPotQA数据快速构建")
    print("   3. 跳过知识库构建（使用提供的上下文）")

    while True:
        choice = input("\n请选择（1-3）[默认:2]：").strip()
        if not choice:
            choice = "2"

        if choice == "1":
            return run_simple_setup()
        elif choice == "2":
            print("\n⚡ 快速构建知识库...")
            # 查找默认数据文件
            possible_paths = [
                "dataset/hotpot_medium_batch_1.json",
                "../dataset/hotpot_medium_batch_1.json",
                "../../dataset/hotpot_medium_batch_1.json"
            ]

            dataset_file = None
            for path in possible_paths:
                if Path(path).exists():
                    dataset_file = path
                    break

            if dataset_file:
                return builder.build_from_hotpotqa(dataset_file)
            else:
                print("❌ 未找到默认数据文件")
                return False
        elif choice == "3":
            print("⏭️  跳过知识库构建")
            return True
        else:
            print("❌ 请输入有效的选项（1-3）")

def run_batch_experiment_cli(batch_id: int, experiment_type: str):
    """运行批次实验（命令行接口）"""
    print(f"🚀 运行批次实验 - 批次{batch_id} ({experiment_type})")

    try:
        # 检查并构建知识库
        if not check_and_build_knowledge_base():
            print("⚠️  知识库构建失败，但仍可继续实验（使用提供的上下文）")

        # 创建批次管理器
        manager = BatchExperimentManager(batch_id, experiment_type)

        # 加载之前的进度
        if manager.load_previous_progress():
            if manager.batch_stats.status == 'completed':
                print(f"✅ 批次 {batch_id} 已完成，跳过处理")
                return

        # 加载数据
        samples = manager.load_batch_dataset()
        if not samples:
            print(f"❌ 无法加载批次 {batch_id} 的数据")
            return

        # 创建实验运行器
        runner = ExperimentRunner(experiment_type)

        # 运行实验
        print(f"开始处理批次 {batch_id} - 样本数: {len(samples)}")

        for i, sample in enumerate(samples[manager.current_sample_idx:], manager.current_sample_idx):
            try:
                # 适配HotPotQA数据格式
                question = sample.get('question', '')

                # 使用answer作为reference
                reference = sample.get('answer', '')

                # 处理HotPotQA的context格式 - 转换为字符串列表
                hotpot_context = sample.get('context', [])
                contexts = []

                if isinstance(hotpot_context, list):
                    for ctx_item in hotpot_context:
                        if isinstance(ctx_item, list) and len(ctx_item) >= 2:
                            # ctx_item格式: [标题, 文本列表]
                            title, text_list = ctx_item[0], ctx_item[1]
                            if isinstance(text_list, list):
                                # 将文本段落合并
                                full_text = ' '.join(text_list)
                                contexts.append(full_text)
                            else:
                                contexts.append(str(text_list))
                        else:
                            contexts.append(str(ctx_item))

                if not all([question, contexts, reference]):
                    manager.add_error_result(i, f"缺少必要数据字段 - question: {bool(question)}, contexts: {bool(contexts)}, reference: {bool(reference)}")
                    continue

                # 运行实验
                result = runner.run_single_experiment(question, contexts, reference)

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
                print(f"❌ 处理样本 {i} 失败: {e}")
                manager.add_error_result(i, str(e))

        # 计算汇总统计
        summary_stats = runner._calculate_summary_stats(manager.results)

        # 完成批次
        final_results = manager.finalize_batch(summary_stats)

        print(f"✅ 批次 {batch_id} 实验完成")
        return final_results

    except Exception as e:
        print(f"❌ 批次实验运行失败: {e}")
        return None

def run_comparison_experiment():
    """运行对比实验"""
    print("🚀 运行对比实验 - 基线 vs 混合检索")

    # 测试数据
    test_questions = [
        "什么是机器学习？",
        "深度学习和机器学习有什么区别？",
        "自然语言处理的主要应用有哪些？"
    ]

    test_contexts = [
        ["机器学习是人工智能的一个分支，通过数据学习模式。"],
        ["深度学习是机器学习的一个子集，使用多层神经网络。", "机器学习包括多种算法和技术。"],
        ["自然语言处理用于文本分析、机器翻译等应用。", "NLP技术包括分词、命名实体识别等。"]
    ]

    test_references = [
        "机器学习是AI的分支，通过数据学习模式。",
        "深度学习使用神经网络，是机器学习的子集。",
        "自然语言处理用于文本分析和机器翻译。"
    ]

    experiment_types = ["baseline", "hybrid_standard"]
    all_results = {}

    for exp_type in experiment_types:
        print(f"\n{'='*60}")
        print(f"运行实验类型: {exp_type}")
        print('='*60)

        try:
            runner = ExperimentRunner(exp_type)
            results = runner.run_batch_experiment(
                questions=test_questions,
                contexts_list=test_contexts,
                references=test_references
            )

            # 计算平均指标
            summary = runner._calculate_summary_stats(results)
            all_results[exp_type] = summary

            print(f"✅ {exp_type} 实验完成")
            print(f"   总样本数: {summary.get('total_samples', 0)}")
            print(f"   成功率: {summary.get('success_rate', 0):.2%}")

            # 显示关键指标
            if exp_type == "baseline":
                key_metrics = ['context_precision', 'faithfulness', 'answer_relevancy']
            else:
                key_metrics = ['hybrid_context_precision', 'avg_hybrid_score']

            for metric in key_metrics:
                if metric in summary:
                    print(f"   {metric}: {summary[metric].get('mean', 0):.4f}")

        except Exception as e:
            print(f"❌ {exp_type} 实验失败: {e}")
            all_results[exp_type] = {'error': str(e)}

    # 对比结果
    print(f"\n{'='*60}")
    print("对比实验结果总结")
    print('='*60)

    for exp_type, results in all_results.items():
        if 'error' in results:
            print(f"{exp_type}: 实验失败 - {results['error']}")
        else:
            print(f"{exp_type}: 成功率 {results.get('success_rate', 0):.2%}")

    return all_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="混合检索评估系统")
    parser.add_argument("--mode", choices=["test", "batch", "compare", "integration"],
                       default="test", help="运行模式")
    parser.add_argument("--batch-id", type=int, default=1, help="批次ID")
    parser.add_argument("--experiment-type", choices=["baseline", "hybrid_standard", "hybrid_precision"],
                       default="baseline", help="实验类型")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="日志级别")

    args = parser.parse_args()

    # 设置日志级别
    setup_logging(level=args.log_level)

    # 重新加载配置（如果指定了配置文件）
    if args.config:
        reload_config(args.config)

    print("🚀 混合检索评估系统")
    print(f"模式: {args.mode}")
    print(f"实验类型: {args.experiment_type}")
    print('='*60)

    if args.mode == "test":
        # 基础测试
        test_basic_functionality()
        test_evaluation_functionality()

    elif args.mode == "integration":
        # 完整集成测试
        success = run_full_integration_test()
        if success:
            print("\n🎉 所有集成测试通过！")
        else:
            print("\n❌ 部分集成测试失败")
            sys.exit(1)

    elif args.mode == "batch":
        # 运行批次实验
        result = run_batch_experiment_cli(args.batch_id, args.experiment_type)
        if result:
            print("\n🎉 批次实验成功完成")
        else:
            print("\n❌ 批次实验失败")
            sys.exit(1)

    elif args.mode == "compare":
        # 运行对比实验
        results = run_comparison_experiment()
        print("\n🎉 对比实验完成")

    else:
        print(f"❌ 未知的运行模式: {args.mode}")
        sys.exit(1)

    print("\n✅ 程序执行完成")

if __name__ == "__main__":
    main()