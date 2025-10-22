"""
批次实验命令行工具。

封装批次实验运行所需的流程，包括知识库检查、数据加载以及结果汇总，
便于在命令行或其他脚本中复用相同逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..experiment import BatchExperimentManager, ExperimentRunner
from ..knowledge_base import SimpleKnowledgeBuilder, run_simple_setup


def ensure_knowledge_base(builder: Optional[SimpleKnowledgeBuilder] = None) -> bool:
    """
    检查并在必要时构建知识库。

    Args:
        builder: 复用的 `SimpleKnowledgeBuilder` 实例，默认创建新的实例。

    Returns:
        True 表示知识库已就绪；False 表示构建失败。
    """
    print("🔍 检查知识库状态...")

    kb_builder = builder or SimpleKnowledgeBuilder()
    kb_info = kb_builder.get_knowledge_info()

    if kb_info["status"] == "ready":
        print(f"✅ 知识库已就绪，文档数量: {kb_info['total_documents']}")
        return True

    print("⚠️  知识库为空，需要构建")
    print("\n💡 您可以选择：")
    print("   1. 运行知识库设置向导")
    print("   2. 使用默认的HotPotQA数据快速构建")
    print("   3. 跳过知识库构建（使用提供的上下文）")

    while True:
        choice = input("\n请选择（1-3）[默认:2]：").strip() or "2"

        if choice == "1":
            return run_simple_setup()
        if choice == "2":
            print("\n⚡ 快速构建知识库...")
            possible_paths = [
                "dataset/hotpot_medium_batch_1.json",
                "../dataset/hotpot_medium_batch_1.json",
                "../../dataset/hotpot_medium_batch_1.json",
            ]

            dataset_file: Optional[str] = None
            for path in possible_paths:
                if Path(path).exists():
                    dataset_file = path
                    break

            if dataset_file:
                return kb_builder.build_from_hotpotqa(dataset_file)
            print("❌ 未找到默认数据文件")
            return False
        if choice == "3":
            print("⏭️  跳过知识库构建")
            return True

        print("❌ 请输入有效的选项（1-3）")


def _normalize_hotpot_context(raw_context: Any) -> List[str]:
    """
    HotPotQA 上下文数据转换。

    Args:
        raw_context: 原始 HotPotQA 上下文字段。

    Returns:
        转换后的纯文本列表。
    """
    if not isinstance(raw_context, list):
        return [str(raw_context)]

    normalized: List[str] = []
    for ctx_item in raw_context:
        if isinstance(ctx_item, list) and len(ctx_item) >= 2:
            title, text_list = ctx_item[0], ctx_item[1]
            if isinstance(text_list, list):
                normalized.append(" ".join(str(text) for text in text_list))
            else:
                normalized.append(str(text_list))
        else:
            normalized.append(str(ctx_item))
    return normalized


def run_batch_experiment(batch_id: int, experiment_type: str) -> Optional[Dict[str, Any]]:
    """
    运行批次实验。

    Args:
        batch_id: 批次编号。
        experiment_type: 实验类型（baseline/hybrid 等）。

    Returns:
        批次实验结果；若流程提前终止则返回 None。
    """
    print(f"🚀 运行批次实验 - 批次{batch_id} ({experiment_type})")

    try:
        if not ensure_knowledge_base():
            print("⚠️  知识库构建失败，但仍可继续实验（使用提供的上下文）")

        manager = BatchExperimentManager(batch_id, experiment_type)

        if manager.load_previous_progress():
            if manager.batch_stats.status == "completed":
                print(f"✅ 批次 {batch_id} 已完成，跳过处理")
                return manager.results

        samples = manager.load_batch_dataset()
        if not samples:
            print(f"❌ 无法加载批次 {batch_id} 的数据")
            return None

        runner = ExperimentRunner(experiment_type)
        print(f"开始处理批次 {batch_id} - 样本数: {len(samples)}")

        for i, sample in enumerate(samples[manager.current_sample_idx :], manager.current_sample_idx):
            try:
                question = sample.get("question", "")
                reference = sample.get("answer", "")
                contexts = _normalize_hotpot_context(sample.get("context", []))

                if not all([question, contexts, reference]):
                    manager.add_error_result(
                        i,
                        (
                            "缺少必要数据字段 - "
                            f"question: {bool(question)}, contexts: {bool(contexts)}, reference: {bool(reference)}"
                        ),
                    )
                    continue

                result = runner.run_single_experiment(question, contexts, reference)
                metrics = result["evaluation"].get("metrics", {})

                manager.add_sample_result(
                    i,
                    metrics,
                    {
                        "sample_id": i + 1,
                        "question": question,
                        "answer": result.get("answer", ""),
                        "contexts_count": len(contexts),
                        "metrics": metrics,
                        "error": result.get("error"),
                        "timestamp": result.get("timestamp"),
                    },
                )

                if (i + 1) % 10 == 0:
                    manager.print_progress()

            except Exception as err:  # noqa: BLE001 - 需捕获所有异常保证批次不中断
                print(f"❌ 处理样本 {i} 失败: {err}")
                manager.add_error_result(i, str(err))

        summary_stats = runner._calculate_summary_stats(manager.results)
        final_results = manager.finalize_batch(summary_stats)

        print(f"✅ 批次 {batch_id} 实验完成")
        return final_results

    except Exception as err:  # noqa: BLE001
        print(f"❌ 批次实验运行失败: {err}")
        return None
