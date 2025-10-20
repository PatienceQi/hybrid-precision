"""
Baseline Only 实验 (纯稠密检索 + 标准RAGAS评估)
专注于baseline方法，不包含混合检索或其他优化
支持批次处理，可处理1000样本分5批次
"""
import os
import json
import time
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import argparse

# 导入基础组件
from fixed_api_client import FixedAPIClient, generate_answer_with_fixed_api
from fully_fixed_ragas import ManualRagasEvaluator
from generator import retrieve_documents
from retriever import load_or_generate_embeddings
from batch_experiment_manager import BatchExperimentManager

def extract_ground_truth_from_supporting_facts(item: Dict, documents: List[Dict]) -> str:
    """基于supporting_facts提取ground truth"""
    supporting_facts = item.get('supporting_facts', [])
    if not supporting_facts:
        return extract_ground_truth_from_context(item)

    doc_map = {doc['title']: doc for doc in documents}
    relevant_texts = []

    for fact in supporting_facts:
        if isinstance(fact, list) and len(fact) >= 1:
            title = fact[0]
            if title in doc_map:
                relevant_texts.append(doc_map[title]['text'])

    return " ".join(relevant_texts) if relevant_texts else extract_ground_truth_from_context(item)

def extract_ground_truth_from_context(item: Dict) -> str:
    """后备方案提取ground truth"""
    ground_truth_texts = []
    for ctx_item in item.get('context', []):
        if isinstance(ctx_item, list) and len(ctx_item) > 1:
            if isinstance(ctx_item[1], list):
                ground_truth_texts.extend(ctx_item[1])
            elif isinstance(ctx_item[1], str):
                ground_truth_texts.append(ctx_item[1])
        elif isinstance(ctx_item, str):
            ground_truth_texts.append(ctx_item)
    return " ".join(ground_truth_texts)

def _log_progress(message: str, log_file: str = None):
    """记录进度日志"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)

    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + "\n")

def run_baseline_only_experiment(batch_id: Optional[int] = None):
    """运行纯baseline实验，支持批次处理"""

    # 如果没有指定批次ID，使用传统模式运行全部200样本
    if batch_id is None:
        batch_id = 1
        use_batch_manager = False
        print("🚀 开始运行 Baseline Only 实验（兼容模式）...")
    else:
        use_batch_manager = True
        print(f"🚀 开始运行 Baseline Only 实验 - 批次{batch_id}...")

    print("=" * 60)

    # 初始化批次管理器（如果使用批次模式）
    if use_batch_manager:
        batch_manager = BatchExperimentManager(batch_id=batch_id, experiment_type='baseline')

        # 检查是否已经处理过该批次
        if batch_manager.load_previous_progress():
            # 批次已完成，直接返回之前的结果
            if batch_manager.progress_file.exists():
                with open(batch_manager.final_results_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None

        # 加载批次数据集
        test_dataset = batch_manager.load_batch_dataset()
        if not test_dataset:
            print("❌ 无法加载批次数据集")
            return None

        print(f"📋 批次{batch_id}样本数: {len(test_dataset)}")
    else:
        # 兼容模式：使用原始数据集
        script_dir = Path(__file__).parent
        progress_log_file = script_dir / 'baseline_only_progress.log'

        _log_progress("=" * 60, progress_log_file)
        _log_progress("🚀 开始运行 Baseline Only 实验...", progress_log_file)
        _log_progress("=" * 60, progress_log_file)

        # 加载数据
        dataset_file = script_dir.parent / 'dataset' / 'hotpot_sample_200.json'
        with open(dataset_file, 'r', encoding='utf-8') as f:
            full_dataset = json.load(f)

        test_samples = 200  # 测试200个样本
        test_dataset = full_dataset[:test_samples]
        print(f"📋 测试样本数: {len(test_dataset)}")
        _log_progress(f"测试样本数: {len(test_dataset)}", progress_log_file)

    # 初始化组件
    print("🔧 初始化组件...")
    if use_batch_manager:
        batch_manager.log_message("初始化实验组件...")
    else:
        _log_progress("初始化组件...", progress_log_file)

    api_client = FixedAPIClient()
    ragas_evaluator = ManualRagasEvaluator(api_client)

    # 加载数据
    script_dir = Path(__file__).parent
    knowledge_base_dir = script_dir.parent / 'knowledge_base'
    embeddings = load_or_generate_embeddings(knowledge_base_dir)

    documents_file = knowledge_base_dir / 'documents.json'
    with open(documents_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # 实验结果存储
    baseline_results = []
    detailed_logs = []

    # 确定起始样本索引（如果恢复进度）
    start_idx = 0
    if use_batch_manager and batch_manager.current_sample_idx > 0:
        start_idx = batch_manager.current_sample_idx
        print(f"📊 从样本 {start_idx + 1} 继续处理...")

    for idx, item in enumerate(test_dataset[start_idx:], start_idx + 1):
        print(f"\n🔄 处理样本 {idx}/{len(test_dataset)}")

        if use_batch_manager:
            batch_manager.log_message(f"开始处理样本 {idx}/{len(test_dataset)}")
        else:
            _log_progress(f"开始处理样本 {idx}/{len(test_dataset)}: {item['question'][:60]}...", progress_log_file)

        log_entry = {
            'sample_id': idx,
            'question': item['question'][:100] + '...' if len(item['question']) > 100 else item['question']
        }

        try:
            question = item['question']

            if use_batch_manager:
                batch_manager.log_message(f"样本 {idx}: 提取ground truth...")
            else:
                _log_progress(f"样本 {idx}: 提取ground truth...", progress_log_file)

            reference_text = extract_ground_truth_from_supporting_facts(item, documents)

            if use_batch_manager:
                batch_manager.log_message(f"样本 {idx}: ground truth长度: {len(reference_text)} 字符")
            else:
                _log_progress(f"样本 {idx}: ground truth长度: {len(reference_text)} 字符", progress_log_file)

            # 纯Baseline实验：稠密检索 + 标准RAGAS
            if use_batch_manager:
                batch_manager.log_message(f"样本 {idx}: 开始Baseline稠密检索...")
            else:
                _log_progress(f"样本 {idx}: 开始Baseline稠密检索...", progress_log_file)

            print("   📊 Baseline: 稠密检索")
            baseline_docs = retrieve_documents(question, documents, embeddings)

            if use_batch_manager:
                batch_manager.log_message(f"样本 {idx}: 检索到 {len(baseline_docs)} 个文档")
            else:
                _log_progress(f"样本 {idx}: 检索到 {len(baseline_docs)} 个文档", progress_log_file)

            if use_batch_manager:
                batch_manager.log_message(f"样本 {idx}: 生成答案...")
            else:
                _log_progress(f"样本 {idx}: 生成答案...", progress_log_file)

            baseline_answer = generate_answer_with_fixed_api(question, baseline_docs, api_client)

            if use_batch_manager:
                batch_manager.log_message(f"样本 {idx}: 运行RAGAS评估...")
            else:
                _log_progress(f"样本 {idx}: 运行RAGAS评估...", progress_log_file)

            baseline_result = ragas_evaluator.evaluate_single_sample(
                question=question,
                answer=baseline_answer,
                contexts=[doc['text'] for doc in baseline_docs],
                reference=reference_text
            )

            if use_batch_manager:
                batch_manager.add_sample_result(idx - 1, baseline_result, log_entry)
            else:
                baseline_results.append(baseline_result)
                log_entry['baseline'] = baseline_result

                _log_progress(f"样本 {idx}: 评估完成 - Context Precision: {baseline_result.get('context_precision', 0):.4f}, "
                             f"Faithfulness: {baseline_result.get('faithfulness', 0):.4f}, "
                             f"Answer Relevancy: {baseline_result.get('answer_relevancy', 0):.4f}", progress_log_file)

            print(f"      ✅ Context Precision: {baseline_result.get('context_precision', 0):.4f}")
            print(f"      ✅ Faithfulness: {baseline_result.get('faithfulness', 0):.4f}")
            print(f"      ✅ Answer Relevancy: {baseline_result.get('answer_relevancy', 0):.4f}")

            if not use_batch_manager:
                detailed_logs.append(log_entry)

        except Exception as e:
            error_msg = f"样本 {idx} 处理失败: {e}"
            print(f"   ❌ {error_msg}")

            if use_batch_manager:
                batch_manager.add_error_result(idx - 1, str(e))
            else:
                _log_progress(error_msg, progress_log_file)

                error_result = {
                    'context_precision': 0.0,
                    'faithfulness': 0.0,
                    'answer_relevancy': 0.0,
                    'context_recall': 0.0,
                    'error': str(e)
                }
                baseline_results.append(error_result)

    # 计算平均结果
    def safe_mean(values, key, default=0.0):
        valid_values = [v.get(key, default) for v in values if isinstance(v, dict) and key in v and isinstance(v.get(key), (int, float))]
        return float(np.mean(valid_values)) if valid_values else default

    if use_batch_manager:
        # 批次模式：使用批次管理器的结果
        baseline_results = batch_manager.results

    summary = {
        'total_samples': len(test_dataset),
        'baseline_only': {
            'avg_context_precision': safe_mean(baseline_results, 'context_precision'),
            'avg_faithfulness': safe_mean(baseline_results, 'faithfulness'),
            'avg_answer_relevancy': safe_mean(baseline_results, 'answer_relevancy'),
            'avg_context_recall': safe_mean(baseline_results, 'context_recall')
        }
    }

    # 保存结果和日志
    if use_batch_manager:
        # 批次模式：通过批次管理器保存最终结果
        final_data = batch_manager.finalize_batch(summary)
        return final_data
    else:
        # 兼容模式：传统保存方式
        final_results = {
            'summary': summary,
            'baseline_results': baseline_results,
            'detailed_logs': detailed_logs,
            'experiment_config': {
                'test_samples': len(test_dataset),
                'top_k': 5,
                'api_service': 'OpenRouter',
                'llm_model': 'gpt-3.5-turbo',
                'experiment_type': 'baseline_only'
            }
        }

        # 保存结果
        results_file = script_dir / 'baseline_only_experiment_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        # 保存详细日志
        log_file = script_dir / 'baseline_only_experiment_log.json'
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_logs, f, indent=2, ensure_ascii=False)

        _log_progress(f"实验结果已保存到: {results_file}", progress_log_file)
        _log_progress(f"详细日志已保存到: {log_file}", progress_log_file)

        # 打印总结
        print("\n" + "=" * 60)
        print("🎯 Baseline Only 实验结果总结")
        print("=" * 60)

        print(f"\n📊 Baseline Only (纯稠密检索):")
        print(f"   平均Context Precision: {summary['baseline_only']['avg_context_precision']:.4f}")
        print(f"   平均Faithfulness: {summary['baseline_only']['avg_faithfulness']:.4f}")
        print(f"   平均Answer Relevancy: {summary['baseline_only']['avg_answer_relevancy']:.4f}")
        print(f"   平均Context Recall: {summary['baseline_only']['avg_context_recall']:.4f}")

        print(f"\n💾 详细结果已保存到: {results_file}")
        print(f"📋 详细日志已保存到: {log_file}")
        print(f"📝 实时进度日志已保存到: {progress_log_file}")

        _log_progress("🎯 Baseline Only 实验结果总结:", progress_log_file)
        _log_progress(f"   平均Context Precision: {summary['baseline_only']['avg_context_precision']:.4f}", progress_log_file)
        _log_progress(f"   平均Faithfulness: {summary['baseline_only']['avg_faithfulness']:.4f}", progress_log_file)
        _log_progress(f"   平均Answer Relevancy: {summary['baseline_only']['avg_answer_relevancy']:.4f}", progress_log_file)
        _log_progress(f"   平均Context Recall: {summary['baseline_only']['avg_context_recall']:.4f}", progress_log_file)
        _log_progress("=" * 60, progress_log_file)

        return final_results

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Baseline Only 实验 - 支持批次处理')
    parser.add_argument('--batch-id', type=int, choices=[1, 2, 3, 4, 5],
                       help='批次ID (1-5)，如果不指定则使用兼容模式运行全部200样本')
    parser.add_argument('--all-batches', action='store_true',
                       help='运行所有5个批次')

    args = parser.parse_args()

    if args.all_batches:
        # 运行所有5个批次
        print("🚀 开始运行所有5个Baseline批次实验...")
        all_results = []

        for batch_id in range(1, 6):
            print(f"\n{'='*60}")
            print(f"🔄 开始处理批次 {batch_id}/5")
            print(f"{'='*60}")

            try:
                batch_result = run_baseline_only_experiment(batch_id=batch_id)
                if batch_result:
                    all_results.append(batch_result)
                    print(f"✅ 批次 {batch_id} 完成")
                else:
                    print(f"⚠️  批次 {batch_id} 跳过（已处理过）")

                # 批次间短暂休息，避免API限流
                if batch_id < 5:
                    print("⏱️  批次间休息30秒...")
                    time.sleep(30)

            except Exception as e:
                print(f"❌ 批次 {batch_id} 处理失败: {e}")
                continue

        print(f"\n🎯 所有批次处理完成! 总计 {len(all_results)} 个批次成功")

    elif args.batch_id:
        # 运行指定批次
        results = run_baseline_only_experiment(batch_id=args.batch_id)
        print(f"\n✅ Baseline Only 批次 {args.batch_id} 实验运行完成")
        print("✅ 纯baseline方法评估完成")
    else:
        # 兼容模式：运行传统200样本实验
        results = run_baseline_only_experiment()
        print("\n✅ Baseline Only 实验运行完成")
        print("✅ 纯baseline方法评估完成")