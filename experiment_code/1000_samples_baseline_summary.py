"""
汇总1000样本baseline实验结果
"""
import json
import numpy as np
from pathlib import Path

def load_batch_results(batch_id):
    """加载单个批次的结果"""
    results_file = f"/Users/qipatience/Desktop/混合检索指标集成在 RAGAS 中的评估/experiment_code/batch_results/baseline_batch_{batch_id}_200_samples_results.json"
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_overall_stats(all_results):
    """计算总体统计信息"""
    all_context_precision = []
    all_faithfulness = []
    all_answer_relevancy = []
    all_context_recall = []

    for batch_results in all_results:
        for result in batch_results['results']:
            all_context_precision.append(result.get('context_precision', 0))
            all_faithfulness.append(result.get('faithfulness', 0))
            all_answer_relevancy.append(result.get('answer_relevancy', 0))
            all_context_recall.append(result.get('context_recall', 0))

    return {
        'total_samples': len(all_context_precision),
        'overall_stats': {
            'avg_context_precision': np.mean(all_context_precision),
            'std_context_precision': np.std(all_context_precision),
            'avg_faithfulness': np.mean(all_faithfulness),
            'std_faithfulness': np.std(all_faithfulness),
            'avg_answer_relevancy': np.mean(all_answer_relevancy),
            'std_answer_relevancy': np.std(all_answer_relevancy),
            'avg_context_recall': np.mean(all_context_recall),
            'std_context_recall': np.std(all_context_recall)
        },
        'batch_comparison': {
            f'batch_{i+1}': {
                'context_precision': np.mean([r.get('context_precision', 0) for r in all_results[i]['results']]),
                'faithfulness': np.mean([r.get('faithfulness', 0) for r in all_results[i]['results']]),
                'answer_relevancy': np.mean([r.get('answer_relevancy', 0) for r in all_results[i]['results']]),
                'context_recall': np.mean([r.get('context_recall', 0) for r in all_results[i]['results']])
            } for i in range(5)
        }
    }

def main():
    print("🎯 开始汇总1000样本baseline实验结果...")

    # 加载所有批次结果
    all_results = []
    for batch_id in range(1, 6):
        print(f"📊 加载批次{batch_id}结果...")
        batch_result = load_batch_results(batch_id)
        all_results.append(batch_result)
        print(f"   ✅ 批次{batch_id}: {len(batch_result['results'])}个样本")

    # 计算总体统计
    overall_stats = calculate_overall_stats(all_results)

    # 显示结果
    print("\n" + "="*60)
    print("🎯 1000样本baseline实验结果汇总")
    print("="*60)

    print(f"\n📊 总体统计 (总计{overall_stats['total_samples']}个样本):")
    stats = overall_stats['overall_stats']
    print(f"   平均Context Precision: {stats['avg_context_precision']:.4f} (±{stats['std_context_precision']:.4f})")
    print(f"   平均Faithfulness: {stats['avg_faithfulness']:.4f} (±{stats['std_faithfulness']:.4f})")
    print(f"   平均Answer Relevancy: {stats['avg_answer_relevancy']:.4f} (±{stats['std_answer_relevancy']:.4f})")
    print(f"   平均Context Recall: {stats['avg_context_recall']:.4f} (±{stats['std_context_recall']:.4f})")

    print(f"\n📈 批次对比:")
    for batch_name, batch_stats in overall_stats['batch_comparison'].items():
        print(f"   {batch_name}:")
        print(f"     Context Precision: {batch_stats['context_precision']:.4f}")
        print(f"     Faithfulness: {batch_stats['faithfulness']:.4f}")
        print(f"     Answer Relevancy: {batch_stats['answer_relevancy']:.4f}")
        print(f"     Context Recall: {batch_stats['context_recall']:.4f}")

    # 保存汇总结果
    summary_file = "/Users/qipatience/Desktop/混合检索指标集成在 RAGAS 中的评估/experiment_code/1000_samples_baseline_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, indent=2, ensure_ascii=False)

    print(f"\n💾 汇总结果已保存到: {summary_file}")
    print("\n✅ 1000样本baseline实验汇总完成！")

if __name__ == "__main__":
    main()