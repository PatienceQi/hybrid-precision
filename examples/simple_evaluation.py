#!/usr/bin/env python3
"""
简单评估示例 - 对比单一检索与混合检索

本示例展示如何使用RAGAS扩展评估单一检索和混合检索的性能差异
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.hybrid_retrieval import RAGASHybridExtension

def main():
    # 创建RAGAS扩展评估器
    extension = RAGASHybridExtension()

    # 模拟评估数据
    query = "人工智能的发展历程"
    contexts = ["人工智能是计算机科学的一个分支", "机器学习是AI的重要技术", "深度学习推动了AI的快速发展"]
    generated_answer = "人工智能经历了符号主义、连接主义和行为主义三个发展阶段"
    reference_answer = "人工智能发展包括符号主义、连接主义、行为主义等阶段"

    # 模拟检索器分数
    dense_scores = [0.85, 0.72, 0.68]
    sparse_scores = [0.78, 0.65, 0.82]

    # 执行混合检索评估
    results = extension.evaluate_hybrid_retrieval(
        query=query,
        retrieved_contexts=contexts,
        generated_answer=generated_answer,
        reference_answer=reference_answer,
        dense_scores=dense_scores,
        sparse_scores=sparse_scores
    )

    # 输出评估报告
    report = extension.generate_hybrid_report(results)
    print(report)

    # 获取优化建议
    recommendations = extension.get_recommendations(results)
    print("\n=== 优化建议 ===")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    # 对比单一检索与混合检索
    print("\n=== 性能对比 ===")
    dense_results = {'context_precision': 0.75}
    sparse_results = {'context_precision': 0.68}
    hybrid_results = {'hybrid_precision': 0.82, 'hybrid_confidence': 0.78}

    comparison = extension.compare_hybrid_vs_standard(
        dense_results, sparse_results, hybrid_results
    )

    print(f"相比稠密检索提升: {comparison['improvement_vs_dense']:.1f}%")
    print(f"相比稀疏检索提升: {comparison['improvement_vs_sparse']:.1f}%")
    print(f"混合检索置信度: {comparison['hybrid_confidence_score']:.3f}")

if __name__ == "__main__":
    main()