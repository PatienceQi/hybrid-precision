#!/usr/bin/env python3
"""
混合检索评估基础使用示例

本示例展示如何使用Hybrid Precision指标评估混合检索系统
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.hybrid_retrieval import HybridPrecisionEvaluator

def main():
    # 创建评估器
    evaluator = HybridPrecisionEvaluator()

    # 模拟检索结果分数
    # 稠密检索器分数（如向量检索）
    dense_scores = np.array([0.85, 0.72, 0.68, 0.91, 0.55])

    # 稀疏检索器分数（如BM25）
    sparse_scores = np.array([0.78, 0.65, 0.82, 0.73, 0.69])

    # 查询文本
    query = "什么是混合检索系统"

    # 执行评估
    results = evaluator.evaluate(dense_scores, sparse_scores, [query])

    # 输出结果
    print("=== 混合检索评估结果 ===")
    print(f"Hybrid Precision: {results['hybrid_precision']:.4f}")
    print(f"信息熵置信度: {results['entropy_confidence']:.4f}")
    print(f"互信息置信度: {results['mutual_information_confidence']:.4f}")
    print(f"统计显著性: {results['statistical_confidence']:.4f}")
    print(f"自适应权重 - 稠密: {results['adaptive_weights']['dense']:.3f}, 稀疏: {results['adaptive_weights']['sparse']:.3f}")
    print(f"不确定性惩罚: {results['uncertainty_penalty']:.4f}")

if __name__ == "__main__":
    main()