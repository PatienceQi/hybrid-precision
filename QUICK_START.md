# 快速开始

本指南将帮助您在5分钟内开始使用混合检索评估工具。

## 步骤1：安装

```bash
# 克隆项目
git clone https://github.com/your-username/hybrid-retrieval-evaluation.git
cd hybrid-retrieval-evaluation

# 安装
pip install -e .
```

## 步骤2：基础评估

创建Python文件 `evaluate.py`：

```python
import numpy as np
from src.hybrid_retrieval import HybridPrecisionEvaluator

# 创建评估器
evaluator = HybridPrecisionEvaluator()

# 您的检索器分数
dense_scores = np.array([0.85, 0.72, 0.68, 0.91, 0.55])  # 稠密检索
sparse_scores = np.array([0.78, 0.65, 0.82, 0.73, 0.69])  # 稀疏检索

# 执行评估
results = evaluator.evaluate(dense_scores, sparse_scores, ["您的查询"])

# 查看结果
print(f"混合精度: {results['hybrid_precision']:.4f}")
print(f"置信度: {results['entropy_confidence']:.4f}")
```

## 步骤3：运行评估

```bash
python evaluate.py
```

## 步骤4：使用RAGAS扩展（可选）

```python
from src.hybrid_retrieval import RAGASHybridExtension

# 创建扩展评估器
extension = RAGASHybridExtension()

# 完整评估
results = extension.evaluate_hybrid_retrieval(
    query="您的查询",
    retrieved_contexts=["上下文1", "上下文2"],
    generated_answer="生成的答案",
    reference_answer="参考答案",
    dense_scores=[0.8, 0.7],
    sparse_scores=[0.75, 0.85]
)

# 生成报告
report = extension.generate_hybrid_report(results)
print(report)
```

## 结果解读

### 核心指标
- **Hybrid Precision**: 混合检索精度（0-1，越高越好）
- **Entropy Confidence**: 信息熵置信度（0-1，越高表示分数分布越有序）
- **Mutual Information**: 互信息置信度（0-1，越高表示检索器间一致性越好）
- **Statistical Confidence**: 统计显著性（0-1，越高表示差异越显著）

### 自适应权重
- **Dense Weight**: 稠密检索权重（0-1）
- **Sparse Weight**: 稀疏检索权重（0-1）

### 优化建议
运行后会自动提供基于评估结果的优化建议。

## 下一步

- 查看 [详细文档](README.md)
- 探索 [更多示例](examples/)
- 了解 [API参考](docs/)（即将推出）

## 获取帮助

遇到问题？请查看 [安装指南](INSTALL.md) 或在GitHub上创建Issue。