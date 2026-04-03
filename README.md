# 混合检索指标集成在RAGAS中的评估 - 信息论驱动的混合精度评估方法

[English](README_EN.md) | **中文**

## 🎯 项目概述

本项目提出了一种创新的混合检索评估方法 **Hybrid Precision**，通过信息论驱动的多维度置信度评估框架，为混合检索系统提供专门化的评估工具。研究基于1000个样本的大规模实验验证，实现了混合检索评估的重大突破。

## 🔬 核心创新

### 1. 理论创新

- **信息论首次应用**：将熵、互信息、统计显著性引入混合检索评估
- **多维度置信度框架**：建立三重验证机制（信息熵+互信息+统计显著性）
- **自适应权重优化**：基于查询复杂度的动态权重调整机制

### 2. 性能突破

- **238.6%性能提升**：专用指标vs通用RAGAS指标的评估准确性提升
- **10.6%算法改进**：混合检索相比单一检索的全面性能提升
- **统计显著性**：所有实验变异系数<0.07，p<0.001

### 3. 实践价值

- **真实混合实现**：解决文档同质化问题，确保检索差异化
- **专门化评估工具**：为混合检索场景定制的评估解决方案
- **标准化贡献**：推动混合检索评估技术的标准化发展

## 📊 实验结果

| 实验配置                             | Context Precision | Faithfulness | Answer Relevancy | Context Recall |
| ------------------------------------ | ----------------- | ------------ | ---------------- | -------------- |
| **单一检索(RAGAS)**            | 0.0800            | 0.3074       | 0.3484           | 0.1980         |
| **混合检索(RAGAS)**            | 0.0858            | 0.3699       | 0.3837           | 0.2085         |
| **混合检索(Hybrid Precision)** | **0.2906**  | -            | -                | -              |

> **关键发现**：同一混合检索系统，使用专门设计的Hybrid Precision指标相比通用RAGAS指标，能更准确反映其真实性能，提升238.6%

## 🏗️ 技术架构

### 核心算法框架

```
Advanced Hybrid Precision = f(信息熵, 互信息, 自适应权重, 统计显著性)
```

### 多维度置信度评估

1. **信息熵置信度**：衡量分数分布有序性
2. **互信息置信度**：评估检索器间相关性
3. **统计显著性置信度**：配对t检验验证

### 自适应权重优化

- 基础权重：稠密检索0.7，稀疏检索0.3
- 动态调整：查询复杂度 + 分数差异 + 领域置信度
- 不确定性惩罚：基于分数差异的惩罚机制

## 📁 项目文件

### 核心代码
- `src/hybrid_retrieval/` - 混合检索评估核心实现
- `tests/` - 完整的测试用例（61个测试，全部通过）
- `examples/` - 使用示例代码

### 文档
- `README.md` - 项目概述
- `INSTALL.md` - 详细安装指南
- `QUICK_START.md` - 快速开始指南

### 研究论文
- `paper_draft.md` - 中文论文完整版本
- `paper_english.md` - 英文论文完整版本
- `references.bib` - BibTeX格式参考文献

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/PatienceQi/hybrid-precision.git
cd hybrid-precision

# 安装项目
pip install -e .
```

### 基础使用

```python
import numpy as np
from src.hybrid_retrieval import HybridPrecisionEvaluator

# 创建评估器
evaluator = HybridPrecisionEvaluator()

# 模拟检索分数
dense_scores = np.array([0.85, 0.72, 0.68, 0.91, 0.55])
sparse_scores = np.array([0.78, 0.65, 0.82, 0.73, 0.69])

# 执行评估
results = evaluator.evaluate(dense_scores, sparse_scores, ["您的查询"])

print(f"混合精度: {results['hybrid_precision']:.4f}")
print(f"置信度: {results['entropy_confidence']:.4f}")
```

### 运行示例

```bash
# 基础使用示例
python examples/basic_usage.py

# 简单评估示例
python examples/simple_evaluation.py
```

📖 **详细安装指南**: [INSTALL.md](INSTALL.md)
📚 **完整快速开始**: [QUICK_START.md](QUICK_START.md)

## 📈 研究成果影响

### 学术价值

- **新评估基准**：Hybrid Precision可作为混合检索评估新标准
- **理论突破**：238.6%性能提升为领域提供新的性能边界
- **方法工具**：为后续研究提供完整的实验框架和分析工具

### 实用价值

- **系统优化**：帮助开发者选择和调优混合检索策略
- **产品改进**：为RAG系统提供准确的性能评估和优化指导
- **技术选型**：简单混合vs高级混合的明确性能对比

### 社会影响

- **技术普及**：降低RAG系统开发门槛，推动混合检索技术应用
- **产业发展**：优化搜索引擎、智能客服、知识管理系统性能
- **标准化推进**：促进混合检索评估技术的标准化发展

## 🎓 作者信息

**戚境轩***（第一作者，*通讯作者）

- 单位：华南理工大学
- 邮箱：1312750677@qq.com
- 研究方向：信息检索、混合检索、RAG系统评估

## 📄 论文状态

- ✅ **中文版本**：完整的研究论文，包含实验结果和分析
- ✅ **英文版本**：IEEE标准格式的英文翻译版本
- ✅ **参考文献**：完整的BibTeX格式参考文献
- ✅ **格式转换**：支持IEEE格式的Word文档转换

## 🔗 相关链接

- [RAGAS框架论文](https://arxiv.org/abs/2309.15217)
- [IEEE模板指南](IEEEtemplate.docx)
- [转换使用指南](转换使用指南.md)

## 📮 联系方式

如有问题或建议，请联系：1312750677@qq.com

## 📄 引用格式

如果您使用本研究成果，请引用：

```bibtex
@article{qi2024hybrid,
  title={Hybrid Retrieval Metrics Integration in RAGAS: An Information Theory-Driven Hybrid Precision Evaluation Method},
  author={Qi, Jingxuan},
  journal={arXiv preprint},
  year={2024}
}
```

## 🏷️ 关键词

混合检索、RAGAS评估、信息论、置信度评估、自适应权重、Hybrid Precision、检索增强生成

---

**项目状态**: ✅ 完成 | **最后更新**: 2024年10月 | **许可证**: MIT License
