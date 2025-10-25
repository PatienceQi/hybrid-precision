# 混合检索评估系统 - 模块化重构版

## 🎯 项目概述

这是一个基于信息论的混合检索评估系统，支持RAGAS评估指标和自定义混合检索评估方法。系统采用模块化架构，提供统一的API接口和配置管理。

## 🏗️ 架构设计

### 核心模块结构

```
experiment_code/
├── core/                    # 核心模块
│   ├── config.py           # 统一配置管理
│   ├── api_client.py       # API客户端基类
│   ├── evaluator.py        # 评估器基类
│   └── utils.py            # 通用工具函数
├── evaluators/             # 评估器模块
│   ├── ragas_evaluator.py  # RAGAS评估器
│   ├── hybrid_evaluator.py # 混合检索评估器
│   ├── manual_evaluator.py # 手动评估器
│   └── evaluator_factory.py # 评估器工厂
├── retrievers/             # 检索器模块
│   ├── base_retriever.py   # 检索器基类
│   ├── embedding_retriever.py # 向量检索
│   ├── hybrid_retriever.py    # 混合检索
│   └── retriever_factory.py   # 检索器工厂
├── generators/             # 生成器模块
│   ├── response_generator.py  # 响应生成器
│   └── llm_client.py         # LLM客户端
├── experiment/             # 实验管理
│   ├── batch_manager.py    # 批次管理
│   └── experiment_runner.py # 实验运行器
├── cli/                    # 命令行辅助模块
│   ├── batch.py            # 批次实验 CLI
│   ├── comparison.py       # 对比实验 CLI
│   └── integration.py      # 集成测试 CLI
├── main.py                 # 精简命令行入口
└── config.json            # 配置文件
```

## 🚀 快速开始

### 1. 基础测试

```bash
# 运行基础功能测试
python -m experiment_code.main --mode test

# 运行完整集成测试
python -m experiment_code.main --mode integration
```

### 2. 运行实验

```bash
# 运行单个批次实验
python -m experiment_code.main --mode batch --batch-id 1 --experiment-type baseline

# 运行对比实验
python -m experiment_code.main --mode compare
```

### 3. 使用配置文件

```bash
# 使用自定义配置文件
python -m experiment_code.main --mode test --config my_config.json

# 设置日志级别
python -m experiment_code.main --mode test --log-level DEBUG
```

## 📋 核心功能

### 1. 统一配置管理
- 支持环境变量、配置文件、默认值三级配置
- 集中管理API密钥、模型参数、实验设置
- 提供配置验证和错误处理

### 2. 模块化评估器
- **RAGAS评估器**: 基于RAGAS框架的标准评估
- **混合评估器**: 基于信息论的混合检索评估
- **手动评估器**: 纯Python实现，无外部依赖

### 3. 智能检索器
- **向量检索器**: 基于嵌入向量的相似度检索
- **混合检索器**: 结合向量和关键词的混合检索
- 支持多种融合方法（加权求和、RRF、级联等）

### 4. 批次实验管理
- 支持1000样本分批次处理
- 提供断点续跑、进度监控、错误恢复
- 自动保存中间结果和最终统计

## 🔧 配置说明

### 环境变量
```bash
# API配置
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"
export LLM_MODEL="gpt-3.5-turbo"
export API_CLIENT_TYPE="auto"              # 可选: 强制使用 openai/openrouter/mock
export SKIP_API_CONNECTION_TEST="false"    # 可选: 离线环境下跳过连接检测
export FORCE_REAL_API="false"              # 可选: 禁止连接失败时回退到模拟API
export ALLOW_MOCK_FALLBACK="true"          # 可选: 允许自动切换至模拟API
export EMBEDDING_SERVICE_URL="https://wolfai.top/v1/embeddings"
export EMBEDDING_API_KEY="sk-7tk8aNrEJw3nmix9FeciFbgvvcr77hSwlpTaWKMH4FRwu84j"
export FORCE_EMBEDDING_SERVICE="false"     # 可选: 嵌入服务失败时立即报错
export EMBEDDING_FALLBACK_LOCAL="true"     # 可选: 允许使用本地模拟嵌入

# 实验配置
export BATCH_SIZE="200"
export USE_MOCK_API="false"
export LOG_LEVEL="INFO"
```

### 配置文件 (config.json)
```json
{
  "api": {
    "model": "gpt-3.5-turbo",
    "max_retries": 3,
    "retry_delay": 2,
    "timeout": 30
  },
  "evaluation": {
    "batch_size": 200,
    "use_mock_api": false,
    "metrics": ["context_precision", "faithfulness", "answer_relevancy", "context_recall"]
  },
  "retrieval": {
    "embedding_model": "text-embedding-3-large",
    "top_k": 5,
    "similarity_threshold": 0.7
  }
}
```

## 📊 评估指标

### 标准RAGAS指标
- **Context Precision**: 上下文精确度
- **Faithfulness**: 答案忠实度
- **Answer Relevancy**: 答案相关性
- **Context Recall**: 上下文召回率

### 混合检索专用指标
- **Hybrid Context Precision**: 混合上下文精确度
- **Information Entropy**: 信息熵
- **Mutual Information**: 互信息
- **Statistical Significance**: 统计显著性
- **Average Hybrid Score**: 平均混合分数

## 🛠️ API使用示例

### 基础使用
```python
from experiment_code import ExperimentRunner

# 创建实验运行器
runner = ExperimentRunner("hybrid_standard")

# 运行单个实验
result = runner.run_single_experiment(
    question="什么是机器学习？",
    contexts=["机器学习是人工智能的一个分支..."],
    reference="机器学习是AI的分支..."
)

print(f"评估结果: {result['evaluation']['metrics']}")
```

### 批量实验
```python
# 运行批量实验
results = runner.run_batch_experiment(
    questions=["问题1", "问题2", "问题3"],
    contexts_list=[["上下文1"], ["上下文2"], ["上下文3"]],
    references=["参考1", "参考2", "参考3"]
)
```

### 使用批次管理器
```python
from experiment_code import BatchExperimentManager

# 创建批次管理器
manager = BatchExperimentManager(batch_id=1, experiment_type="baseline")

# 加载进度
if manager.load_previous_progress():
    print("继续之前的实验...")

# 运行实验
# ... 处理样本 ...

# 完成批次
final_results = manager.finalize_batch(summary_stats)
```

## 🔍 高级功能

### 自定义评估器
```python
from experiment_code.core.evaluator import (
    BaseEvaluator,
    EvaluationResult,
)

class MyEvaluator(BaseEvaluator):
    def evaluate_single_sample(self, question, answer, contexts, reference):
        # 自定义评估逻辑
        metrics = {"my_metric": 0.8}
        return EvaluationResult(
            question=question,
            answer=answer,
            contexts=contexts,
            reference=reference,
            metrics=metrics,
            evaluator_type="my_evaluator"
        )
```

### 自定义检索器
```python
from experiment_code.retrievers.base_retriever import BaseRetriever

class MyRetriever(BaseRetriever):
    def retrieve(self, query, top_k=5):
        # 自定义检索逻辑
        documents = [...]  # 检索到的文档
        scores = [...]     # 相关性分数

        return RetrievalResult(
            query=query,
            documents=documents,
            scores=scores,
            retriever_type="my_retriever"
        )
```

## 📈 性能优化

### 1. 缓存机制
- 嵌入向量缓存
- API响应缓存
- 评估结果缓存

### 2. 批量处理
- 支持批量API调用
- 并行处理多个样本
- 内存使用优化

### 3. 错误处理
- 完善的异常处理
- 自动重试机制
- 错误恢复和日志记录

## 🧪 测试和验证

### 单元测试
```bash
# 运行所有测试
python -m pytest tests/

# 运行特定模块测试
python -m pytest tests/test_evaluators.py
```

### 集成测试
```bash
# 运行集成测试
python -m experiment_code.main --mode integration

# 运行性能测试
python tests/performance_test.py
```

## 📚 向后兼容

系统提供完整的向后兼容支持：
- 原有API函数仍然可用
- 配置文件格式兼容
- 实验结果格式一致

```python
# 向后兼容的使用方式
from experiment_code.evaluators.ragas_evaluator import test_fixed_ragas
from experiment_code.evaluators.hybrid_evaluator import test_hybrid_precision

# 运行原有测试
test_fixed_ragas()
test_hybrid_precision()
```

## 🔧 故障排除

### 常见问题

1. **API连接失败**
   - 检查API密钥配置
   - 验证网络连接
   - 尝试使用模拟模式

2. **嵌入服务不可用**
   - 确认远程嵌入服务可访问
   - 检查嵌入模型配置
   - 使用缓存的嵌入向量

3. **批次实验中断**
   - 使用批次管理器的断点续跑功能
   - 检查磁盘空间和权限
   - 查看详细日志信息

### 调试模式
```bash
# 启用调试模式
python -m experiment_code.main --mode test --log-level DEBUG

# 查看详细日志
tail -f batch_results/*.log
```

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
- 创建讨论

---

**🎉 混合检索评估系统 - 让评估更智能、更可靠！**
