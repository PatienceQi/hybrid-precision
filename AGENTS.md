# 仓库指南

本文档为“混合检索指标集成在 RAGAS 中的评估”仓库提供贡献指南，专注于评估集成到 RAGAS 框架中的混合检索指标，用于 LLM 应用。

## 项目结构与模块组织

- **src/**: 包含指标实现和 RAGAS 集成的核心源代码。
- **tests/**: 指标和评估管道的单元和集成测试。
- **docs/**: 文档，包括 API 参考和评估报告。
- **examples/**: 演示 RAGAS 中混合检索使用的示例脚本。
- **requirements.txt**: 列出项目的 Python 依赖。

## 构建、测试和发展命令

- 安装依赖: `pip install -r requirements.txt`
- 运行测试: `pytest tests/ -v` (以详细输出运行所有测试)。
- 代码检查: `ruff check src/` (检查风格问题；使用 `ruff format src/` 自动修复)。
- 本地运行: `python src/evaluate_hybrid.py --data path/to/dataset.json` (评估示例数据)。

## 编码风格与命名规范

遵循 Python 代码的 PEP 8 标准。使用 4 空格缩进。变量名应使用 snake_case (例如，`hybrid_metric_score`)。函数和模块使用描述性名称，如 `compute_retrieval_recall`。所有函数使用类型提示。在提交前运行 `black src/` 进行格式化。

## 测试指南

使用 pytest 作为测试框架。新代码覆盖率目标 80% 以上 (使用 `pytest --cov=src` 检查)。测试文件名应为 `test_[unit_of_work]_[expected_behavior].py` (例如，`test_hybrid_retrieval.py`)。为每个指标包含单元测试，并为 RAGAS 管道包含集成测试。在提交 PR 前运行测试。

## 提交与拉取请求指南

- 使用约定式提交消息: `feat: 添加混合指标集成`，`fix: 解决评估 bug`。
- PR 必须包含: 清晰描述、链接 issue (例如 #123)、测试更新，以及通过检查的证据 (例如评估结果截图)。
- 确保无 linting 错误和完整测试覆盖。

## 代理特定指令

- 全局记忆: 在项目中维护一个全局记忆机制，用于跟踪评估过程中的关键指标和配置，支持连续的混合检索实验。
- 全局使用中文: 代理应全局使用中文进行对话和输出，确保所有响应和文档均为中文。
- 当在子目录中工作时，优先阅读此 AGENTS.md 和嵌套文件以获取范围规则。始终针对 RAGAS 兼容性验证变更。
