"""
实验运行与评估功能包。

该包封装了混合检索实验运行过程中使用的核心组件，包括配置管理、
检索器、评估器、响应生成器以及知识库构建工具。通过显式导出关键
接口，方便在脚本与测试中复用相同的实验流水线。
"""

from .core.config import Config, get_config, reload_config
from .core.utils import setup_logging, format_time
from .experiment import BatchExperimentManager, ExperimentRunner
from .evaluators import create_evaluator
from .retrievers import create_retriever
from .generators import LLMClient, ResponseGenerator
from .knowledge_base import SimpleKnowledgeBuilder, run_simple_setup

__all__ = [
    "Config",
    "get_config",
    "reload_config",
    "setup_logging",
    "format_time",
    "BatchExperimentManager",
    "ExperimentRunner",
    "create_evaluator",
    "create_retriever",
    "LLMClient",
    "ResponseGenerator",
    "SimpleKnowledgeBuilder",
    "run_simple_setup",
]
