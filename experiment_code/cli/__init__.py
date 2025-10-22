"""
命令行辅助模块。

该子包集中封装了命令行入口所需的高阶操作，包括批次实验执行、
对比实验以及集成测试，确保 `main.py` 保持精简并专注于参数解析。
"""

from .batch import run_batch_experiment, ensure_knowledge_base
from .comparison import run_comparison_experiment
from .integration import (
    run_full_integration_test,
    run_smoke_tests,
)

__all__ = [
    "run_batch_experiment",
    "ensure_knowledge_base",
    "run_comparison_experiment",
    "run_full_integration_test",
    "run_smoke_tests",
]
