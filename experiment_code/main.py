"""
混合检索评估系统命令行入口。

该模块负责解析命令行参数，并委托给 `experiment_code.cli` 中的
具体功能模块执行批次实验、对比实验或集成测试。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Optional

# 允许作为脚本直接运行，同时支持包内相对导入
if __package__ is None or __package__ == "":
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "experiment_code"

from . import reload_config, setup_logging  # noqa: E402
from .cli import (  # noqa: E402
    run_batch_experiment,
    run_comparison_experiment,
    run_full_integration_test,
    run_smoke_tests,
)


@dataclass
class CLIArgs:
    """命令行参数集合。"""

    mode: str
    batch_id: int
    experiment_type: str
    config: Optional[str]
    log_level: str


def parse_args(argv: Optional[list[str]] = None) -> CLIArgs:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="混合检索评估系统")
    parser.add_argument(
        "--mode",
        choices=["test", "batch", "compare", "integration"],
        default="test",
        help="运行模式",
    )
    parser.add_argument("--batch-id", type=int, default=1, help="批次ID")
    parser.add_argument(
        "--experiment-type",
        choices=["baseline", "hybrid_standard", "hybrid_precision"],
        default="baseline",
        help="实验类型",
    )
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别",
    )

    namespace = parser.parse_args(argv)
    return CLIArgs(
        mode=namespace.mode,
        batch_id=namespace.batch_id,
        experiment_type=namespace.experiment_type,
        config=namespace.config,
        log_level=namespace.log_level,
    )


def _handle_test_mode(_: CLIArgs) -> int:
    print("🚀 混合检索评估系统")
    print("模式: test")
    print("=" * 60)
    run_smoke_tests()
    return 0


def _handle_integration_mode(_: CLIArgs) -> int:
    print("🚀 混合检索评估系统")
    print("模式: integration")
    print("=" * 60)
    success = run_full_integration_test()
    if success:
        print("\n🎉 所有集成测试通过！")
        return 0

    print("\n❌ 部分集成测试失败")
    return 1


def _handle_batch_mode(args: CLIArgs) -> int:
    print("🚀 混合检索评估系统")
    print(f"模式: batch, 批次: {args.batch_id}, 实验类型: {args.experiment_type}")
    print("=" * 60)
    result = run_batch_experiment(args.batch_id, args.experiment_type)
    if result:
        print("\n🎉 批次实验成功完成")
        return 0

    print("\n❌ 批次实验失败")
    return 1


def _handle_compare_mode(_: CLIArgs) -> int:
    print("🚀 混合检索评估系统")
    print("模式: compare")
    print("=" * 60)
    run_comparison_experiment()
    print("\n🎉 对比实验完成")
    return 0


MODE_HANDLERS: Dict[str, Callable[[CLIArgs], int]] = {
    "test": _handle_test_mode,
    "integration": _handle_integration_mode,
    "batch": _handle_batch_mode,
    "compare": _handle_compare_mode,
}


def main(argv: Optional[list[str]] = None) -> int:
    """命令行入口。"""
    args = parse_args(argv)

    setup_logging(level=args.log_level)

    if args.config:
        reload_config(args.config)

    handler = MODE_HANDLERS.get(args.mode)
    if handler is None:
        print(f"❌ 未知的运行模式: {args.mode}")
        return 1

    exit_code = handler(args)
    print("\n✅ 程序执行完成")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
