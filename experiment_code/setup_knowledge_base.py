#!/usr/bin/env python3
"""
知识库设置入口 - 用户友好的知识库配置工具
提供简单的命令行界面，让用户轻松构建知识库
"""

import sys
import os
import argparse
from pathlib import Path

# 适配脚本/模块两种运行方式
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from knowledge_base import run_simple_setup, test_knowledge_base_building  # type: ignore
    from core.utils import setup_logging  # type: ignore
else:
    from .knowledge_base import run_simple_setup, test_knowledge_base_building
    from .core.utils import setup_logging

def print_welcome_message():
    """打印欢迎信息"""
    print("🎯 混合检索评估系统 - 知识库设置工具")
    print("=" * 60)
    print("欢迎使用知识库设置工具！")
    print("\n这个工具将帮助您在2分钟内完成知识库设置。")
    print("您可以选择以下数据源：")
    print("  📊 当前实验数据（最快）")
    print("  🌐 维基百科主题词（内容丰富）")
    print("  📁 本地文本文件（完全定制）")
    print("\n让我们开始吧！")
    print("=" * 60)

def print_completion_message(success: bool):
    """打印完成信息"""
    print("\n" + "=" * 60)
    if success:
        print("🎉 知识库设置完成！")
        print("\n💡 现在您可以运行实验了：")
        print("   python -m experiment_code.main --mode batch --batch-id 1")
        print("\n📚 知识库已就绪，系统将自动使用它进行检索增强！")
    else:
        print("❌ 知识库设置未完成")
        print("\n💡 您可以：")
        print("   1. 重新运行本工具")
        print("   2. 检查错误信息并解决问题")
        print("   3. 使用默认的实验数据模式")
    print("=" * 60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="知识库设置工具 - 轻松构建您的知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 交互式设置（推荐）
  python setup_knowledge_base.py

  # 测试知识库功能
  python setup_knowledge_base.py --test

  # 使用默认数据源快速设置
  python setup_knowledge_base.py --quick
        """
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="测试知识库构建功能"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="使用默认设置快速构建知识库"
    )

    parser.add_argument(
        "--source",
        choices=["hotpotqa", "wikipedia", "file"],
        help="指定数据源类型"
    )

    parser.add_argument(
        "--topic",
        type=str,
        help="维基百科主题词（与--source wikipedia一起使用）"
    )

    parser.add_argument(
        "--file",
        type=str,
        help="本地文件路径（与--source file一起使用）"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别"
    )

    args = parser.parse_args()

    # 设置日志级别
    setup_logging(level=args.log_level)

    try:
        if args.test:
            # 测试模式
            print("🔧 知识库功能测试模式")
            print("-" * 40)
            success = test_knowledge_base_building()

            if success:
                print("\n✅ 所有测试通过！知识库功能正常。")
                return 0
            else:
                print("\n❌ 测试失败，请检查错误信息。")
                return 1

        elif args.quick:
            # 快速模式 - 使用默认HotPotQA数据
            print("⚡ 快速知识库构建模式")
            print("-" * 40)
            print("正在使用默认的HotPotQA数据构建知识库...")

            from knowledge_base import SimpleKnowledgeBuilder
            builder = SimpleKnowledgeBuilder()

            # 查找默认数据文件
            dataset_file = None
            possible_paths = [
                "dataset/hotpot_medium_batch_1.json",
                "../dataset/hotpot_medium_batch_1.json",
                "../../dataset/hotpot_medium_batch_1.json"
            ]

            for path in possible_paths:
                if Path(path).exists():
                    dataset_file = path
                    break

            if dataset_file:
                success = builder.build_from_hotpotqa(dataset_file)
                if success:
                    print(f"\n✅ 快速构建成功！")
                    print(f"📄 文档数量: {builder.get_document_count()}")
                    return 0
                else:
                    print(f"\n❌ 快速构建失败")
                    return 1
            else:
                print("❌ 未找到默认数据文件，请使用交互式模式")
                return 1

        elif args.source:
            # 命令行指定模式
            print(f"🎯 指定数据源模式: {args.source}")
            print("-" * 40)

            from knowledge_base import SimpleKnowledgeBuilder
            builder = SimpleKnowledgeBuilder()
            success = False

            if args.source == "hotpotqa":
                # 查找数据文件
                dataset_file = None
                possible_paths = [
                    "dataset/hotpot_medium_batch_1.json",
                    "../dataset/hotpot_medium_batch_1.json",
                    "../../dataset/hotpot_medium_batch_1.json"
                ]

                for path in possible_paths:
                    if Path(path).exists():
                        dataset_file = path
                        break

                if dataset_file:
                    success = builder.build_from_hotpotqa(dataset_file)
                else:
                    print("❌ 未找到HotPotQA数据文件")
                    return 1

            elif args.source == "wikipedia":
                if not args.topic:
                    print("❌ 请提供主题词: --topic \"人工智能\"")
                    return 1
                success = builder.build_from_wikipedia([args.topic])

            elif args.source == "file":
                if not args.file:
                    print("❌ 请提供文件路径: --file \"document.txt\"")
                    return 1
                success = builder.build_from_file(args.file)

            if success:
                print(f"\n✅ 构建成功！")
                print(f"📄 文档数量: {builder.get_document_count()}")
                return 0
            else:
                print(f"\n❌ 构建失败")
                return 1

        else:
            # 交互式模式（默认）
            print_welcome_message()
            success = run_simple_setup()
            print_completion_message(success)

            return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消了操作")
        return 1
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
