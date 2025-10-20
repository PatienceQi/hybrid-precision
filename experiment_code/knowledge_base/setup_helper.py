"""
交互式设置向导 - 用户友好的知识库设置界面
让用户通过简单的问答完成知识库配置
"""

import os
import sys
import time
from typing import List, Dict, Optional
from pathlib import Path

from core.utils import setup_logging
from .simple_builder import SimpleKnowledgeBuilder

logger = setup_logging(level="INFO")

class KnowledgeSetupWizard:
    """知识库设置向导 - 交互式配置界面"""

    def __init__(self):
        """初始化设置向导"""
        self.builder = SimpleKnowledgeBuilder()
        self.config = {}

    def run_setup(self) -> bool:
        """
        运行完整的设置流程

        Returns:
            是否成功完成设置
        """
        print("🎯 知识库设置向导")
        print("=" * 50)
        print("让我们用2分钟时间设置您的知识库！")
        print()

        try:
            # 步骤1：选择数据源
            source_choice = self._choose_data_source()
            if not source_choice:
                return False

            # 步骤2：根据选择进行相应设置
            success = False
            if source_choice == "1":
                success = self._setup_from_hotpotqa()
            elif source_choice == "2":
                success = self._setup_from_wikipedia()
            elif source_choice == "3":
                success = self._setup_from_file()
            else:
                print("❌ 无效的选择")
                return False

            if success:
                self._show_completion_message()
                return True
            else:
                print("❌ 设置过程中出现错误")
                return False

        except KeyboardInterrupt:
            print("\n\n⚠️  用户取消了设置")
            return False
        except Exception as e:
            print(f"\n❌ 设置失败: {e}")
            return False

    def _choose_data_source(self) -> Optional[str]:
        """
        让用户选择数据源

        Returns:
            用户选择或None
        """
        print("📋 步骤1: 选择知识库数据源")
        print("-" * 30)
        print("""
请选择您的知识库来源：

1️⃣  使用当前实验数据（推荐）
   ✓ 最快，30秒完成
   ✓ 使用现有的HotPotQA数据集
   ✓ 立即可用，无需网络

2️⃣  维基百科主题词
   ✓ 输入主题词自动获取
   ✓ 内容丰富，覆盖面广
   ✓ 需要网络连接

3️⃣  本地文本文件
   ✓ 上传您自己的文档
   ✓ 完全定制化
   ✓ 支持.txt和.md文件
""")

        while True:
            choice = input("\n请输入选项（1-3）[默认:1]：").strip()
            if not choice:
                choice = "1"

            if choice in ["1", "2", "3"]:
                return choice
            else:
                print("❌ 请输入有效的选项（1-3）")

    def _setup_from_hotpotqa(self) -> bool:
        """从HotPotQA数据设置"""
        print("\n📊 您选择了：使用当前实验数据")
        print("-" * 40)

        # 查找HotPotQA数据文件
        possible_paths = [
            "dataset/hotpot_medium_batch_1.json",
            "../dataset/hotpot_medium_batch_1.json",
            "../../dataset/hotpot_medium_batch_1.json"
        ]

        dataset_file = None
        for path in possible_paths:
            if Path(path).exists():
                dataset_file = path
                break

        if not dataset_file:
            print("❌ 未找到HotPotQA数据文件")
            print("请确保数据文件在 dataset/ 目录下")
            return False

        print(f"找到数据文件: {dataset_file}")
        print("正在构建知识库...")

        # 显示进度条效果
        self._show_progress_bar("构建中", 30)

        # 构建知识库
        success = self.builder.build_from_hotpotqa(dataset_file)

        if success:
            doc_count = self.builder.get_document_count()
            print(f"\n✅ 知识库构建成功！")
            print(f"📄 文档数量: {doc_count}")
            return True
        else:
            print("\n❌ 知识库构建失败")
            return False

    def _setup_from_wikipedia(self) -> bool:
        """从维基百科设置"""
        print("\n🌐 您选择了：维基百科主题词")
        print("-" * 40)

        # 检查维基百科库是否可用
        try:
            import wikipedia
        except ImportError:
            print("❌ 维基百科库未安装")
            print("请运行: pip install wikipedia")
            return False

        # 获取主题词
        topic = input("请输入主题词（如：人工智能、机器学习）：").strip()
        if not topic:
            print("❌ 主题词不能为空")
            return False

        print(f"正在搜索'{topic}'相关内容...")

        # 构建知识库
        success = self.builder.build_from_wikipedia([topic], max_pages=3)

        if success:
            doc_count = self.builder.get_document_count()
            print(f"\n✅ 维基百科知识库构建成功！")
            print(f"📄 文档数量: {doc_count}")
            print(f"🌐 数据来源: 维基百科")
            return True
        else:
            print("\n❌ 维基百科知识库构建失败")
            print("可能是网络问题或主题词无效")
            return False

    def _setup_from_file(self) -> bool:
        """从本地文件设置"""
        print("\n📁 您选择了：本地文本文件")
        print("-" * 40)

        file_path = input("请输入文本文件路径（支持.txt和.md）：").strip()
        if not file_path:
            print("❌ 文件路径不能为空")
            return False

        # 检查文件是否存在
        if not Path(file_path).exists():
            print(f"❌ 文件不存在: {file_path}")
            return False

        # 检查文件类型
        if not file_path.endswith(('.txt', '.md')):
            print("❌ 只支持.txt和.md文件")
            return False

        print(f"正在处理文件: {file_path}")

        # 构建知识库
        success = self.builder.build_from_file(file_path)

        if success:
            doc_count = self.builder.get_document_count()
            print(f"\n✅ 文件知识库构建成功！")
            print(f"📄 文档数量: {doc_count}")
            print(f"📁 源文件: {file_path}")
            return True
        else:
            print("\n❌ 文件知识库构建失败")
            return False

    def _show_progress_bar(self, message: str, duration: int):
        """显示简单的进度条"""
        print(f"{message}: ", end="", flush=True)
        for i in range(duration):
            print("█", end="", flush=True)
            time.sleep(0.1)
        print(" 100%")

    def _show_completion_message(self):
        """显示完成信息"""
        print("\n" + "=" * 50)
        print("🎉 知识库设置完成！")
        print("=" * 50)

        info = self.builder.get_knowledge_info()
        print(f"📊 知识库状态: {info['status']}")
        print(f"📄 文档总数: {info.get('total_documents', 0)}")
        print(f"📂 存储位置: {info.get('file_path', 'unknown')}")

        print("\n💡 现在您可以运行实验了：")
        print("   python main.py --mode batch --batch-id 1")

        print("\n📖 知识库已就绪，系统将自动使用它进行检索增强！")


def run_simple_setup() -> bool:
    """
    运行简单的知识库设置

    Returns:
        是否成功完成设置
    """
    wizard = KnowledgeSetupWizard()
    return wizard.run_setup()


# 快速测试函数
def test_knowledge_base_building():
    """测试知识库构建功能"""
    print("🔧 测试知识库构建功能...")

    builder = SimpleKnowledgeBuilder()

    # 测试数据
    test_docs = [
        {
            'id': 0,
            'title': '测试文档1',
            'content': '这是一个测试文档，用于验证知识库构建功能。',
            'source': 'test',
            'metadata': {'test': True}
        },
        {
            'id': 1,
            'title': '测试文档2',
            'content': '人工智能是计算机科学的一个分支，致力于创造能够执行通常需要人类智能的任务的系统。',
            'source': 'test',
            'metadata': {'test': True}
        }
    ]

    success = builder._save_knowledge_base(test_docs)

    if success:
        print("✅ 知识库构建功能测试通过")
        info = builder.get_knowledge_info()
        print(f"📊 测试知识库信息: {info}")
        return True
    else:
        print("❌ 知识库构建功能测试失败")
        return False


if __name__ == "__main__":
    # 如果直接运行此文件，启动设置向导
    print("🚀 启动知识库设置向导...")
    success = run_simple_setup()

    if success:
        print("\n🎉 设置完成！您可以开始使用知识库了。")
    else:
        print("\n❌ 设置未完成，请重试或检查错误信息。")