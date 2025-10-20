"""
简单知识库模块 - 用户友好的知识库构建工具
提供极简的API，让用户轻松构建和使用知识库
"""

from .simple_builder import SimpleKnowledgeBuilder
from .setup_helper import run_simple_setup, KnowledgeSetupWizard, test_knowledge_base_building

__all__ = ['SimpleKnowledgeBuilder', 'run_simple_setup', 'KnowledgeSetupWizard', 'test_knowledge_base_building']