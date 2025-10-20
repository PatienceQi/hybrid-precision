"""
LLM客户端 - 统一的LLM服务接口
"""

import logging
from typing import Dict, List, Any, Optional
from core.api_client import BaseAPIClient, APIClientFactory
from core.config import get_config

logger = logging.getLogger(__name__)

class LLMClient:
    """LLM客户端 - 提供统一的LLM服务接口"""

    def __init__(self, api_client: Optional[BaseAPIClient] = None, model: str = "gpt-3.5-turbo"):
        self.api_client = api_client or APIClientFactory.create_client(model=model)
        self.model = model
        self.config = get_config()

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        生成响应

        Args:
            prompt: 输入提示
            **kwargs: 其他参数

        Returns:
            生成的响应文本
        """
        try:
            # 构建模拟上下文（用于兼容现有接口）
            mock_contexts = [{"title": "context", "text": prompt}]
            return self.api_client.generate_answer("生成响应", mock_contexts, **kwargs)

        except Exception as e:
            logger.error(f"响应生成失败: {e}")
            raise

    def generate_answer(self, question: str, contexts: List[Dict[str, Any]], **kwargs) -> str:
        """
        基于上下文生成答案

        Args:
            question: 问题
            contexts: 上下文文档列表
            **kwargs: 其他参数

        Returns:
            生成的答案
        """
        return self.api_client.generate_answer(question, contexts, **kwargs)

    def generate_with_system_prompt(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        使用系统提示生成响应

        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            **kwargs: 其他参数

        Returns:
            生成的响应
        """
        try:
            # 构建模拟上下文
            contexts = [
                {"title": "system", "text": system_prompt},
                {"title": "user", "text": user_prompt}
            ]
            return self.api_client.generate_answer("生成响应", contexts, **kwargs)

        except Exception as e:
            logger.error(f"系统提示生成失败: {e}")
            raise

    def test_connection(self) -> bool:
        """测试LLM连接"""
        return self.api_client.test_connection()

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model': self.model,
            'client_type': type(self.api_client).__name__,
            'config': {
                'max_retries': self.config.api.max_retries,
                'retry_delay': self.config.api.retry_delay,
                'timeout': self.config.api.timeout
            }
        }

# 向后兼容的函数
def create_llm_client(model: str = "gpt-3.5-turbo") -> LLMClient:
    """创建LLM客户端（向后兼容）"""
    return LLMClient(model=model)

def generate_with_llm(prompt: str, model: str = "gpt-3.5-turbo", **kwargs) -> str:
    """使用LLM生成响应（向后兼容）"""
    client = LLMClient(model=model)
    return client.generate_response(prompt, **kwargs)