"""
API客户端基类和工厂模式
提供统一的API客户端接口，支持多种LLM服务
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from openai import OpenAI
import openai

from .config import get_config

logger = logging.getLogger(__name__)

class BaseAPIClient(ABC):
    """API客户端基类"""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.config = get_config()
        self._setup_client()

    @abstractmethod
    def _setup_client(self):
        """设置API客户端"""
        pass

    @abstractmethod
    def generate_answer(self, question: str, contexts: List[Dict], **kwargs) -> str:
        """生成回答"""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """测试API连接"""
        pass

    def _build_prompt(self, question: str, contexts: List[Dict]) -> str:
        """构建提示词"""
        context_text = "\n\n".join([
            f"文档{i+1}: {ctx.get('title', f'Doc_{i+1}')}\n{ctx.get('text', ctx) if isinstance(ctx, dict) else ctx}"
            for i, ctx in enumerate(contexts)
        ])

        return f"""基于以下检索到的文档，请回答问题：

问题：{question}

检索到的文档：
{context_text}

请根据上述文档内容，提供一个准确、简洁的回答。如果文档中没有足够的信息来回答问题，请明确说明。"""

    def _handle_api_error(self, error: Exception, attempt: int, max_retries: int) -> bool:
        """处理API错误"""
        logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {error}")

        if attempt < max_retries - 1:
            retry_delay = self.config.api.retry_delay * (2 ** attempt)  # 指数退避
            logger.info(f"等待 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
            return True
        else:
            logger.error(f"API调用最终失败，已重试 {max_retries} 次")
            return False

class OpenAICompatibleClient(BaseAPIClient):
    """OpenAI兼容的API客户端"""

    def _setup_client(self):
        """设置OpenAI兼容客户端"""
        if not self.config.api.api_key:
            raise ValueError("API密钥未配置")

        try:
            self.client = OpenAI(
                api_key=self.config.api.api_key,
                base_url=self.config.api.base_url,
                default_headers=self.config.api.default_headers,
                timeout=self.config.api.timeout
            )
            logger.info(f"✅ 创建OpenAI兼容客户端 - 基础URL: {self.config.api.base_url}")
        except Exception as e:
            logger.error(f"创建API客户端失败: {e}")
            raise

    def generate_answer(self, question: str, contexts: List[Dict], **kwargs) -> str:
        """生成回答"""
        max_retries = kwargs.get('max_retries', self.config.api.max_retries)
        prompt = self._build_prompt(question, contexts)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个基于检索文档回答问题的助手。请严格根据提供的文档内容回答问题，不要添加文档中没有的信息。"
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=kwargs.get('max_tokens', 150),
                    temperature=kwargs.get('temperature', 0.3)
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                if not self._handle_api_error(e, attempt, max_retries):
                    raise RuntimeError(f"API调用失败，无法生成回答: {e}")

        raise RuntimeError("API调用失败，已达到最大重试次数")

    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Connection test"}],
                max_tokens=5
            )
            logger.info("✅ API连接测试成功")
            return True
        except Exception as e:
            logger.error(f"API连接测试失败: {e}")
            return False

class MockAPIClient(BaseAPIClient):
    """模拟API客户端，用于测试"""

    def _setup_client(self):
        """设置模拟客户端"""
        logger.info("使用模拟API客户端")
        self.client = None

    def generate_answer(self, question: str, contexts: List[Dict], **kwargs) -> str:
        """生成模拟回答"""
        logger.info("使用模拟回答生成")

        # 简单的基于关键词的回答模拟
        all_text = " ".join([
            ctx.get('text', ctx) if isinstance(ctx, dict) else str(ctx)
            for ctx in contexts
        ])

        # 提取问题关键词
        question_lower = question.lower()
        key_words = [word for word in question_lower.split() if len(word) > 3]

        if not key_words:
            return "基于检索到的文档，我可以确认相关信息存在，但需要更具体的问题来获得准确答案。"

        # 简单的相关度检查
        sentences = all_text.split('.')
        relevant_sentences = []

        for sentence in sentences[:3]:  # 只检查前3句
            sentence_lower = sentence.lower()
            score = sum(1 for word in key_words if word in sentence_lower)
            if score >= 2:
                relevant_sentences.append(sentence.strip())

        if relevant_sentences:
            return ". ".join(relevant_sentences) + "."
        else:
            return "基于检索到的文档，相关信息与问题相关，但需要更具体的上下文来提供准确答案。"

    def test_connection(self) -> bool:
        """模拟连接测试"""
        logger.info("✅ 模拟API连接测试成功")
        return True

class APIClientFactory:
    """API客户端工厂类"""

    _clients = {
        'openai': OpenAICompatibleClient,
        'openrouter': OpenAICompatibleClient,
        'mock': MockAPIClient
    }

    @classmethod
    def create_client(cls, client_type: str = 'auto', model: str = "gpt-3.5-turbo") -> BaseAPIClient:
        """创建API客户端"""
        config = get_config()

        if client_type == 'auto':
            # 自动选择客户端类型
            if config.evaluation.use_mock_api or not config.api.api_key:
                client_type = 'mock'
            else:
                client_type = 'openai'

        if client_type not in cls._clients:
            raise ValueError(f"不支持的客户端类型: {client_type}")

        client_class = cls._clients[client_type]
        logger.info(f"创建API客户端: {client_type}")

        try:
            client = client_class(model=model)
            # 测试连接
            if not client.test_connection():
                if client_type != 'mock':
                    logger.warning("API连接测试失败，回退到模拟模式")
                    return cls.create_client('mock', model)
            return client
        except Exception as e:
            logger.error(f"创建API客户端失败: {e}")
            if client_type != 'mock':
                logger.info("回退到模拟模式")
                return cls.create_client('mock', model)
            raise

    @classmethod
    def register_client(cls, name: str, client_class: type):
        """注册新的客户端类型"""
        if not issubclass(client_class, BaseAPIClient):
            raise ValueError("客户端类必须继承自BaseAPIClient")
        cls._clients[name] = client_class
        logger.info(f"注册新的客户端类型: {name}")

# 向后兼容的函数
def setup_openai_api() -> BaseAPIClient:
    """设置OpenAI API（向后兼容）"""
    return APIClientFactory.create_client('auto')

def generate_answer_with_fixed_api(question: str, retrieved_docs: list, client: Optional[BaseAPIClient] = None) -> str:
    """使用修复版API生成回答（向后兼容）"""
    if client is None:
        client = APIClientFactory.create_client('auto')
    return client.generate_answer(question, retrieved_docs)

# 全局客户端实例（用于缓存）
_global_client = None

def get_global_client() -> BaseAPIClient:
    """获取全局API客户端实例"""
    global _global_client
    if _global_client is None:
        _global_client = APIClientFactory.create_client('auto')
    return _global_client

def reset_global_client():
    """重置全局API客户端实例"""
    global _global_client
    _global_client = None