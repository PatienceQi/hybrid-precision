"""
修复版API客户端
解决认证问题和RAGAS集成问题
"""
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class FixedAPIClient:
    """修复版API客户端，解决认证和兼容性问题"""

    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.client = None
        self._setup_client()

    def _setup_client(self):
        """设置API客户端，处理各种认证问题"""

        # 获取API密钥
        api_key = self._get_api_key()

        if not api_key:
            logger.warning("未找到有效的API密钥，使用模拟模式")
            self.use_mock = True
            return

        self.use_mock = False
        # 强制使用真实API，不降级到模拟模式

        # 获取环境变量中的基础URL
        base_url = os.getenv('OPENAI_API_BASE', 'https://openrouter.ai/api/v1')

        # 使用.env文件中的配置
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://ragas-hybrid-evaluation.com",
                "X-Title": "RAGAS Hybrid Evaluation"
            }
        )
        logger.info(f"✅ 使用.env配置创建API客户端 - 基础URL: {base_url}")

        # 测试连接
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Connection test"}],
                max_tokens=5
            )
            logger.info("✅ API连接测试成功")
        except Exception as e:
            logger.error(f"API连接测试失败: {e}")
            raise

        # 配置已创建完成，无需循环测试

    def _get_api_key(self):
        """获取API密钥，尝试多种方式"""

        # 方法1：环境变量
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key.strip():
            logger.info("使用环境变量中的API密钥")
            return api_key.strip()

        # 方法2：.env文件
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key.strip():
            logger.info("使用.env文件中的API密钥")
            return api_key.strip()

        # 方法3：检查其他可能的变量名
        alt_names = ["OPENROUTER_API_KEY", "API_KEY", "LLM_API_KEY"]
        for name in alt_names:
            api_key = os.getenv(name)
            if api_key and api_key.strip():
                logger.info(f"使用{name}作为API密钥")
                return api_key.strip()

        logger.error("未找到任何有效的API密钥")
        return None

    def generate_answer(self, question: str, contexts: list, **kwargs) -> str:
        """生成回答，强制使用真实API，带重试机制"""

        if self.use_mock:
            logger.error("配置错误：尝试使用模拟模式，但应该使用真实API")
            raise RuntimeError("API配置错误：模拟模式被禁用，需要有效的API密钥")

        # 重试配置
        max_retries = kwargs.get('max_retries', 3)
        retry_delay = kwargs.get('retry_delay', 2)  # 秒

        for attempt in range(max_retries):
            try:
                # 构建提示
                context_text = "\n\n".join([
                    f"文档{i+1}: {ctx.get('title', f'Doc_{i+1}')}\n{ctx.get('text', ctx) if isinstance(ctx, dict) else ctx}"
                    for i, ctx in enumerate(contexts)
                ])

                prompt = f"""基于以下检索到的文档，请回答问题：

问题：{question}

检索到的文档：
{context_text}

请根据上述文档内容，提供一个准确、简洁的回答。如果文档中没有足够的信息来回答问题，请明确说明。"""

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个基于检索文档回答问题的助手。请严格根据提供的文档内容回答问题，不要添加文档中没有的信息。"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=kwargs.get('max_tokens', 150),
                    temperature=kwargs.get('temperature', 0.3)
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    logger.info(f"等待 {retry_delay} 秒后重试...")
                    import time
                    time.sleep(retry_delay)
                    # 指数退避
                    retry_delay *= 2
                else:
                    logger.error(f"API调用最终失败，已重试 {max_retries} 次")
                    raise RuntimeError(f"API调用失败，无法生成回答: {e}")

    def _mock_generate(self, question: str, contexts: list) -> str:
        """模拟回答生成"""
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

# 向后兼容的API
setup_openai_api = lambda: FixedAPIClient()

def generate_answer_with_fixed_api(question: str, retrieved_docs: list, client=None) -> str:
    """使用修复版API生成回答"""
    if client is None:
        client = FixedAPIClient()
    return client.generate_answer(question, retrieved_docs)

if __name__ == "__main__":
    print("🔧 测试修复版API客户端...")

    client = FixedAPIClient()
    test_contexts = [
        {"title": "测试文档1", "text": "这是一个关于机器学习的文档。机器学习是人工智能的一个分支。"},
        {"title": "测试文档2", "text": "深度学习是机器学习的一个子集，使用神经网络。"}
    ]

    answer = client.generate_answer("什么是机器学习？", test_contexts)
    print(f"回答: {answer}")
    print("✅ 修复版API客户端测试完成")