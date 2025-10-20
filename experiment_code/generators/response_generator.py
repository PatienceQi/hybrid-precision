"""
响应生成器 - 基于检索结果生成回答
"""

import logging
from typing import Dict, List, Any, Optional
from core.api_client import BaseAPIClient, APIClientFactory
from core.config import get_config
from retrievers import BaseRetriever, RetrievalResult, create_retriever

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """响应生成器 - 基于检索结果生成回答"""

    def __init__(self,
                 retriever: Optional[BaseRetriever] = None,
                 api_client: Optional[BaseAPIClient] = None,
                 fusion_method: str = "unknown"):
        self.config = get_config()
        self.retriever = retriever or create_retriever('hybrid')
        self.api_client = api_client or APIClientFactory.create_client()
        self.fusion_method = fusion_method

    def generate_response(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        生成响应

        Args:
            query: 查询文本
            top_k: 检索文档数量
            **kwargs: 其他参数

        Returns:
            包含问题和检索结果的字典
        """
        try:
            # 检索相关文档
            retrieval_result = self.retriever.retrieve(query, top_k, **kwargs)

            # 生成回答
            answer = self.api_client.generate_answer(query, retrieval_result.documents, **kwargs)

            return {
                'question': query,
                'answer': answer,
                'contexts': retrieval_result.get_document_texts(),
                'retrieved_documents': retrieval_result.documents,
                'retrieval_scores': retrieval_result.scores,
                'retrieval_metadata': retrieval_result.metadata,
                'fusion_method': self.fusion_method
            }

        except Exception as e:
            logger.error(f"响应生成失败: {e}")
            return {
                'question': query,
                'answer': f"抱歉，生成回答时出现错误: {str(e)}",
                'contexts': [],
                'retrieved_documents': [],
                'retrieval_scores': [],
                'retrieval_metadata': {},
                'fusion_method': self.fusion_method,
                'error': str(e)
            }

    def generate_hybrid_response(self, query: str, context: List[str], top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        生成增强的混合响应

        Args:
            query: 查询文本
            context: 真实上下文（用于评估）
            top_k: 检索文档数量
            **kwargs: 其他参数

        Returns:
            包含混合检索元数据的响应字典
        """
        try:
            # 检索相关文档
            retrieval_result = self.retriever.retrieve(query, top_k, **kwargs)

            # 生成回答
            answer = self.api_client.generate_answer(query, retrieval_result.documents, **kwargs)

            # 构建混合检索元数据
            hybrid_metadata = {
                "retrieval_method": "hybrid_search",
                "fusion_method": self.fusion_method,
                "retrieval_strategy": f"Consider hybrid search blending semantic and keyword relevance",
                "context_count": len(retrieval_result.documents),
                "fusion_optimized": True,
                "retrieval_time": retrieval_result.retrieval_time
            }

            # 增强上下文（添加混合检索提示）
            enhanced_contexts = []
            for doc in retrieval_result.documents:
                text = doc.get('text', '')
                # 添加混合检索提示到每个上下文的开头
                if self.fusion_method == 'cascading':
                    enhanced_text = f"[Cascading Hybrid Retrieval] {text}"
                elif self.fusion_method == 'weighted_rrf':
                    enhanced_text = f"[Weighted RRF Fusion] {text}"
                elif self.fusion_method == 'linear_weighted':
                    enhanced_text = f"[Linear Weighted Fusion] {text}"
                else:
                    enhanced_text = text
                enhanced_contexts.append(enhanced_text)

            return {
                'question': query,
                'answer': answer,
                'contexts': enhanced_contexts,
                'ground_truth': context,
                'hybrid_metadata': hybrid_metadata,
                'fusion_method': self.fusion_method,
                'retrieved_documents': retrieval_result.documents,
                'retrieval_scores': retrieval_result.scores
            }

        except Exception as e:
            logger.error(f"混合响应生成失败: {e}")
            return {
                'question': query,
                'answer': f"抱歉，生成混合回答时出现错误: {str(e)}",
                'contexts': [],
                'ground_truth': context,
                'hybrid_metadata': {},
                'fusion_method': self.fusion_method,
                'retrieved_documents': [],
                'retrieval_scores': [],
                'error': str(e)
            }

    def batch_generate_responses(self, queries: List[str], top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        批量生成响应

        Args:
            queries: 查询列表
            top_k: 检索文档数量
            **kwargs: 其他参数

        Returns:
            响应列表
        """
        responses = []
        for i, query in enumerate(queries):
            try:
                response = self.generate_response(query, top_k, **kwargs)
                responses.append(response)

                if (i + 1) % 10 == 0:
                    logger.info(f"已生成 {i + 1}/{len(queries)} 个响应")

            except Exception as e:
                logger.error(f"批量生成中查询 {i} 失败: {e}")
                error_response = {
                    'question': query,
                    'answer': f"生成失败: {str(e)}",
                    'contexts': [],
                    'retrieved_documents': [],
                    'retrieval_scores': [],
                    'error': str(e)
                }
                responses.append(error_response)

        return responses

    def setup_knowledge_base(self, documents: List[Dict[str, Any]],
                           embeddings: Optional[List[List[float]]] = None) -> None:
        """
        设置知识库

        Args:
            documents: 文档列表
            embeddings: 嵌入向量列表（可选）
        """
        if hasattr(self.retriever, 'setup_knowledge_base'):
            self.retriever.setup_knowledge_base(documents, embeddings)
            logger.info(f"知识库设置完成 - 文档数量: {len(documents)}")
        else:
            logger.warning("检索器不支持知识库设置")

    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            'generator_type': 'response_generator',
            'retriever_type': self.retriever.retriever_type,
            'fusion_method': self.fusion_method,
            'api_client_type': type(self.api_client).__name__,
            'model': self.api_client.model
        }

# 向后兼容的函数
def generate_response(query: str, retrieved_docs: List[Dict], context: List[str], **kwargs) -> Dict[str, Any]:
    """生成响应（向后兼容）"""
    generator = ResponseGenerator()
    return generator.generate_response(query, **kwargs)

def generate_hybrid_response(query: str, retrieved_docs: List[Dict], context: List[str],
                           fusion_method: str = "unknown", **kwargs) -> Dict[str, Any]:
    """生成混合响应（向后兼容）"""
    generator = ResponseGenerator(fusion_method=fusion_method)
    return generator.generate_hybrid_response(query, context, **kwargs)

if __name__ == "__main__":
    # 测试响应生成器
    print("🔧 测试响应生成器...")

    # 创建响应生成器
    generator = ResponseGenerator()

    print(f"✅ 响应生成器创建成功")
    print(f"✅ 配置信息: {generator.get_config_info()}")

    print("\n✅ 响应生成器测试完成")