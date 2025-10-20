"""
基于嵌入向量的检索器实现
使用向量相似度进行文档检索
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional
import requests
import json
import os
from pathlib import Path

from .base_retriever import BaseRetriever, RetrievalResult
from core.config import get_config
from core.utils import calculate_similarity, load_json_file, save_json_file

logger = logging.getLogger(__name__)

class EmbeddingRetriever(BaseRetriever):
    """基于嵌入向量的检索器"""

    def __init__(self, embedding_model: str = "bge-m3", cache_dir: str = "cache"):
        super().__init__("embedding")
        self.config = get_config()
        self.embedding_model = embedding_model or self.config.retrieval.embedding_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.documents = []
        self.embeddings = []
        self.document_embeddings_cache = {}

    def setup_knowledge_base(self, documents: List[Dict[str, Any]],
                           embeddings: Optional[List[List[float]]] = None,
                           cache_file: Optional[str] = None) -> None:
        """设置知识库"""
        self.documents = documents

        if embeddings is not None:
            self.embeddings = embeddings
        elif cache_file and os.path.exists(cache_file):
            # 从缓存加载嵌入
            try:
                self.embeddings = load_json_file(cache_file)
                logger.info(f"从缓存加载嵌入向量: {len(self.embeddings)} 个文档")
            except Exception as e:
                logger.warning(f"加载嵌入缓存失败: {e}")
                self.embeddings = self._generate_embeddings_for_documents()
        else:
            # 生成嵌入
            self.embeddings = self._generate_embeddings_for_documents()

        logger.info(f"知识库设置完成 - 文档数量: {len(self.documents)}")

    def _generate_embeddings_for_documents(self) -> List[List[float]]:
        """为文档生成嵌入向量"""
        logger.info("为文档生成嵌入向量...")

        if not self.documents:
            return []

        embeddings = []
        for i, doc in enumerate(self.documents):
            try:
                text = doc.get('text', '')
                if not text:
                    logger.warning(f"文档 {i} 没有文本内容，使用零向量")
                    embeddings.append([0.0] * self.config.retrieval.embedding_dim)
                    continue

                embedding = self._get_embedding(text)
                embeddings.append(embedding)

                if (i + 1) % 10 == 0:
                    logger.info(f"已处理 {i + 1}/{len(self.documents)} 个文档")

            except Exception as e:
                logger.error(f"文档 {i} 嵌入生成失败: {e}")
                embeddings.append([0.0] * self.config.retrieval.embedding_dim)

        logger.info(f"嵌入向量生成完成 - 维度: {len(embeddings[0]) if embeddings else 0}")
        return embeddings

    def _get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        # 检查缓存
        if text in self.document_embeddings_cache:
            return self.document_embeddings_cache[text]

        try:
            # 调用本地ollama服务
            model_url = 'http://localhost:11434/api/embeddings'
            headers = {'Content-Type': 'application/json'}

            data = {
                'model': self.embedding_model,
                'prompt': text
            }

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(model_url, json=data, headers=headers, timeout=30)
                    response.raise_for_status()
                    embedding = response.json()['embedding']

                    # 缓存结果
                    self.document_embeddings_cache[text] = embedding
                    return embedding

                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = (attempt + 1) * 5  # 指数退避
                    logger.warning(f"嵌入服务调用失败，尝试 {attempt + 1}: {e}。等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)

        except Exception as e:
            logger.error(f"获取嵌入向量失败: {e}")
            # 返回零向量作为回退
            return [0.0] * self.config.retrieval.embedding_dim

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> RetrievalResult:
        """检索文档"""
        start_time = time.time()

        try:
            self._validate_query(query)

            if not self.documents or not self.embeddings:
                logger.warning("知识库未设置或为空")
                return self._create_empty_result(query)

            # 获取查询的嵌入向量
            query_embedding = self._get_embedding(query)

            # 计算相似度
            similarities = []
            for doc_embedding in self.embeddings:
                if doc_embedding and query_embedding:
                    sim = calculate_similarity(doc_embedding, query_embedding)
                    similarities.append(sim)
                else:
                    similarities.append(0.0)

            # 获取top-k文档
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            retrieved_docs = [self.documents[i] for i in top_indices]
            retrieved_scores = [similarities[i] for i in top_indices]

            # 过滤低相似度文档
            threshold = kwargs.get('similarity_threshold', self.config.retrieval.similarity_threshold)
            filtered_docs = []
            filtered_scores = []

            for doc, score in zip(retrieved_docs, retrieved_scores):
                if score >= threshold:
                    filtered_docs.append(doc)
                    filtered_scores.append(score)

            retrieval_time = time.time() - start_time

            return RetrievalResult(
                query=query,
                documents=filtered_docs,
                scores=filtered_scores,
                retriever_type=self.retriever_type,
                retrieval_time=retrieval_time,
                total_documents=len(filtered_docs),
                metadata={
                    'embedding_model': self.embedding_model,
                    'similarity_threshold': threshold,
                    'top_k_requested': top_k
                }
            )

        except Exception as e:
            logger.error(f"检索失败: {e}")
            retrieval_time = time.time() - start_time
            result = self._create_empty_result(query)
            result.retrieval_time = retrieval_time
            return result

    def batch_retrieve(self, queries: List[str], top_k: int = 5, **kwargs) -> List[RetrievalResult]:
        """批量检索文档"""
        results = []
        for i, query in enumerate(queries):
            try:
                result = self.retrieve(query, top_k, **kwargs)
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"已处理 {i + 1}/{len(queries)} 个查询")

            except Exception as e:
                logger.error(f"批量检索中查询 {i} 失败: {e}")
                error_result = self._create_empty_result(query)
                results.append(error_result)

        return results

    def save_embeddings_cache(self, cache_file: str) -> None:
        """保存嵌入向量缓存"""
        try:
            save_json_file(self.embeddings, cache_file)
            logger.info(f"嵌入向量缓存已保存: {cache_file}")
        except Exception as e:
            logger.error(f"保存嵌入向量缓存失败: {e}")

    def load_embeddings_cache(self, cache_file: str) -> bool:
        """加载嵌入向量缓存"""
        try:
            self.embeddings = load_json_file(cache_file)
            logger.info(f"嵌入向量缓存已加载: {len(self.embeddings)} 个文档")
            return True
        except Exception as e:
            logger.error(f"加载嵌入向量缓存失败: {e}")
            return False

    def get_supported_features(self) -> List[str]:
        """获取支持的功能列表"""
        return ['vector_similarity', 'cosine_similarity', 'threshold_filtering', 'batch_processing']

    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息"""
        info = super().get_config_info()
        info.update({
            'embedding_model': self.embedding_model,
            'embedding_dim': self.config.retrieval.embedding_dim,
            'similarity_threshold': self.config.retrieval.similarity_threshold,
            'cache_enabled': self.config.retrieval.cache_embeddings
        })
        return info

# 向后兼容的函数
def get_embeddings(texts: str) -> List[float]:
    """获取文本嵌入向量（向后兼容）"""
    retriever = EmbeddingRetriever()
    return retriever._get_embedding(texts)

def load_or_generate_embeddings(knowledge_base_path: str) -> List[List[float]]:
    """加载或生成知识库文档的嵌入向量（向后兼容）"""
    from pathlib import Path

    documents_file = Path(knowledge_base_path) / 'documents.json'
    embeddings_file = Path(knowledge_base_path) / 'document_embeddings.json'

    # 加载文档
    documents = load_json_file(str(documents_file))

    # 创建检索器
    retriever = EmbeddingRetriever()
    retriever.setup_knowledge_base(documents)

    # 保存缓存
    if retriever.embeddings:
        retriever.save_embeddings_cache(str(embeddings_file))

    return retriever.embeddings

def retrieve_documents(query: str, documents: List[Dict], embeddings: List[List[float]], top_k: int = 5) -> List[Dict]:
    """检索文档（向后兼容）"""
    retriever = EmbeddingRetriever()
    retriever.setup_knowledge_base(documents, embeddings)
    result = retriever.retrieve(query, top_k)
    return result.documents

if __name__ == "__main__":
    # 测试检索器
    print("🔧 测试嵌入向量检索器...")

    # 创建测试文档
    test_documents = [
        {'id': 1, 'title': '机器学习', 'text': '机器学习是人工智能的一个分支，通过数据学习模式。'},
        {'id': 2, 'title': '深度学习', 'text': '深度学习是机器学习的一个子集，使用多层神经网络。'},
        {'id': 3, 'title': '自然语言处理', 'text': '自然语言处理是人工智能的重要应用领域。'}
    ]

    # 创建检索器
    retriever = EmbeddingRetriever()

    # 注意：由于需要ollama服务，这里只测试基本功能
    print(f"✅ 检索器创建成功: {retriever.retriever_type}")
    print(f"✅ 支持的功能: {retriever.get_supported_features()}")
    print(f"✅ 配置信息: {retriever.get_config_info()}")

    print("\n✅ 嵌入向量检索器测试完成")