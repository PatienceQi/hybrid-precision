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
import re
from pathlib import Path

from ..core.config import get_config
from ..core.utils import calculate_similarity, load_json_file, save_json_file
from .base_retriever import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)

class EmbeddingRetriever(BaseRetriever):
    """基于嵌入向量的检索器"""

    def __init__(self, embedding_model: Optional[str] = None, cache_dir: str = "cache"):
        super().__init__("embedding")
        self.config = get_config()
        self.embedding_model = embedding_model or self.config.retrieval.embedding_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.documents = []
        self.embeddings = []
        self.document_embeddings_cache = {}
        self.embedding_service_url = (self.config.retrieval.embedding_service_url or "").strip()
        if not self.embedding_service_url:
            self.embedding_service_url = "https://wolfai.top/v1/embeddings"
        self.embedding_api_key = (
            (self.config.retrieval.embedding_api_key or "") or os.getenv("EMBEDDING_API_KEY") or "sk-7tk8aNrEJw3nmix9FeciFbgvvcr77hSwlpTaWKMH4FRwu84j"
        )
        self._force_service = self.config.retrieval.force_embedding_service
        self._allow_fallback = self.config.retrieval.fallback_to_local_embeddings
        self._embedding_service_available = True
        self._service_warning_emitted = False
        self._missing_text_warning_emitted = False

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
                text = self._extract_text(doc)
                if not text:
                    if not self._missing_text_warning_emitted:
                        logger.warning("文档缺少文本内容，后续将使用零向量占位")
                        self._missing_text_warning_emitted = True
                    embeddings.append([0.0] * self.config.retrieval.embedding_dim)
                    continue
                if isinstance(doc, dict):
                    doc.setdefault('text', text)

                embedding = self._get_embedding(text)
                embeddings.append(embedding)

                if (i + 1) % 10 == 0:
                    logger.info(f"已处理 {i + 1}/{len(self.documents)} 个文档")

            except Exception as e:
                logger.error(f"文档 {i} 嵌入生成失败: {e}")
                embeddings.append([0.0] * self.config.retrieval.embedding_dim)

        logger.info(f"嵌入向量生成完成 - 维度: {len(embeddings[0]) if embeddings else 0}")
        return embeddings

    def _extract_text(self, document: Dict[str, Any]) -> str:
        """从文档中提取文本内容"""
        if not document:
            return ""

        candidate_keys = ["text", "content", "body", "summary", "passage", "document"]
        for key in candidate_keys:
            value = document.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                combined = " ".join(str(item) for item in value if isinstance(item, str))
                if combined.strip():
                    return combined.strip()

        # 尝试从嵌套结构提取
        for key in ["paragraphs", "sentences"]:
            value = document.get(key)
            if isinstance(value, list):
                combined = " ".join(str(item) for item in value if isinstance(item, str))
                if combined.strip():
                    return combined.strip()

        return ""

    def _get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        # 检查缓存
        if text in self.document_embeddings_cache:
            return self.document_embeddings_cache[text]

        if not self._embedding_service_available and self._allow_fallback:
            embedding = self._generate_local_embedding(text)
            self.document_embeddings_cache[text] = embedding
            return embedding

        try:
            # 调用远程嵌入服务
            model_url = self.embedding_service_url
            headers = {'Content-Type': 'application/json'}
            if self.embedding_api_key:
                headers['Authorization'] = f"Bearer {self.embedding_api_key}"

            data = {
                'model': self.embedding_model,
                'input': text,
                'encoding_format': 'float'
            }

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(model_url, json=data, headers=headers, timeout=30)
                    response.raise_for_status()
                    response_data = response.json()

                    embedding = None

                    if isinstance(response_data, dict):
                        if 'embedding' in response_data:
                            embedding = response_data['embedding']
                        elif 'embeddings' in response_data:
                            embeddings_field = response_data['embeddings']
                            if isinstance(embeddings_field, list) and embeddings_field:
                                embedding = embeddings_field[0]
                        elif 'data' in response_data and isinstance(response_data['data'], list) and response_data['data']:
                            first_item = response_data['data'][0]
                            if isinstance(first_item, dict) and 'embedding' in first_item:
                                embedding = first_item['embedding']
                            else:
                                embedding = first_item
                        else:
                            raise ValueError(f"嵌入响应缺少预期字段: {response_data}")
                    elif isinstance(response_data, list) and response_data:
                        embedding = response_data[0]
                    else:
                        raise ValueError(f"未知的嵌入响应格式: {response_data}")

                    if isinstance(embedding, dict):
                        raise ValueError(f"无法解析嵌入向量: {embedding}")

                    if isinstance(embedding, tuple):
                        embedding = list(embedding)

                    if not isinstance(embedding, list):
                        raise ValueError(f"嵌入向量不是列表: {type(embedding)}")

                    try:
                        embedding = [float(value) for value in embedding]
                    except (TypeError, ValueError) as conversion_error:
                        raise ValueError(f"嵌入向量包含非数值元素: {embedding}") from conversion_error

                    # 缓存结果
                    self.document_embeddings_cache[text] = embedding
                    return embedding

                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = (attempt + 1) * 5  # 指数退避
                    logger.warning(f"远程嵌入服务调用失败，尝试 {attempt + 1}: {e}。等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)

        except Exception as e:
            if self._force_service:
                if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                    status = e.response.status_code
                    if status == 405:
                        raise RuntimeError(
                            "嵌入服务返回 405 Method Not Allowed，请确认端点是否正确。"
                        ) from e
                    if status == 404:
                        raise RuntimeError(
                            "嵌入服务返回 404 Not Found，可能是路径错误或服务未启动。"
                        ) from e
                    body = e.response.text
                    raise RuntimeError(
                        f"嵌入服务HTTP错误 {status}: {body}"
                    ) from e
                raise RuntimeError(
                    f"嵌入服务调用失败且已启用FORCE_EMBEDDING_SERVICE: {e}. "
                    f"请确认服务正在 {self.embedding_service_url} 运行。"
                ) from e

            if not self._allow_fallback:
                raise RuntimeError(
                    f"嵌入服务调用失败且未允许本地回退: {e}. "
                    "可设置 EMBEDDING_FALLBACK_LOCAL=true 允许本地模拟嵌入。"
                ) from e

            if not self._service_warning_emitted:
                logger.warning(f"嵌入服务不可用，使用本地模拟嵌入: {e}")
                self._service_warning_emitted = True
            self._embedding_service_available = False

        # 服务不可用时的回退
        embedding = self._generate_local_embedding(text)
        self.document_embeddings_cache[text] = embedding
        return embedding

    def _ensure_text_field(self, document: Dict[str, Any]) -> None:
        """确保文档包含text字段"""
        if not isinstance(document, dict):
            return
        text = document.get("text")
        if isinstance(text, str) and text.strip():
            return
        extracted = self._extract_text(document)
        if extracted:
            document["text"] = extracted

    def _generate_local_embedding(self, text: str) -> List[float]:
        """使用简单的哈希技巧生成本地模拟嵌入"""
        tokens = [token for token in re.split(r"\W+", text.lower()) if token]
        dim = self.config.retrieval.embedding_dim
        if not tokens or dim <= 0:
            return [0.0] * dim

        vector = np.zeros(dim, dtype=float)
        for token in tokens:
            index = hash(token) % dim
            vector[index] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector.tolist()

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

            for doc in retrieved_docs:
                self._ensure_text_field(doc)

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

    # 注意：由于依赖远程嵌入服务，这里只测试基本功能
    print(f"✅ 检索器创建成功: {retriever.retriever_type}")
    print(f"✅ 支持的功能: {retriever.get_supported_features()}")
    print(f"✅ 配置信息: {retriever.get_config_info()}")

    print("\n✅ 嵌入向量检索器测试完成")
