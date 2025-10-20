"""
混合检索器实现
结合向量检索和关键词检索的混合方法
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from collections import Counter
import re

from .base_retriever import BaseRetriever, RetrievalResult
from .embedding_retriever import EmbeddingRetriever
from core.config import get_config
from core.utils import normalize_text, extract_keywords, calculate_similarity, load_json_file

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """混合检索器 - 结合向量检索和关键词检索"""

    def __init__(self,
                 embedding_model: str = "bge-m3",
                 fusion_method: str = "weighted_rrf",
                 cache_dir: str = "cache"):
        super().__init__("hybrid")
        self.config = get_config()
        self.embedding_retriever = EmbeddingRetriever(embedding_model, cache_dir)
        self.fusion_method = fusion_method
        self.documents = []
        self.keyword_index = {}  # 关键词索引

        # 融合权重配置
        self.fusion_weights = {
            "vector_weight": 0.7,
            "keyword_weight": 0.3,
            "rrf_k": 60  # RRF公式中的k值
        }

    def setup_knowledge_base(self, documents: List[Dict[str, Any]],
                           embeddings: Optional[List[List[float]]] = None,
                           build_keyword_index: bool = True) -> None:
        """设置知识库"""
        self.documents = documents

        # 设置嵌入检索器
        self.embedding_retriever.setup_knowledge_base(documents, embeddings)

        # 构建关键词索引
        if build_keyword_index:
            self._build_keyword_index()

    def load_knowledge_base_from_file(self, knowledge_file: str = "knowledge_data/knowledge_base.json") -> bool:
        """
        从文件加载知识库

        Args:
            knowledge_file: 知识库文件路径

        Returns:
            是否成功加载
        """
        try:
            logger.info(f"正在加载知识库: {knowledge_file}")

            # 加载知识库数据
            kb_data = load_json_file(knowledge_file)
            if not kb_data:
                logger.warning("知识库文件不存在或为空")
                return False

            # 提取文档
            documents = kb_data.get('documents', [])
            if not documents:
                logger.warning("知识库中没有文档")
                return False

            logger.info(f"加载了 {len(documents)} 个文档")

            # 设置知识库
            self.setup_knowledge_base(documents)

            logger.info("✅ 知识库加载成功")
            return True

        except Exception as e:
            logger.error(f"加载知识库失败: {e}")
            return False

    def has_knowledge_base(self) -> bool:
        """检查是否有知识库"""
        return len(self.documents) > 0

    def _build_keyword_index(self) -> None:
        """构建关键词索引"""
        logger.info("构建关键词索引...")

        self.keyword_index = {}

        for i, doc in enumerate(self.documents):
            try:
                text = doc.get('text', '')
                if not text:
                    continue

                # 提取关键词
                keywords = extract_keywords(text, min_length=3, top_k=20)

                # 构建倒排索引
                for keyword in keywords:
                    if keyword not in self.keyword_index:
                        self.keyword_index[keyword] = []
                    self.keyword_index[keyword].append({
                        'doc_id': i,
                        'frequency': text.lower().count(keyword),
                        'doc': doc
                    })

                if (i + 1) % 100 == 0:
                    logger.info(f"已处理 {i + 1}/{len(self.documents)} 个文档的关键词")

            except Exception as e:
                logger.error(f"文档 {i} 关键词索引构建失败: {e}")

        logger.info(f"关键词索引构建完成 - 关键词数量: {len(self.keyword_index)}")

    def _keyword_search(self, query: str, top_k: int = 10) -> tuple:
        """关键词检索"""
        try:
            # 提取查询关键词
            query_keywords = extract_keywords(query, min_length=3, top_k=10)

            if not query_keywords:
                return [], []

            # 计算文档得分
            doc_scores = Counter()
            doc_matches = {}  # 存储匹配的文档信息

            for keyword in query_keywords:
                if keyword in self.keyword_index:
                    for match in self.keyword_index[keyword]:
                        doc_id = match['doc_id']

                        # 计算得分：词频 * 查询词权重
                        score = match['frequency'] * (query_keywords.index(keyword) + 1)
                        doc_scores[doc_id] += score

                        # 存储匹配信息
                        if doc_id not in doc_matches:
                            doc_matches[doc_id] = {
                                'matched_keywords': [],
                                'total_frequency': 0
                            }
                        doc_matches[doc_id]['matched_keywords'].append(keyword)
                        doc_matches[doc_id]['total_frequency'] += match['frequency']

            if not doc_scores:
                return [], []

            # 获取top文档
            top_doc_ids = [doc_id for doc_id, _ in doc_scores.most_common(top_k)]
            keyword_docs = [self.documents[doc_id] for doc_id in top_doc_ids]
            keyword_scores = [doc_scores[doc_id] for doc_id in top_doc_ids]

            # 标准化分数到0-1范围
            if keyword_scores:
                max_score = max(keyword_scores)
                if max_score > 0:
                    keyword_scores = [score / max_score for score in keyword_scores]

            return keyword_docs, keyword_scores

        except Exception as e:
            logger.error(f"关键词检索失败: {e}")
            return [], []

    def _fuse_results(self, vector_result: RetrievalResult, keyword_docs: List[Dict],
                     keyword_scores: List[float], top_k: int = 5) -> RetrievalResult:
        """融合检索结果"""
        try:
            if self.fusion_method == "weighted_sum":
                return self._fuse_weighted_sum(vector_result, keyword_docs, keyword_scores, top_k)
            elif self.fusion_method == "rrf":
                return self._fuse_rrf(vector_result, keyword_docs, keyword_scores, top_k)
            elif self.fusion_method == "cascading":
                return self._fuse_cascading(vector_result, keyword_docs, keyword_scores, top_k)
            else:
                # 默认使用加权RRF
                return self._fuse_weighted_rrf(vector_result, keyword_docs, keyword_scores, top_k)

        except Exception as e:
            logger.error(f"结果融合失败: {e}")
            # 回退到向量检索结果
            return vector_result

    def _fuse_weighted_sum(self, vector_result: RetrievalResult, keyword_docs: List[Dict],
                          keyword_scores: List[float], top_k: int) -> RetrievalResult:
        """加权求和融合"""
        try:
            # 创建文档得分映射
            doc_scores = {}

            # 添加向量检索得分
            for doc, score in zip(vector_result.documents, vector_result.scores):
                doc_id = doc.get('id', hash(str(doc)))
                doc_scores[doc_id] = {
                    'doc': doc,
                    'vector_score': score,
                    'keyword_score': 0.0
                }

            # 添加关键词检索得分
            for doc, score in zip(keyword_docs, keyword_scores):
                doc_id = doc.get('id', hash(str(doc)))
                if doc_id in doc_scores:
                    doc_scores[doc_id]['keyword_score'] = score
                else:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'vector_score': 0.0,
                        'keyword_score': score
                    }

            # 计算融合得分
            fused_results = []
            for doc_id, scores in doc_scores.items():
                vector_score = scores['vector_score'] * self.fusion_weights['vector_weight']
                keyword_score = scores['keyword_score'] * self.fusion_weights['keyword_weight']
                fused_score = vector_score + keyword_score

                fused_results.append({
                    'doc': scores['doc'],
                    'score': fused_score,
                    'vector_score': scores['vector_score'],
                    'keyword_score': scores['keyword_score']
                })

            # 排序并获取top-k
            fused_results.sort(key=lambda x: x['score'], reverse=True)
            top_results = fused_results[:top_k]

            # 构建最终结果
            final_docs = [result['doc'] for result in top_results]
            final_scores = [result['score'] for result in top_results]

            # 更新检索结果
            vector_result.documents = final_docs
            vector_result.scores = final_scores
            vector_result.total_documents = len(final_docs)
            vector_result.metadata.update({
                'fusion_method': 'weighted_sum',
                'vector_weight': self.fusion_weights['vector_weight'],
                'keyword_weight': self.fusion_weights['keyword_weight']
            })

            return vector_result

        except Exception as e:
            logger.error(f"加权求和融合失败: {e}")
            return vector_result

    def _fuse_rrf(self, vector_result: RetrievalResult, keyword_docs: List[Dict],
                  keyword_scores: List[float], top_k: int) -> RetrievalResult:
        """RRF (Reciprocal Rank Fusion) 融合"""
        try:
            k = self.fusion_weights['rrf_k']
            doc_scores = {}

            # 向量检索排名得分
            for rank, (doc, score) in enumerate(zip(vector_result.documents, vector_result.scores)):
                doc_id = doc.get('id', hash(str(doc)))
                rrf_score = 1.0 / (k + rank + 1)
                doc_scores[doc_id] = {
                    'doc': doc,
                    'rrf_score': rrf_score,
                    'vector_score': score,
                    'keyword_score': 0.0
                }

            # 关键词检索排名得分
            for rank, (doc, score) in enumerate(zip(keyword_docs, keyword_scores)):
                doc_id = doc.get('id', hash(str(doc)))
                rrf_score = 1.0 / (k + rank + 1)

                if doc_id in doc_scores:
                    doc_scores[doc_id]['rrf_score'] += rrf_score
                    doc_scores[doc_id]['keyword_score'] = score
                else:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'rrf_score': rrf_score,
                        'vector_score': 0.0,
                        'keyword_score': score
                    }

            # 排序并获取top-k
            fused_results = []
            for doc_id, scores in doc_scores.items():
                fused_results.append({
                    'doc': scores['doc'],
                    'score': scores['rrf_score'],
                    'vector_score': scores['vector_score'],
                    'keyword_score': scores['keyword_score']
                })

            fused_results.sort(key=lambda x: x['score'], reverse=True)
            top_results = fused_results[:top_k]

            # 构建最终结果
            final_docs = [result['doc'] for result in top_results]
            final_scores = [result['score'] for result in top_results]

            # 更新检索结果
            vector_result.documents = final_docs
            vector_result.scores = final_scores
            vector_result.total_documents = len(final_docs)
            vector_result.metadata.update({
                'fusion_method': 'rrf',
                'rrf_k': k
            })

            return vector_result

        except Exception as e:
            logger.error(f"RRF融合失败: {e}")
            return vector_result

    def _fuse_cascading(self, vector_result: RetrievalResult, keyword_docs: List[Dict],
                       keyword_scores: List[float], top_k: int) -> RetrievalResult:
        """级联融合 - 优先使用向量检索结果"""
        try:
            # 如果向量检索结果足够好，直接使用
            if vector_result.documents and len(vector_result.documents) >= top_k * 0.8:
                # 使用向量检索结果，但用关键词结果补充
                final_docs = vector_result.documents[:top_k]
                final_scores = vector_result.scores[:top_k]

                # 如果需要，用关键词结果补充
                if len(final_docs) < top_k and keyword_docs:
                    needed = top_k - len(final_docs)
                    final_docs.extend(keyword_docs[:needed])
                    final_scores.extend(keyword_scores[:needed])

            else:
                # 向量检索结果不足，使用关键词检索结果
                final_docs = keyword_docs[:top_k]
                final_scores = keyword_scores[:top_k]

            # 更新检索结果
            vector_result.documents = final_docs
            vector_result.scores = final_scores
            vector_result.total_documents = len(final_docs)
            vector_result.metadata.update({
                'fusion_method': 'cascading',
                'primary_method': 'vector',
                'secondary_method': 'keyword'
            })

            return vector_result

        except Exception as e:
            logger.error(f"级联融合失败: {e}")
            return vector_result

    def _fuse_weighted_rrf(self, vector_result: RetrievalResult, keyword_docs: List[Dict],
                          keyword_scores: List[float], top_k: int) -> RetrievalResult:
        """加权RRF融合"""
        try:
            # 先进行RRF融合
            rrf_result = self._fuse_rrf(vector_result, keyword_docs, keyword_scores, top_k * 2)

            # 然后进行加权求和微调
            return self._fuse_weighted_sum(rrf_result, [], [], top_k)

        except Exception as e:
            logger.error(f"加权RRF融合失败: {e}")
            return vector_result

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> RetrievalResult:
        """混合检索"""
        start_time = time.time()

        try:
            self._validate_query(query)

            if not self.documents:
                logger.warning("知识库为空")
                return self._create_empty_result(query)

            # 1. 向量检索
            vector_result = self.embedding_retriever.retrieve(query, top_k * 2, **kwargs)

            # 2. 关键词检索
            keyword_docs, keyword_scores = self._keyword_search(query, top_k * 2)

            # 3. 结果融合
            if not vector_result.documents and not keyword_docs:
                return vector_result

            fused_result = self._fuse_results(vector_result, keyword_docs, keyword_scores, top_k)

            # 更新检索时间和类型
            retrieval_time = time.time() - start_time
            fused_result.retrieval_time = retrieval_time
            fused_result.retriever_type = self.retriever_type
            fused_result.metadata.update({
                'fusion_method': self.fusion_method,
                'vector_results': len(vector_result.documents),
                'keyword_results': len(keyword_docs)
            })

            return fused_result

        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            retrieval_time = time.time() - start_time
            result = self._create_empty_result(query)
            result.retrieval_time = retrieval_time
            return result

    def batch_retrieve(self, queries: List[str], top_k: int = 5, **kwargs) -> List[RetrievalResult]:
        """批量混合检索"""
        results = []
        for i, query in enumerate(queries):
            try:
                result = self.retrieve(query, top_k, **kwargs)
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"已处理 {i + 1}/{len(queries)} 个混合查询")

            except Exception as e:
                logger.error(f"批量混合检索中查询 {i} 失败: {e}")
                error_result = self._create_empty_result(query)
                results.append(error_result)

        return results

    def get_supported_features(self) -> List[str]:
        """获取支持的功能列表"""
        return [
            'vector_similarity', 'keyword_matching', 'hybrid_fusion',
            'weighted_sum_fusion', 'rrf_fusion', 'cascading_fusion',
            'threshold_filtering', 'batch_processing'
        ]

    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息"""
        info = super().get_config_info()
        info.update({
            'fusion_method': self.fusion_method,
            'fusion_weights': self.fusion_weights,
            'embedding_model': self.embedding_retriever.embedding_model
        })
        return info

# 向后兼容的函数
def create_hybrid_retriever(embedding_model: str = "bge-m3", fusion_method: str = "weighted_rrf") -> HybridRetriever:
    """创建混合检索器（向后兼容）"""
    return HybridRetriever(embedding_model, fusion_method)

if __name__ == "__main__":
    # 测试混合检索器
    print("🔧 测试混合检索器...")

    # 创建测试文档
    test_documents = [
        {'id': 1, 'title': '机器学习', 'text': '机器学习是人工智能的一个分支，通过数据学习模式。'},
        {'id': 2, 'title': '深度学习', 'text': '深度学习是机器学习的一个子集，使用多层神经网络。'},
        {'id': 3, 'title': '自然语言处理', 'text': '自然语言处理是人工智能的重要应用领域。'}
    ]

    # 创建混合检索器
    retriever = HybridRetriever(fusion_method="weighted_rrf")

    print(f"✅ 混合检索器创建成功: {retriever.retriever_type}")
    print(f"✅ 融合方法: {retriever.fusion_method}")
    print(f"✅ 支持的功能: {retriever.get_supported_features()}")
    print(f"✅ 配置信息: {retriever.get_config_info()}")

    print("\n✅ 混合检索器测试完成")