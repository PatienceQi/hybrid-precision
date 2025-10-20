import json
import numpy as np
import os
from typing import List, Dict
from retriever import get_embeddings, load_or_generate_embeddings

def retrieve_documents(
    query: str, 
    documents: List[Dict], 
    embeddings: List[List[float]], 
    top_k: int = 5
) -> List[Dict]:
    """
    根据查询检索最相关的文档
    
    Args:
        query: 查询文本
        documents: 知识库文档列表
        embeddings: 文档嵌入向量
        top_k: 返回的文档数量
        
    Returns:
        检索到的文档列表
    """
    # 生成查询嵌入
    query_embedding = get_embeddings(query)
    
    # 计算文档范数
    doc_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    
    # 计算点积
    dot_products = np.dot(embeddings, query_embedding)
    
    # 计算相似度，避免零范数除法
    similarities = np.zeros_like(dot_products)
    mask = doc_norms > 0
    similarities[mask] = dot_products[mask] / (doc_norms[mask] * query_norm)
    similarities[~mask] = -1.0  # 空文档的相似度设为低值
    
    # 获取top-k文档
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [documents[i] for i in top_indices]

def generate_response(
    query: str,
    retrieved_docs: List[Dict],
    context: List[str]
) -> Dict:
    """
    生成响应结果，适配RAGAS评估需求

    Args:
        query: 查询文本
        retrieved_docs: 检索到的文档
        context: 真实上下文

    Returns:
        包含检索结果和文档内容的字典，供RAGAS评估使用
    """
    return {
        'question': query,
        'answer': '',  # 由后续LLM生成
        'contexts': [doc['text'] for doc in retrieved_docs],
        'ground_truth': context
    }

def generate_hybrid_response(
    query: str,
    retrieved_docs: List[Dict],
    context: List[str],
    fusion_method: str = "unknown"
) -> Dict:
    """
    生成增强的响应结果，支持混合检索优化提示

    Args:
        query: 查询文本
        retrieved_docs: 检索到的文档（包含混合检索信息）
        context: 真实上下文
        fusion_method: 融合方法名称

    Returns:
        包含检索结果和文档内容的字典，供RAGAS评估使用
        包含混合检索元数据用于LLM提示优化
    """
    # 构建增强的LLM提示，考虑混合检索特性
    hybrid_metadata = {
        "retrieval_method": "hybrid_search",
        "fusion_method": fusion_method,
        "retrieval_strategy": f"Consider hybrid search blending semantic and keyword relevance",
        "context_count": len(retrieved_docs),
        "fusion_optimized": True
    }

    # 提取文档文本用于RAGAS评估
    contexts = []
    for doc in retrieved_docs:
        text = doc.get('text', '')
        # 添加混合检索提示到每个上下文的开头
        if doc.get('fusion_method') == 'cascading':
            enhanced_text = f"[Cascading Hybrid Retrieval] {text}"
        elif doc.get('fusion_method') == 'weighted_rrf':
            enhanced_text = f"[Weighted RRF Fusion] {text}"
        elif doc.get('fusion_method') == 'linear_weighted':
            enhanced_text = f"[Linear Weighted Fusion] {text}"
        else:
            enhanced_text = text
        contexts.append(enhanced_text)

    return {
        'question': query,
        'answer': '',  # 由后续LLM生成
        'contexts': contexts,
        'ground_truth': context,
        'hybrid_metadata': hybrid_metadata,
        'fusion_method': fusion_method
    }

if __name__ == "__main__":
    # 测试生成功能
    from retriever import load_or_generate_embeddings
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    documents_file = os.path.join(script_dir, 'knowledge_base', 'documents.json')
    
    with open(documents_file, 'r') as f:
        documents = json.load(f)
    
    embeddings = load_or_generate_embeddings(
        os.path.join(script_dir, 'knowledge_base')
    )
    
    test_query = "What is the capital of France?"
    retrieved = retrieve_documents(test_query, documents, embeddings)
    print(f"Retrieved documents: {[doc['title'] for doc in retrieved]}")