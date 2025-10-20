import json
import os
import requests
import time
from typing import List, Dict

def get_embeddings(texts: str) -> List[float]:
    """
    调用本地ollama服务获取文本嵌入向量

    Args:
        texts: 输入文本

    Returns:
        文本嵌入向量
    """
    model_url = 'http://localhost:11434/api/embeddings'
    headers = {'Content-Type': 'application/json'}

    data = {
        'model': 'bge-m3',
        'prompt': texts if isinstance(texts, str) else ' '.join(texts)
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(model_url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()['embedding']
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (attempt + 1) * 5  # 指数退避
            print(f"本地ollama服务调用失败，尝试 {attempt + 1}: {str(e)}。等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)

def load_or_generate_embeddings(knowledge_base_path: str) -> List[List[float]]:
    """
    加载或生成知识库文档的嵌入向量
    
    Args:
        knowledge_base_path: 知识库文档路径
        
    Returns:
        文档嵌入向量列表，每个元素是一个嵌入向量
    """
    documents_file = os.path.join(knowledge_base_path, 'documents.json')
    embeddings_file = os.path.join(knowledge_base_path, 'document_embeddings.json')
    
    with open(documents_file, 'r') as f:
        documents = json.load(f)
    
    # 检查是否存在缓存的嵌入
    if os.path.exists(embeddings_file):
        print("Loading embeddings from cache...")
        with open(embeddings_file, 'r') as f:
            embeddings = json.load(f)
        
        # 验证嵌入格式和维度一致性
        if not embeddings or not all(isinstance(e, list) and len(e) > 0 for e in embeddings):
            print("Invalid embedding format in cache, regenerating...")
        else:
            lengths = [len(e) for e in embeddings]
            if len(set(lengths)) != 1:
                print(f"Inconsistent embedding lengths: {set(lengths)}, regenerating...")
            else:
                print(f"Embeddings loaded successfully. Dimension: {lengths[0]}")
                return embeddings
    
    # 生成新嵌入
    print("Generating embeddings for knowledge base documents...")
    
    # 收集非空文档
    valid_indices = []
    valid_texts = []
    for i in range(len(documents)):
        text = documents[i]['text']
        if text and text.strip():
            valid_indices.append(i)
            valid_texts.append(text)

    if not valid_texts:
        raise ValueError("No valid documents with non-empty text")

    # 为有效文档生成嵌入
    print(f"Generating embeddings for {len(valid_texts)} valid documents...")
    valid_embeddings = []
    first_dim = None
    for j, text in enumerate(valid_texts, 1):
        emb = get_embeddings(text)
        if not isinstance(emb, list) or len(emb) == 0:
            orig_idx = valid_indices[j-1] + 1
            raise ValueError(f"Invalid embedding for document {orig_idx}: {emb}")
        if first_dim is None:
            first_dim = len(emb)
        elif len(emb) != first_dim:
            orig_idx = valid_indices[j-1] + 1
            raise ValueError(f"Inconsistent embedding dimension at document {orig_idx}: expected {first_dim}, got {len(emb)}")
        valid_embeddings.append(emb)
        print(f"Processed valid document {j}/{len(valid_texts)} ({j/len(valid_texts)*100:.1f}%)")

    # 为所有文档创建嵌入列表，空文档填充零向量
    kb_embeddings = [[0.0] * first_dim for _ in range(len(documents))]
    for idx, v_idx in enumerate(valid_indices):
        kb_embeddings[v_idx] = valid_embeddings[idx]

    num_empty = len(documents) - len(valid_texts)
    if num_empty > 0:
        print(f"Filled zero vectors for {num_empty} empty documents")

    # 保存嵌入到缓存文件
    print("Saving embeddings to cache...")
    with open(embeddings_file, 'w') as f:
        json.dump(kb_embeddings, f)
    
    print(f"Embeddings generated successfully. Total documents: {len(kb_embeddings)}, Valid: {len(valid_texts)}, Dimension: {first_dim}")
    return kb_embeddings

if __name__ == "__main__":
    # 测试检索功能
    embeddings = load_or_generate_embeddings(
        os.path.join(os.path.dirname(__file__), 'knowledge_base')
    )
    print(f"Loaded {len(embeddings)} document embeddings")