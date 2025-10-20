"""
简单知识库构建器 - 极简API设计
让用户在2分钟内完成知识库构建
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

from core.utils import load_json_file, save_json_file, setup_logging
from retrievers import HybridRetriever

logger = logging.getLogger(__name__)

class SimpleKnowledgeBuilder:
    """简单知识库构建器 - 用户友好的知识库构建工具"""

    def __init__(self, data_dir: str = "knowledge_data"):
        """
        初始化知识库构建器

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.knowledge_file = self.data_dir / "knowledge_base.json"
        self.index_file = self.data_dir / "knowledge_index.json"

        # 设置日志
        self.logger = setup_logging(level="INFO")

    def build_from_hotpotqa(self, dataset_file: str) -> bool:
        """
        从HotPotQA数据集构建知识库

        Args:
            dataset_file: HotPotQA数据文件路径

        Returns:
            是否成功构建
        """
        self.logger.info("📚 从HotPotQA数据构建知识库...")

        try:
            # 加载HotPotQA数据
            self.logger.info(f"加载数据文件: {dataset_file}")
            data_file_content = load_json_file(dataset_file)

            if not data_file_content:
                self.logger.error("无法加载数据文件")
                return False

            # 检查数据结构 - 可能是 {'metadata': ..., 'samples': [...]} 或者直接是列表
            if isinstance(data_file_content, dict) and 'samples' in data_file_content:
                # 新格式：{'metadata': ..., 'samples': [...]}
                data = data_file_content['samples']
                self.logger.info(f"检测到结构化数据格式，样本数: {len(data)}")
            elif isinstance(data_file_content, list):
                # 旧格式：直接是样本列表
                data = data_file_content
                self.logger.info(f"检测到列表数据格式，样本数: {len(data)}")
            else:
                self.logger.error(f"未知的数据格式: {type(data_file_content)}")
                return False

            # 提取所有上下文文档
            documents = []
            doc_id = 0

            self.logger.info("提取文档内容...")
            for i, sample in enumerate(data):
                if i % 50 == 0:
                    print(f"进度: {i}/{len(data)} 样本处理中...", end="\r")

                # 处理HotPotQA的context格式
                hotpot_context = sample.get('context', [])

                for ctx_item in hotpot_context:
                    if isinstance(ctx_item, list) and len(ctx_item) >= 2:
                        # ctx_item格式: [标题, 文本列表]
                        title, text_list = ctx_item[0], ctx_item[1]

                        if isinstance(text_list, list):
                            # 将文本段落合并
                            full_text = ' '.join(text_list)
                        else:
                            full_text = str(text_list)

                        documents.append({
                            'id': doc_id,
                            'title': title,
                            'content': full_text,
                            'source': 'hotpotqa',
                            'metadata': {
                                'sample_index': i,
                                'length': len(full_text)
                            }
                        })
                        doc_id += 1

            print(f"\n✅ 提取完成，共{len(documents)}个文档")

            # 保存知识库
            return self._save_knowledge_base(documents)

        except Exception as e:
            self.logger.error(f"构建知识库失败: {e}")
            return False

    def build_from_wikipedia(self, topics: List[str], max_pages: int = 5) -> bool:
        """
        从维基百科构建知识库

        Args:
            topics: 主题词列表
            max_pages: 每个主题最大页面数

        Returns:
            是否成功构建
        """
        if not WIKIPEDIA_AVAILABLE:
            self.logger.error("维基百科库不可用，请先安装: pip install wikipedia")
            return False

        self.logger.info(f"📚 从维基百科构建知识库，主题: {topics}")

        try:
            documents = []
            doc_id = 0

            # 设置维基百科语言为中文
            wikipedia.set_lang("zh")

            for topic in topics:
                self.logger.info(f"搜索主题: {topic}")

                try:
                    # 搜索相关页面
                    search_results = wikipedia.search(topic, results=max_pages)

                    for page_title in search_results[:max_pages]:
                        try:
                            # 获取页面内容
                            page = wikipedia.page(page_title)

                            # 提取摘要和正文
                            content = page.content

                            # 如果内容太长，截取前部分
                            if len(content) > 5000:
                                content = content[:5000] + "..."

                            documents.append({
                                'id': doc_id,
                                'title': page_title,
                                'content': content,
                                'source': 'wikipedia',
                                'metadata': {
                                    'url': page.url,
                                    'topic': topic,
                                    'length': len(content)
                                }
                            })
                            doc_id += 1

                            # 显示进度
                            print(f"获取页面: {page_title} ({len(content)}字符)")

                        except wikipedia.exceptions.DisambiguationError:
                            self.logger.warning(f"页面{page_title}存在歧义，跳过")
                            continue
                        except wikipedia.exceptions.PageError:
                            self.logger.warning(f"页面{page_title}不存在，跳过")
                            continue

                except Exception as e:
                    self.logger.error(f"搜索主题{topic}失败: {e}")
                    continue

            if not documents:
                self.logger.error("未能获取任何维基百科文档")
                return False

            print(f"✅ 维基百科数据获取完成，共{len(documents)}个文档")
            return self._save_knowledge_base(documents)

        except Exception as e:
            self.logger.error(f"维基百科构建失败: {e}")
            return False

    def build_from_file(self, file_path: str) -> bool:
        """
        从本地文件构建知识库

        Args:
            file_path: 文本文件路径

        Returns:
            是否成功构建
        """
        self.logger.info(f"📚 从文件构建知识库: {file_path}")

        try:
            file_path = Path(file_path)

            if not file_path.exists():
                self.logger.error(f"文件不存在: {file_path}")
                return False

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 简单的文本分块（按段落）
            paragraphs = content.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

            documents = []
            for i, paragraph in enumerate(paragraphs):
                documents.append({
                    'id': i,
                    'title': f"文档段落_{i+1}",
                    'content': paragraph,
                    'source': 'local_file',
                    'metadata': {
                        'file_path': str(file_path),
                        'paragraph_index': i,
                        'length': len(paragraph)
                    }
                })

            print(f"✅ 文件处理完成，共{len(documents)}个段落")
            return self._save_knowledge_base(documents)

        except Exception as e:
            self.logger.error(f"文件构建失败: {e}")
            return False

    def _save_knowledge_base(self, documents: List[Dict]) -> bool:
        """
        保存知识库到文件

        Args:
            documents: 文档列表

        Returns:
            是否成功保存
        """
        try:
            # 保存文档数据
            knowledge_data = {
                'metadata': {
                    'total_documents': len(documents),
                    'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'sources': list(set(doc.get('source', 'unknown') for doc in documents))
                },
                'documents': documents
            }

            save_json_file(knowledge_data, str(self.knowledge_file))
            self.logger.info(f"✅ 知识库已保存: {self.knowledge_file}")

            # 构建向量索引（简化版）
            return self._build_simple_index(documents)

        except Exception as e:
            self.logger.error(f"保存知识库失败: {e}")
            return False

    def _build_simple_index(self, documents: List[Dict]) -> bool:
        """
        构建简单的向量索引

        Args:
            documents: 文档列表

        Returns:
            是否成功构建索引
        """
        try:
            self.logger.info("构建向量索引...")

            # 这里可以集成更复杂的向量化逻辑
            # 目前先保存基本的索引信息
            index_data = {
                'total_docs': len(documents),
                'indexed_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'document_ids': [doc['id'] for doc in documents]
            }

            save_json_file(index_data, str(self.index_file))
            self.logger.info("✅ 索引构建完成")
            return True

        except Exception as e:
            self.logger.error(f"构建索引失败: {e}")
            return False

    def load_knowledge_base(self) -> Optional[Dict]:
        """
        加载知识库

        Returns:
            知识库数据或None
        """
        try:
            if not self.knowledge_file.exists():
                self.logger.warning("知识库文件不存在")
                return None

            return load_json_file(str(self.knowledge_file))

        except Exception as e:
            self.logger.error(f"加载知识库失败: {e}")
            return None

    def get_document_count(self) -> int:
        """获取文档数量"""
        kb_data = self.load_knowledge_base()
        return kb_data.get('metadata', {}).get('total_documents', 0) if kb_data else 0

    def get_knowledge_info(self) -> Dict[str, Any]:
        """获取知识库信息"""
        kb_data = self.load_knowledge_base()
        if not kb_data:
            return {'status': 'empty', 'message': '知识库为空'}

        metadata = kb_data.get('metadata', {})
        return {
            'status': 'ready',
            'total_documents': metadata.get('total_documents', 0),
            'sources': metadata.get('sources', []),
            'created_time': metadata.get('created_time', 'unknown'),
            'file_path': str(self.knowledge_file)
        }