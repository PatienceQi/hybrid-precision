"""
通用工具函数
提供日志设置、数据验证、重试机制等通用功能
"""

import os
import json
import time
import logging
import functools
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
import numpy as np

def setup_logging(level: str = "INFO", log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    设置日志配置

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，如果为None则不写入文件
        format_string: 日志格式字符串

    Returns:
        配置好的logger实例
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 设置日志级别
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # 配置根日志器
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=[]
    )

    # 创建logger
    logger = logging.getLogger(__name__)

    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 添加文件处理器（如果指定了日志文件）
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.setLevel(numeric_level)
    logger.info(f"日志系统初始化完成 - 级别: {level}")
    if log_file:
        logger.info(f"日志文件: {log_file}")

    return logger

def validate_data(data: Any, data_type: type, allow_none: bool = False,
                 min_length: Optional[int] = None, max_length: Optional[int] = None,
                 custom_validator: Optional[Callable] = None) -> bool:
    """
    验证数据有效性

    Args:
        data: 要验证的数据
        data_type: 期望的数据类型
        allow_none: 是否允许None值
        min_length: 最小长度（对于字符串、列表等）
        max_length: 最大长度（对于字符串、列表等）
        custom_validator: 自定义验证函数

    Returns:
        数据是否有效
    """
    # 检查None值
    if data is None:
        return allow_none

    # 检查类型
    if not isinstance(data, data_type):
        return False

    # 检查长度
    if hasattr(data, '__len__'):
        length = len(data)
        if min_length is not None and length < min_length:
            return False
        if max_length is not None and length > max_length:
            return False

    # 自定义验证
    if custom_validator is not None:
        try:
            return custom_validator(data)
        except Exception:
            return False

    return True

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0,
                    exceptions: tuple = (Exception,), logger: Optional[logging.Logger] = None):
    """
    重试装饰器

    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 退避因子
        exceptions: 需要重试的异常类型
        logger: 日志记录器

    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        if logger:
                            logger.warning(f"函数 {func.__name__} 失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                            logger.info(f"等待 {current_delay} 秒后重试...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        if logger:
                            logger.error(f"函数 {func.__name__} 最终失败，已重试 {max_retries} 次")

            # 如果所有重试都失败，抛出最后的异常
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(f"函数 {func.__name__} 执行失败")

        return wrapper
    return decorator

def load_json_file(file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    安全加载JSON文件

    Args:
        file_path: JSON文件路径
        encoding: 文件编码

    Returns:
        JSON数据字典

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON解析失败
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON解析失败: {file_path}", e.doc, e.pos)

def save_json_file(data: Dict[str, Any], file_path: str, encoding: str = 'utf-8',
                  indent: int = 2, ensure_ascii: bool = False) -> None:
    """
    安全保存JSON文件

    Args:
        data: 要保存的数据
        file_path: 保存路径
        encoding: 文件编码
        indent: JSON缩进
        ensure_ascii: 是否确保ASCII输出
    """
    # 确保目录存在
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    try:
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    except Exception as e:
        raise IOError(f"保存JSON文件失败: {file_path} - {e}")

def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算两个向量的余弦相似度

    Args:
        vec1: 向量1
        vec2: 向量2

    Returns:
        余弦相似度 (-1 到 1)
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"向量维度不匹配: {len(vec1)} vs {len(vec2)}")

    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)

    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1_np, vec2_np) / (norm1 * norm2))

def normalize_text(text: str) -> str:
    """
    标准化文本

    Args:
        text: 输入文本

    Returns:
        标准化后的文本
    """
    if not text:
        return ""

    # 转换为小写
    text = text.lower()

    # 移除多余空白字符
    text = ' '.join(text.split())

    # 移除标点符号（可选）
    # import re
    # text = re.sub(r'[^\w\s]', '', text)

    return text.strip()

def extract_keywords(text: str, min_length: int = 3, top_k: Optional[int] = None) -> List[str]:
    """
    提取关键词

    Args:
        text: 输入文本
        min_length: 关键词最小长度
        top_k: 返回前k个关键词，如果为None则返回所有

    Returns:
        关键词列表
    """
    if not text:
        return []

    # 标准化文本
    normalized_text = normalize_text(text)

    # 分词
    words = normalized_text.split()

    # 过滤短词
    keywords = [word for word in words if len(word) >= min_length]

    # 统计词频
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1

    # 按频率排序
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # 提取关键词
    if top_k is not None:
        sorted_words = sorted_words[:top_k]

    return [word for word, freq in sorted_words]

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法

    Args:
        numerator: 分子
        denominator: 分母
        default: 除数为0时的默认值

    Returns:
        除法结果
    """
    if denominator == 0:
        return default
    return numerator / denominator

def format_time(seconds: float) -> str:
    """
    格式化时间

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"

def create_progress_bar(current: int, total: int, bar_length: int = 50) -> str:
    """
    创建进度条

    Args:
        current: 当前进度
        total: 总进度
        bar_length: 进度条长度

    Returns:
        进度条字符串
    """
    if total == 0:
        return "[████████████████████] 100.0%"

    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percentage = progress * 100

    return f"[{bar}] {percentage:.1f}% ({current}/{total})"

class Timer:
    """计时器工具类"""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """开始计时"""
        self.start_time = time.time()
        self.end_time = None

    def stop(self):
        """停止计时"""
        self.end_time = time.time()

    def elapsed_time(self) -> float:
        """获取经过的时间"""
        if self.start_time is None:
            return 0.0

        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if self.end_time is None:
            self.stop()

class DataCache:
    """数据缓存工具类"""

    def __init__(self, cache_dir: str = "cache", max_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self._cache = {}

    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        if key in self._cache:
            return self._cache[key]

        # 尝试从文件加载
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._cache[key] = data
                    return data
            except Exception as e:
                logging.warning(f"加载缓存失败: {key} - {e}")

        return None

    def set(self, key: str, value: Any) -> None:
        """设置缓存数据"""
        self._cache[key] = value

        # 保存到文件
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"保存缓存失败: {key} - {e}")

        # 清理过期缓存
        if len(self._cache) > self.max_size:
            self._cleanup_cache()

    def _cleanup_cache(self):
        """清理缓存"""
        # 简单的LRU清理策略
        keys = list(self._cache.keys())
        remove_count = len(keys) - self.max_size + 100  # 保留一些空间

        for key in keys[:remove_count]:
            del self._cache[key]
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                cache_file.unlink()