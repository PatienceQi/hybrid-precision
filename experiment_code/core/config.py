"""
统一配置管理器
提供集中式的配置管理，支持环境变量、配置文件和默认值
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """API配置"""
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "gpt-3.5-turbo"
    client_type: str = "auto"
    max_retries: int = 3
    retry_delay: int = 2
    timeout: int = 30
    skip_connection_test: bool = False
    force_real_api: bool = False
    allow_mock_fallback: bool = True
    default_headers: Dict[str, str] = field(default_factory=lambda: {
        "HTTP-Referer": "https://ragas-hybrid-evaluation.com",
        "X-Title": "RAGAS Hybrid Evaluation"
    })

@dataclass
class EvaluationConfig:
    """评估配置"""
    batch_size: int = 200
    use_mock_api: bool = False
    metrics: list = field(default_factory=lambda: [
        "context_precision", "faithfulness", "answer_relevancy", "context_recall"
    ])
    hybrid_metrics: list = field(default_factory=lambda: [
        "hybrid_context_precision", "avg_hybrid_score"
    ])

@dataclass
class RetrievalConfig:
    """检索配置"""
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    top_k: int = 5
    similarity_threshold: float = 0.7
    cache_embeddings: bool = True
    embedding_service_url: str = "https://wolfai.top/v1/embeddings"
    embedding_api_key: Optional[str] = None
    force_embedding_service: bool = False
    fallback_to_local_embeddings: bool = True

@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_types: list = field(default_factory=lambda: [
        "baseline", "hybrid_standard", "hybrid_precision"
    ])
    num_batches: int = 5
    save_interval: int = 10
    log_level: str = "INFO"
    results_dir: str = "batch_results"

class Config:
    """统一配置管理器"""

    def __init__(self, config_file: Optional[str] = None):
        """初始化配置管理器"""
        self.config_file = config_file or self._find_config_file()
        self._load_env_file()
        self._init_configs()
        self._load_config_file()
        self._validate_configs()

    def _find_config_file(self) -> Optional[str]:
        """查找配置文件"""
        possible_paths = [
            "config.json",
            "experiment_code/config.json",
            "../config.json",
            "../../config.json"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def _load_env_file(self):
        """加载环境变量文件"""
        env_paths = [".env", "experiment_code/.env", "../.env"]
        for env_path in env_paths:
            if os.path.exists(env_path):
                load_dotenv(env_path)
                logger.info(f"加载环境变量文件: {env_path}")
                break

    def _init_configs(self):
        """初始化各个配置模块"""
        self.api = self._init_api_config()
        self.evaluation = self._init_evaluation_config()
        self.retrieval = self._init_retrieval_config()
        self.experiment = self._init_experiment_config()

    def _init_api_config(self) -> APIConfig:
        """初始化API配置"""
        api_key = self._get_api_key()
        base_url = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

        return APIConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            client_type=os.getenv("API_CLIENT_TYPE", "auto"),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=int(os.getenv("RETRY_DELAY", "2")),
            timeout=int(os.getenv("API_TIMEOUT", "30")),
            skip_connection_test=os.getenv("SKIP_API_CONNECTION_TEST", "false").lower() == "true",
            force_real_api=os.getenv("FORCE_REAL_API", "false").lower() == "true",
            allow_mock_fallback=os.getenv("ALLOW_MOCK_FALLBACK", "true").lower() == "true"
        )

    def _init_evaluation_config(self) -> EvaluationConfig:
        """初始化评估配置"""
        return EvaluationConfig(
            batch_size=int(os.getenv("BATCH_SIZE", "200")),
            use_mock_api=os.getenv("USE_MOCK_API", "false").lower() == "true",
            metrics=self._parse_list_env("EVALUATION_METRICS", ["context_precision", "faithfulness", "answer_relevancy", "context_recall"]),
            hybrid_metrics=self._parse_list_env("HYBRID_METRICS", ["hybrid_context_precision", "avg_hybrid_score"])
        )

    def _init_retrieval_config(self) -> RetrievalConfig:
        """初始化检索配置"""
        service_url_env = os.getenv("EMBEDDING_SERVICE_URL")
        if service_url_env:
            service_url = service_url_env.strip()
        else:
            service_url = "https://wolfai.top/v1/embeddings"

        return RetrievalConfig(
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "3072")),
            top_k=int(os.getenv("TOP_K", "5")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
            cache_embeddings=os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true",
            embedding_service_url=service_url,
            embedding_api_key=os.getenv("EMBEDDING_API_KEY"),
            force_embedding_service=os.getenv("FORCE_EMBEDDING_SERVICE", "false").lower() == "true",
            fallback_to_local_embeddings=os.getenv("EMBEDDING_FALLBACK_LOCAL", "true").lower() == "true"
        )

    def _init_experiment_config(self) -> ExperimentConfig:
        """初始化实验配置"""
        return ExperimentConfig(
            experiment_types=self._parse_list_env("EXPERIMENT_TYPES", ["baseline", "hybrid_standard", "hybrid_precision"]),
            num_batches=int(os.getenv("NUM_BATCHES", "5")),
            save_interval=int(os.getenv("SAVE_INTERVAL", "10")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            results_dir=os.getenv("RESULTS_DIR", "batch_results")
        )

    def _get_api_key(self) -> Optional[str]:
        """获取API密钥，尝试多种方式"""
        # 方法1：环境变量
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key.strip():
            logger.info("使用环境变量中的API密钥")
            return api_key.strip()

        # 方法2：检查其他可能的变量名
        alt_names = ["OPENROUTER_API_KEY", "API_KEY", "LLM_API_KEY"]
        for name in alt_names:
            api_key = os.getenv(name)
            if api_key and api_key.strip():
                logger.info(f"使用{name}作为API密钥")
                return api_key.strip()

        logger.warning("未找到有效的API密钥")
        return None

    def _parse_list_env(self, env_name: str, default: list) -> list:
        """解析列表类型的环境变量"""
        env_value = os.getenv(env_name)
        if env_value:
            return [item.strip() for item in env_value.split(",")]
        return default

    def _load_config_file(self):
        """加载配置文件"""
        if not self.config_file or not os.path.exists(self.config_file):
            return

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # 更新配置
            if 'api' in config_data:
                self._update_config_from_dict(self.api, config_data['api'])
            if 'evaluation' in config_data:
                self._update_config_from_dict(self.evaluation, config_data['evaluation'])
            if 'retrieval' in config_data:
                self._update_config_from_dict(self.retrieval, config_data['retrieval'])
            if 'experiment' in config_data:
                self._update_config_from_dict(self.experiment, config_data['experiment'])

            logger.info(f"加载配置文件: {self.config_file}")

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")

    def _update_config_from_dict(self, config_obj: Any, config_dict: Dict):
        """从字典更新配置对象"""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)

    def _validate_configs(self):
        """验证配置有效性"""
        # 验证API配置
        if not self.api.api_key and not self.evaluation.use_mock_api:
            logger.warning("未配置API密钥且未启用模拟模式")

        # 验证评估配置
        if self.evaluation.batch_size <= 0:
            raise ValueError("批次大小必须大于0")

        # 验证检索配置
        if self.retrieval.top_k <= 0:
            raise ValueError("top_k必须大于0")
        if not (0 <= self.retrieval.similarity_threshold <= 1):
            raise ValueError("相似度阈值必须在0-1之间")

        logger.info("配置验证通过")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'api': self.api.__dict__,
            'evaluation': self.evaluation.__dict__,
            'retrieval': self.retrieval.__dict__,
            'experiment': self.experiment.__dict__
        }

    def save_config(self, file_path: str):
        """保存配置到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise

# 全局配置实例
_config_instance = None

def get_config(config_file: Optional[str] = None) -> Config:
    """获取全局配置实例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_file)
    return _config_instance

def reload_config(config_file: Optional[str] = None):
    """重新加载配置"""
    global _config_instance
    _config_instance = Config(config_file)
    return _config_instance
