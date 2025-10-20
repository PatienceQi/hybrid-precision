"""
批次实验管理器 - 重构版
支持1000样本分批次处理，每批200个样本
提供中途保存、断点续跑、进度监控等功能
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

from core.config import get_config
from core.utils import setup_logging, save_json_file, load_json_file, format_time, create_progress_bar

logger = logging.getLogger(__name__)

@dataclass
class BatchStats:
    """批次统计信息"""
    batch_id: int
    experiment_type: str
    batch_size: int
    start_time: str
    processed_samples: int = 0
    error_count: int = 0
    current_sample_idx: int = 0
    estimated_completion_time: Optional[str] = None
    status: str = "initialized"
    last_update: Optional[str] = None
    elapsed_time: float = 0.0
    completion_time: Optional[float] = None
    completion_timestamp: Optional[str] = None

class BatchExperimentManager:
    """批次实验管理器"""

    def __init__(self, batch_id: int, experiment_type: str, batch_size: int = 200):
        """
        初始化批次实验管理器

        Args:
            batch_id: 批次ID (1-5)
            experiment_type: 实验类型 ('baseline', 'hybrid_standard', 'hybrid_precision')
            batch_size: 批次大小，默认200
        """
        self.config = get_config()
        self.batch_id = batch_id
        self.experiment_type = experiment_type
        self.batch_size = batch_size
        self.start_time = time.time()
        self.processed_samples = 0
        self.current_sample_idx = 0
        self.error_count = 0

        # 设置文件路径
        self.script_dir = Path(__file__).parent.parent
        self.results_dir = self.script_dir / self.config.experiment.results_dir
        self.results_dir.mkdir(exist_ok=True)

        # 生成文件名
        self.base_filename = f"{experiment_type}_batch_{batch_id}_{batch_size}_samples"
        self.progress_file = self.results_dir / f"{self.base_filename}_progress.json"
        self.intermediate_results_file = self.results_dir / f"{self.base_filename}_intermediate.json"
        self.final_results_file = self.results_dir / f"{self.base_filename}_results.json"
        self.log_file = self.results_dir / f"{self.base_filename}.log"

        # 设置日志
        self.logger = setup_logging(
            level=self.config.experiment.log_level,
            log_file=str(self.log_file)
        )

        # 批次统计信息
        self.batch_stats = BatchStats(
            batch_id=batch_id,
            experiment_type=experiment_type,
            batch_size=batch_size,
            start_time=datetime.now().isoformat()
        )

        # 实验结果存储
        self.results = []
        self.detailed_logs = []

        self.log_message(f"🚀 初始化批次实验管理器 - 批次{batch_id} ({experiment_type})")

    def log_message(self, message: str, level: str = "INFO"):
        """记录日志信息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        self.logger.info(message)

    def load_batch_dataset(self) -> Optional[List[Dict]]:
        """加载指定批次的数据集"""
        try:
            # 构造批次文件路径
            batch_file = self.script_dir / 'dataset' / f'hotpot_medium_batch_{self.batch_id}.json'

            if not batch_file.exists():
                self.log_message(f"❌ 批次文件不存在: {batch_file}", "ERROR")
                return None

            self.log_message(f"📁 加载批次数据集: {batch_file}")

            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)

            samples = batch_data.get('samples', [])
            self.log_message(f"✅ 成功加载 {len(samples)} 个样本")
            return samples

        except Exception as e:
            self.log_message(f"❌ 加载批次数据集失败: {e}", "ERROR")
            return None

    def save_progress(self) -> None:
        """保存当前进度"""
        try:
            # 更新批次统计
            self.batch_stats.processed_samples = self.processed_samples
            self.batch_stats.current_sample_idx = self.current_sample_idx
            self.batch_stats.error_count = self.error_count
            self.batch_stats.last_update = datetime.now().isoformat()
            self.batch_stats.elapsed_time = time.time() - self.start_time
            self.batch_stats.status = 'running' if self.processed_samples < self.batch_size else 'completed'

            # 保存进度信息
            progress_data = {
                'batch_stats': asdict(self.batch_stats),
                'status': self.batch_stats.status
            }

            save_json_file(progress_data, str(self.progress_file))
            self.log_message(f"💾 进度已保存 - 已处理 {self.processed_samples}/{self.batch_size} 样本")

        except Exception as e:
            self.log_message(f"⚠️  保存进度失败: {e}", "WARNING")

    def save_intermediate_results(self, force_save: bool = False) -> None:
        """保存中间结果"""
        try:
            # 每10个样本保存一次，或强制保存
            if not force_save and self.processed_samples % self.config.experiment.save_interval != 0:
                return

            intermediate_data = {
                'batch_id': self.batch_id,
                'experiment_type': self.experiment_type,
                'processed_samples': self.processed_samples,
                'results': self.results.copy(),
                'detailed_logs': self.detailed_logs.copy(),
                'last_save_time': datetime.now().isoformat()
            }

            save_json_file(intermediate_data, str(self.intermediate_results_file))
            self.log_message(f"💾 中间结果已保存 - 样本 {self.processed_samples}")

        except Exception as e:
            self.log_message(f"⚠️  保存中间结果失败: {e}", "WARNING")

    def load_previous_progress(self) -> bool:
        """加载之前的进度"""
        try:
            if not self.progress_file.exists():
                self.log_message("ℹ️  未发现之前的进度文件，从第一个样本开始处理")
                return False

            # 加载进度信息
            progress_data = load_json_file(str(self.progress_file))

            batch_stats_data = progress_data.get('batch_stats', {})
            status = progress_data.get('status', 'unknown')

            if status == 'completed':
                self.log_message("✅ 该批次实验已完成，跳过处理")
                return True

            # 恢复批次统计
            self.batch_stats = BatchStats(**batch_stats_data)
            self.processed_samples = self.batch_stats.processed_samples
            self.current_sample_idx = self.batch_stats.current_sample_idx
            self.error_count = self.batch_stats.error_count

            # 加载中间结果
            if self.intermediate_results_file.exists():
                intermediate_data = load_json_file(str(self.intermediate_results_file))
                self.results = intermediate_data.get('results', [])
                self.detailed_logs = intermediate_data.get('detailed_logs', [])

            self.log_message(f"📊 恢复进度 - 已处理 {self.processed_samples} 样本，从样本 {self.current_sample_idx + 1} 继续")
            return True

        except Exception as e:
            self.log_message(f"⚠️  加载进度失败: {e}，重新开始处理", "WARNING")
            return False

    def add_sample_result(self, sample_idx: int, result: Dict, log_entry: Dict) -> None:
        """添加单个样本的结果"""
        self.results.append(result)
        self.detailed_logs.append(log_entry)
        self.processed_samples += 1
        self.current_sample_idx = sample_idx

        # 每10个样本保存一次进度和中间结果
        if self.processed_samples % self.config.experiment.save_interval == 0:
            self.save_progress()
            self.save_intermediate_results()

    def add_error_result(self, sample_idx: int, error: str, context: str = "") -> None:
        """添加错误结果"""
        self.error_count += 1
        error_result = {
            'sample_idx': sample_idx,
            'error': error,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }

        # 根据实验类型创建默认错误结果
        error_metrics = self._get_default_error_metrics()

        self.results.append(error_metrics)

        error_log = {
            'sample_id': sample_idx + 1,
            'error': error,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.detailed_logs.append(error_log)

        self.log_message(f"❌ 样本 {sample_idx + 1} 处理失败: {error}", "ERROR")

    def _get_default_error_metrics(self) -> Dict[str, float]:
        """获取默认错误指标"""
        if self.experiment_type == 'baseline':
            return {
                'context_precision': 0.0,
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_recall': 0.0
            }
        elif self.experiment_type == 'hybrid_standard':
            return {
                'context_precision': 0.0,
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_recall': 0.0
            }
        else:  # hybrid_precision
            return {
                'hybrid_context_precision': 0.0,
                'avg_hybrid_score': 0.0
            }

    def get_estimated_completion_time(self) -> Optional[str]:
        """估算剩余完成时间"""
        if self.processed_samples == 0:
            return None

        elapsed_time = time.time() - self.start_time
        samples_per_second = self.processed_samples / elapsed_time
        remaining_samples = self.batch_size - self.processed_samples

        if samples_per_second > 0:
            remaining_time = remaining_samples / samples_per_second
            return format_time(remaining_time)
        else:
            return None

    def finalize_batch(self, summary_stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """完成批次处理"""
        try:
            # 更新最终统计
            completion_time = time.time() - self.start_time
            self.batch_stats.completion_time = completion_time
            self.batch_stats.completion_timestamp = datetime.now().isoformat()
            self.batch_stats.status = 'completed'

            # 保存最终完整结果
            final_data = {
                'batch_info': asdict(self.batch_stats),
                'summary': summary_stats,
                'results': self.results,
                'detailed_logs': self.detailed_logs,
                'experiment_config': {
                    'batch_id': self.batch_id,
                    'experiment_type': self.experiment_type,
                    'batch_size': self.batch_size,
                    'processed_samples': self.processed_samples,
                    'error_count': self.error_count
                }
            }

            save_json_file(final_data, str(self.final_results_file))

            # 保存最终进度
            self.save_progress()

            self.log_message(f"🎯 批次 {self.batch_id} 处理完成!")
            self.log_message(f"📊 总计处理 {self.processed_samples} 样本，错误 {self.error_count} 个")
            self.log_message(f"⏱️  总用时: {format_time(completion_time)}")
            self.log_message(f"💾 最终结果保存到: {self.final_results_file}")

            return final_data

        except Exception as e:
            self.log_message(f"❌ 完成批次处理失败: {e}", "ERROR")
            return None

    def get_batch_status(self) -> Dict[str, Any]:
        """获取批次状态信息"""
        return {
            'batch_id': self.batch_id,
            'experiment_type': self.experiment_type,
            'processed_samples': self.processed_samples,
            'total_samples': self.batch_size,
            'progress_percentage': (self.processed_samples / self.batch_size) * 100,
            'error_count': self.error_count,
            'estimated_completion_time': self.get_estimated_completion_time(),
            'status': self.batch_stats.status,
            'elapsed_time': format_time(time.time() - self.start_time)
        }

    def print_progress(self) -> None:
        """打印进度信息"""
        status = self.get_batch_status()
        progress_bar = create_progress_bar(status['processed_samples'], status['total_samples'])

        print(f"\n📊 批次 {self.batch_id} 进度:")
        print(f"   {progress_bar}")
        print(f"   已处理: {status['processed_samples']}/{status['total_samples']} 样本")
        print(f"   错误数: {status['error_count']}")
        print(f"   用时: {status['elapsed_time']}")
        if status['estimated_completion_time']:
            print(f"   预计剩余时间: {status['estimated_completion_time']}")

# 向后兼容的函数
def create_batch_manager(batch_id: int, experiment_type: str, batch_size: int = 200) -> BatchExperimentManager:
    """创建批次实验管理器（向后兼容）"""
    return BatchExperimentManager(batch_id, experiment_type, batch_size)

if __name__ == "__main__":
    # 测试批次管理器
    print("🔧 测试批次实验管理器...")

    # 创建管理器
    manager = BatchExperimentManager(batch_id=1, experiment_type="baseline", batch_size=10)

    print(f"✅ 批次管理器创建成功")
    print(f"✅ 批次ID: {manager.batch_id}")
    print(f"✅ 实验类型: {manager.experiment_type}")
    print(f"✅ 批次大小: {manager.batch_size}")

    # 测试状态获取
    status = manager.get_batch_status()
    print(f"✅ 状态信息: {status}")

    print("\n✅ 批次实验管理器测试完成")