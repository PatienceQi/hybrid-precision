"""
批次实验管理器
支持1000样本分批次处理，每批200个样本
提供中途保存、断点续跑、进度监控等功能
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

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
        self.batch_id = batch_id
        self.experiment_type = experiment_type
        self.batch_size = batch_size
        self.start_time = time.time()
        self.processed_samples = 0
        self.current_sample_idx = 0
        self.error_count = 0

        # 设置文件路径
        self.script_dir = Path(__file__).parent
        self.results_dir = self.script_dir / 'batch_results'
        self.results_dir.mkdir(exist_ok=True)

        # 生成文件名
        self.base_filename = f"{experiment_type}_batch_{batch_id}_{batch_size}_samples"
        self.progress_file = self.results_dir / f"{self.base_filename}_progress.json"
        self.intermediate_results_file = self.results_dir / f"{self.base_filename}_intermediate.json"
        self.final_results_file = self.results_dir / f"{self.base_filename}_results.json"
        self.log_file = self.results_dir / f"{self.base_filename}.log"

        # 批次统计信息
        self.batch_stats = {
            'batch_id': batch_id,
            'experiment_type': experiment_type,
            'batch_size': batch_size,
            'start_time': datetime.now().isoformat(),
            'processed_samples': 0,
            'error_count': 0,
            'current_sample_idx': 0,
            'estimated_completion_time': None,
            'status': 'initialized'
        }

        # 实验结果存储
        self.results = []
        self.detailed_logs = []

        print(f"🚀 初始化批次实验管理器 - 批次{batch_id} ({experiment_type})")
        self.log_message(f"批次{batch_id}初始化完成 - 实验类型: {experiment_type}")

    def log_message(self, message: str, level: str = "INFO"):
        """记录日志信息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)

        # 写入日志文件
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"⚠️  日志写入失败: {e}")

    def load_batch_dataset(self) -> Optional[List[Dict]]:
        """加载指定批次的数据集"""
        try:
            # 构造批次文件路径
            batch_file = self.script_dir.parent / 'dataset' / f'hotpot_medium_batch_{self.batch_id}.json'

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

    def save_progress(self):
        """保存当前进度"""
        try:
            # 更新批次统计
            self.batch_stats.update({
                'processed_samples': self.processed_samples,
                'current_sample_idx': self.current_sample_idx,
                'error_count': self.error_count,
                'last_update': datetime.now().isoformat(),
                'elapsed_time': time.time() - self.start_time
            })

            # 保存进度信息
            progress_data = {
                'batch_stats': self.batch_stats,
                'status': 'running' if self.processed_samples < self.batch_size else 'completed'
            }

            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)

            self.log_message(f"💾 进度已保存 - 已处理 {self.processed_samples}/{self.batch_size} 样本")

        except Exception as e:
            self.log_message(f"⚠️  保存进度失败: {e}", "WARNING")

    def save_intermediate_results(self, force_save: bool = False):
        """保存中间结果"""
        try:
            # 每10个样本保存一次，或强制保存
            if not force_save and self.processed_samples % 10 != 0:
                return

            intermediate_data = {
                'batch_id': self.batch_id,
                'experiment_type': self.experiment_type,
                'processed_samples': self.processed_samples,
                'results': self.results.copy(),  # 创建副本避免并发问题
                'detailed_logs': self.detailed_logs.copy(),
                'last_save_time': datetime.now().isoformat()
            }

            with open(self.intermediate_results_file, 'w', encoding='utf-8') as f:
                json.dump(intermediate_data, f, indent=2, ensure_ascii=False)

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
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)

            batch_stats = progress_data.get('batch_stats', {})
            status = progress_data.get('status', 'unknown')

            if status == 'completed':
                self.log_message("✅ 该批次实验已完成，跳过处理")
                return True

            # 恢复进度
            self.processed_samples = batch_stats.get('processed_samples', 0)
            self.current_sample_idx = batch_stats.get('current_sample_idx', 0)
            self.error_count = batch_stats.get('error_count', 0)

            # 加载中间结果
            if self.intermediate_results_file.exists():
                with open(self.intermediate_results_file, 'r', encoding='utf-8') as f:
                    intermediate_data = json.load(f)
                    self.results = intermediate_data.get('results', [])
                    self.detailed_logs = intermediate_data.get('detailed_logs', [])

            self.log_message(f"📊 恢复进度 - 已处理 {self.processed_samples} 样本，从样本 {self.current_sample_idx + 1} 继续")
            return True

        except Exception as e:
            self.log_message(f"⚠️  加载进度失败: {e}，重新开始处理", "WARNING")
            return False

    def add_sample_result(self, sample_idx: int, result: Dict, log_entry: Dict):
        """添加单个样本的结果"""
        self.results.append(result)
        self.detailed_logs.append(log_entry)
        self.processed_samples += 1
        self.current_sample_idx = sample_idx

        # 每10个样本保存一次进度和中间结果
        if self.processed_samples % 10 == 0:
            self.save_progress()
            self.save_intermediate_results()

    def add_error_result(self, sample_idx: int, error: str, context: str = ""):
        """添加错误结果"""
        self.error_count += 1
        error_result = {
            'sample_idx': sample_idx,
            'error': error,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }

        # 根据实验类型创建默认错误结果
        if self.experiment_type == 'baseline':
            error_metrics = {
                'context_precision': 0.0,
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_recall': 0.0
            }
        elif self.experiment_type == 'hybrid_standard':
            error_metrics = {
                'context_precision': 0.0,
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_recall': 0.0
            }
        else:  # hybrid_precision
            error_metrics = {
                'hybrid_context_precision': 0.0,
                'avg_hybrid_score': 0.0
            }

        self.results.append(error_metrics)

        error_log = {
            'sample_id': sample_idx + 1,
            'error': error,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.detailed_logs.append(error_log)

        self.log_message(f"❌ 样本 {sample_idx + 1} 处理失败: {error}", "ERROR")

    def get_estimated_completion_time(self) -> Optional[str]:
        """估算剩余完成时间"""
        if self.processed_samples == 0:
            return None

        elapsed_time = time.time() - self.start_time
        samples_per_second = self.processed_samples / elapsed_time
        remaining_samples = self.batch_size - self.processed_samples

        if samples_per_second > 0:
            remaining_time = remaining_samples / samples_per_second
            return f"{remaining_time / 60:.1f} 分钟"
        else:
            return None

    def finalize_batch(self, summary_stats: Dict):
        """完成批次处理"""
        try:
            # 更新最终统计
            completion_time = time.time() - self.start_time
            self.batch_stats.update({
                'processed_samples': self.processed_samples,
                'error_count': self.error_count,
                'completion_time': completion_time,
                'status': 'completed',
                'completion_timestamp': datetime.now().isoformat()
            })

            # 保存最终完整结果
            final_data = {
                'batch_info': self.batch_stats,
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

            with open(self.final_results_file, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)

            # 保存最终进度
            self.save_progress()

            # 清理中间文件（可选）
            # if self.intermediate_results_file.exists():
            #     self.intermediate_results_file.unlink()

            self.log_message(f"🎯 批次 {self.batch_id} 处理完成!")
            self.log_message(f"📊 总计处理 {self.processed_samples} 样本，错误 {self.error_count} 个")
            self.log_message(f"⏱️  总用时: {completion_time / 60:.1f} 分钟")
            self.log_message(f"💾 最终结果保存到: {self.final_results_file}")

            return final_data

        except Exception as e:
            self.log_message(f"❌ 完成批次处理失败: {e}", "ERROR")
            return None

    def get_batch_status(self) -> Dict:
        """获取批次状态信息"""
        return {
            'batch_id': self.batch_id,
            'experiment_type': self.experiment_type,
            'processed_samples': self.processed_samples,
            'total_samples': self.batch_size,
            'progress_percentage': (self.processed_samples / self.batch_size) * 100,
            'error_count': self.error_count,
            'estimated_completion_time': self.get_estimated_completion_time(),
            'status': self.batch_stats.get('status', 'unknown')
        }