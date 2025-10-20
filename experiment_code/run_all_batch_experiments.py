"""
批量实验运行管理器
自动化运行所有实验的所有批次
支持并行处理、进度监控和结果汇总
"""

import os
import json
import time
import subprocess
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import argparse

class BatchExperimentRunner:
    """批量实验运行管理器"""

    def __init__(self, experiments_dir: str = ".", max_workers: int = 1):
        """
        初始化批量实验运行器

        Args:
            experiments_dir: 实验代码目录
            max_workers: 最大并行工作进程数（默认1，避免API限流）
        """
        self.experiments_dir = Path(experiments_dir)
        self.max_workers = max_workers
        self.results_dir = self.experiments_dir / 'batch_results'
        self.results_dir.mkdir(exist_ok=True)

        # 实验配置
        self.experiments = [
            {
                'name': 'Baseline Only',
                'script': 'baseline_only_experiment.py',
                'type': 'baseline'
            },
            {
                'name': 'Hybrid + Standard RAGAS',
                'script': 'experiment_2_fixed_simple_hybrid_standard_ragas.py',
                'type': 'hybrid_standard'
            },
            {
                'name': 'Hybrid + Hybrid Precision',
                'script': 'experiment_3_simple_hybrid_hybrid_precision.py',
                'type': 'hybrid_precision'
            }
        ]

        self.batch_ids = [1, 2, 3, 4, 5]  # 5个批次
        self.run_log = []
        self.start_time = None

        print("🚀 初始化批量实验运行管理器")
        print(f"📁 实验目录: {self.experiments_dir}")
        print(f"🔧 最大并行数: {max_workers}")

    def log_message(self, message: str, level: str = "INFO"):
        """记录运行日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)

        # 保存到运行日志
        self.run_log.append({
            'timestamp': timestamp,
            'level': level,
            'message': message
        })

    def run_single_batch(self, experiment: Dict, batch_id: int) -> Dict:
        """运行单个实验批次"""
        script_name = experiment['script']
        exp_type = experiment['type']

        self.log_message(f"开始运行 {experiment['name']} - 批次{batch_id}")

        try:
            # 构造命令
            cmd = [
                'python', script_name,
                '--batch-id', str(batch_id)
            ]

            # 运行命令
            result = subprocess.run(
                cmd,
                cwd=self.experiments_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )

            if result.returncode == 0:
                self.log_message(f"✅ {experiment['name']} 批次{batch_id} 运行成功")
                return {
                    'experiment': exp_type,
                    'batch_id': batch_id,
                    'status': 'success',
                    'output': result.stdout,
                    'error': result.stderr
                }
            else:
                self.log_message(f"❌ {experiment['name']} 批次{batch_id} 运行失败", "ERROR")
                return {
                    'experiment': exp_type,
                    'batch_id': batch_id,
                    'status': 'failed',
                    'output': result.stdout,
                    'error': result.stderr
                }

        except subprocess.TimeoutExpired:
            self.log_message(f"⏰ {experiment['name']} 批次{batch_id} 超时", "ERROR")
            return {
                'experiment': exp_type,
                'batch_id': batch_id,
                'status': 'timeout',
                'error': '运行超时（>1小时）'
            }
        except Exception as e:
            self.log_message(f"❌ {experiment['name']} 批次{batch_id} 异常: {e}", "ERROR")
            return {
                'experiment': exp_type,
                'batch_id': batch_id,
                'status': 'error',
                'error': str(e)
            }

    def run_experiment_all_batches(self, experiment: Dict) -> List[Dict]:
        """运行单个实验的所有批次"""
        self.log_message(f"🔄 开始运行 {experiment['name']} 的所有批次")
        results = []

        # 串行运行批次（避免API限流）
        for batch_id in self.batch_ids:
            batch_result = self.run_single_batch(experiment, batch_id)
            results.append(batch_result)

            # 批次间休息（避免API限流）
            if batch_id < self.batch_ids[-1]:
                self.log_message(f"⏱️  批次间休息60秒...")
                time.sleep(60)

        return results

    def run_all_experiments_serial(self) -> Dict:
        """串行运行所有实验（推荐，避免API限流）"""
        self.start_time = time.time()
        self.log_message("🚀 开始串行运行所有实验的所有批次")

        all_results = {}

        for experiment in self.experiments:
            exp_type = experiment['type']
            self.log_message(f"\n{'='*60}")
            self.log_message(f"📊 开始实验: {experiment['name']}")
            self.log_message(f"{'='*60}")

            try:
                exp_results = self.run_experiment_all_batches(experiment)
                all_results[exp_type] = exp_results

                # 实验间休息
                if experiment != self.experiments[-1]:
                    self.log_message(f"⏱️  实验间休息120秒...")
                    time.sleep(120)

            except Exception as e:
                self.log_message(f"❌ 实验 {experiment['name']} 运行失败: {e}", "ERROR")
                all_results[exp_type] = []

        return all_results

    def run_all_experiments_parallel(self) -> Dict:
        """并行运行所有实验（不推荐，可能导致API限流）"""
        self.start_time = time.time()
        self.log_message("🚀 开始并行运行所有实验的所有批次")
        self.log_message("⚠️  注意：并行运行可能导致API限流，建议使用串行模式")

        all_results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有实验任务
            future_to_experiment = {
                executor.submit(self.run_experiment_all_batches, exp): exp
                for exp in self.experiments
            }

            # 收集结果
            for future in concurrent.futures.as_completed(future_to_experiment):
                experiment = future_to_experiment[future]
                exp_type = experiment['type']

                try:
                    exp_results = future.result()
                    all_results[exp_type] = exp_results
                    self.log_message(f"✅ 实验 {experiment['name']} 完成")
                except Exception as e:
                    self.log_message(f"❌ 实验 {experiment['name']} 失败: {e}", "ERROR")
                    all_results[exp_type] = []

        return all_results

    def collect_results_summary(self, all_results: Dict) -> Dict:
        """收集结果汇总"""
        self.log_message("📊 开始收集实验结果汇总")

        summary = {
            'total_experiments': len(self.experiments),
            'total_batches': len(self.experiments) * len(self.batch_ids),
            'experiments_summary': {},
            'detailed_results': all_results,
            'run_metadata': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': time.time() - self.start_time,
                'max_workers': self.max_workers
            }
        }

        # 分析每个实验的结果
        for exp_type, exp_results in all_results.items():
            success_count = sum(1 for r in exp_results if r.get('status') == 'success')
            failed_count = len(exp_results) - success_count

            # 找到对应的实验配置
            experiment_config = next(exp for exp in self.experiments if exp['type'] == exp_type)

            summary['experiments_summary'][exp_type] = {
                'name': experiment_config['name'],
                'total_batches': len(exp_results),
                'successful_batches': success_count,
                'failed_batches': failed_count,
                'success_rate': success_count / len(exp_results) if exp_results else 0
            }

            self.log_message(f"📈 {experiment_config['name']}: {success_count}/{len(exp_results)} 批次成功")

        return summary

    def save_run_summary(self, summary: Dict):
        """保存运行汇总"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.results_dir / f'batch_experiments_summary_{timestamp}.json'

        # 添加运行日志
        summary['run_log'] = self.run_log

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.log_message(f"💾 运行汇总已保存到: {summary_file}")

    def check_batch_results(self) -> Dict:
        """检查批次结果文件"""
        self.log_message("🔍 检查现有批次结果文件")

        existing_results = {}
        expected_files = []

        # 生成期望的文件名列表
        for exp in self.experiments:
            for batch_id in self.batch_ids:
                filename = f"{exp['type']}_batch_{batch_id}_200_samples_results.json"
                expected_files.append({
                    'experiment': exp['type'],
                    'batch_id': batch_id,
                    'filename': filename,
                    'filepath': self.results_dir / filename
                })

        # 检查文件是否存在
        for file_info in expected_files:
            filepath = file_info['filepath']
            exp_type = file_info['experiment']
            batch_id = file_info['batch_id']

            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'batch_info' in data and data['batch_info'].get('status') == 'completed':
                            existing_results[f"{exp_type}_batch_{batch_id}"] = {
                                'status': 'completed',
                                'file': str(filepath)
                            }
                            self.log_message(f"✅ 找到完成的结果: {filepath.name}")
                        else:
                            existing_results[f"{exp_type}_batch_{batch_id}"] = {
                                'status': 'incomplete',
                                'file': str(filepath)
                            }
                            self.log_message(f"⚠️  找到未完成的结果: {filepath.name}")
                except Exception as e:
                    existing_results[f"{exp_type}_batch_{batch_id}"] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    self.log_message(f"❌ 结果文件损坏: {filepath.name}", "ERROR")
            else:
                existing_results[f"{exp_type}_batch_{batch_id}"] = {
                    'status': 'missing'
                }

        # 统计结果
        completed_count = sum(1 for r in existing_results.values() if r.get('status') == 'completed')
        total_count = len(expected_files)

        self.log_message(f"📊 检查结果: {completed_count}/{total_count} 批次已完成")

        return {
            'existing_results': existing_results,
            'completed_count': completed_count,
            'total_count': total_count,
            'completion_rate': completed_count / total_count if total_count > 0 else 0
        }

    def run(self, mode: str = 'serial', skip_existing: bool = True):
        """
        运行所有批量实验

        Args:
            mode: 运行模式 ('serial' 或 'parallel')
            skip_existing: 是否跳过已完成的批次
        """
        self.log_message(f"🚀 开始批量实验运行 - 模式: {mode}")

        # 检查现有结果
        if skip_existing:
            check_result = self.check_batch_results()
            if check_result['completion_rate'] >= 1.0:
                self.log_message("✅ 所有批次已完成，跳过运行")
                return check_result

        # 运行实验
        if mode == 'serial':
            all_results = self.run_all_experiments_serial()
        else:
            all_results = self.run_all_experiments_parallel()

        # 收集汇总
        summary = self.collect_results_summary(all_results)

        # 保存汇总
        self.save_run_summary(summary)

        self.log_message("🎯 批量实验运行完成")
        return summary

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量实验运行管理器')
    parser.add_argument('--mode', choices=['serial', 'parallel'], default='serial',
                       help='运行模式：串行(serial)或并行(parallel)，默认serial')
    parser.add_argument('--max-workers', type=int, default=1,
                       help='最大并行工作进程数，默认1')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='跳过已完成的批次，默认True')
    parser.add_argument('--check-only', action='store_true',
                       help='仅检查现有结果，不运行实验')

    args = parser.parse_args()

    # 创建运行器
    runner = BatchExperimentRunner(max_workers=args.max_workers)

    if args.check_only:
        # 仅检查现有结果
        check_result = runner.check_batch_results()
        print(f"\n📊 检查结果汇总:")
        print(f"   完成率: {check_result['completion_rate']:.1%}")
        print(f"   已完成: {check_result['completed_count']}/{check_result['total_count']}")
    else:
        # 运行实验
        summary = runner.run(mode=args.mode, skip_existing=args.skip_existing)
        print(f"\n🎯 运行完成汇总:")
        print(f"   总实验数: {summary['total_experiments']}")
        print(f"   总批次数: {summary['total_batches']}")
        print(f"   总用时: {summary['run_metadata']['total_duration']/60:.1f} 分钟")

if __name__ == "__main__":
    main()