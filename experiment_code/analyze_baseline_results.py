"""
实验一（Baseline）结果合并与分析脚本
用于提取和合并5个批次的baseline实验结果，生成综合性能报告
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import statistics
from datetime import datetime

class BaselineResultsAnalyzer:
    def __init__(self, batch_results_dir: str = "batch_results"):
        """
        初始化分析器
        Args:
            batch_results_dir: 批次结果文件目录
        """
        self.batch_results_dir = Path(batch_results_dir)
        self.key_metrics = [
            "context_precision",
            "faithfulness",
            "answer_relevancy",
            "context_recall"
        ]
        self.all_results = []
        self.batch_summaries = []

    def load_batch_results(self) -> bool:
        """
        加载所有baseline批次的结果文件
        Returns:
            是否成功加载所有文件
        """
        print("📊 开始加载实验一（Baseline）批次结果...")

        for batch_id in range(1, 6):
            result_file = self.batch_results_dir / f"baseline_batch_{batch_id}_200_samples_results.json"

            if not result_file.exists():
                print(f"❌ 未找到批次{batch_id}结果文件: {result_file}")
                return False

            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)

                # 提取批次摘要数据
                summary = batch_data.get("summary", {})
                baseline_summary = summary.get("baseline_only", {})

                self.batch_summaries.append({
                    "batch_id": batch_id,
                    "total_samples": summary.get("total_samples", 0),
                    "metrics": baseline_summary
                })

                # 提取详细结果
                results = batch_data.get("results", [])
                for idx, result in enumerate(results):
                    processed_result = {
                        "batch_id": batch_id,
                        "sample_id": (batch_id - 1) * 200 + idx + 1,  # 全局样本ID
                        "context_precision": result.get("context_precision", 0.0),
                        "faithfulness": result.get("faithfulness", 0.0),
                        "answer_relevancy": result.get("answer_relevancy", 0.0),
                        "context_recall": result.get("context_recall", 0.0)
                    }
                    self.all_results.append(processed_result)

                print(f"✅ 批次{batch_id}: 成功加载{batch_data.get('summary', {}).get('total_samples', 0)}个样本")

            except Exception as e:
                print(f"❌ 加载批次{batch_id}失败: {e}")
                return False

        print(f"📈 总计加载{len(self.all_results)}个样本数据")
        return True

    def calculate_comprehensive_statistics(self) -> Dict[str, Any]:
        """
        计算综合统计指标
        Returns:
            包含所有统计数据的字典
        """
        if not self.all_results:
            return {}

        print("🔍 计算综合统计指标...")

        # 提取各指标数据
        metrics_data = {
            metric: [result[metric] for result in self.all_results]
            for metric in self.key_metrics
        }

        # 计算基础统计
        statistics = {}
        for metric, values in metrics_data.items():
            statistics[metric] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75))
            }

        # 计算批次间一致性
        batch_consistency = self._calculate_batch_consistency()

        # 识别异常样本
        outliers = self._identify_outliers()

        # 性能分布分析
        distribution = self._analyze_distribution()

        return {
            "total_samples": len(self.all_results),
            "batches_analyzed": len(self.batch_summaries),
            "overall_statistics": statistics,
            "batch_consistency": batch_consistency,
            "outliers": outliers,
            "distribution_analysis": distribution,
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _calculate_batch_consistency(self) -> Dict[str, Any]:
        """
        计算批次间一致性分析
        Returns:
            批次一致性分析结果
        """
        if not self.batch_summaries:
            return {}

        batch_means = {}
        for metric in self.key_metrics:
            batch_means[metric] = [batch["metrics"].get(f"avg_{metric}", 0.0)
                                 for batch in self.batch_summaries]

        consistency = {}
        for metric, means in batch_means.items():
            consistency[metric] = {
                "batch_means": means,
                "std": float(np.std(means)),
                "cv": float(np.std(means) / np.mean(means)) if np.mean(means) > 0 else 0.0,
                "range": float(np.max(means) - np.min(means)),
                "consistency_score": float(1.0 - (np.std(means) / np.mean(means))) if np.mean(means) > 0 else 0.0
            }

        return consistency

    def _identify_outliers(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        识别异常样本
        Returns:
            异常样本分析
        """
        outliers = {}

        for metric in self.key_metrics:
            values = [result[metric] for result in self.all_results]
            q25, q75 = np.percentile(values, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            metric_outliers = []
            for result in self.all_results:
                value = result[metric]
                if value < lower_bound or value > upper_bound:
                    metric_outliers.append({
                        "sample_id": result["sample_id"],
                        "batch_id": result["batch_id"],
                        "value": value,
                        "deviation": abs(value - np.mean(values))
                    })

            # 按偏离程度排序，取前10个
            metric_outliers.sort(key=lambda x: x["deviation"], reverse=True)
            outliers[metric] = metric_outliers[:10]

        return outliers

    def _analyze_distribution(self) -> Dict[str, Any]:
        """
        分析性能分布
        Returns:
            分布分析结果
        """
        distribution = {}

        for metric in self.key_metrics:
            values = [result[metric] for result in self.all_results]

            # 性能分段统计
            segments = {
                "excellent": len([v for v in values if v >= 0.8]),
                "good": len([v for v in values if 0.6 <= v < 0.8]),
                "fair": len([v for v in values if 0.4 <= v < 0.6]),
                "poor": len([v for v in values if 0.2 <= v < 0.4]),
                "very_poor": len([v for v in values if v < 0.2])
            }

            distribution[metric] = {
                "segment_distribution": segments,
                "segment_percentages": {
                    k: (v / len(values)) * 100 for k, v in segments.items()
                }
            }

        return distribution

    def generate_batch_comparison(self) -> Dict[str, Any]:
        """
        生成批次间对比分析
        Returns:
            批次对比分析结果
        """
        if not self.batch_summaries:
            return {}

        comparison = {
            "batch_details": self.batch_summaries,
            "performance_ranking": {},
            "stability_analysis": {}
        }

        # 性能排名
        for metric in self.key_metrics:
            batch_scores = [(batch["batch_id"], batch["metrics"].get(f"avg_{metric}", 0.0))
                           for batch in self.batch_summaries]
            batch_scores.sort(key=lambda x: x[1], reverse=True)
            comparison["performance_ranking"][metric] = batch_scores

        return comparison

    def save_comprehensive_report(self, output_file: str = "baseline_comprehensive_analysis.json"):
        """
        保存综合分析报告
        Args:
            output_file: 输出文件名
        """
        print("💾 生成综合分析报告...")

        comprehensive_stats = self.calculate_comprehensive_statistics()
        batch_comparison = self.generate_batch_comparison()

        report = {
            "experiment_info": {
                "experiment_type": "baseline_ragas_analysis",
                "total_samples": comprehensive_stats.get("total_samples", 0),
                "batches_analyzed": comprehensive_stats.get("batches_analyzed", 0),
                "analysis_timestamp": comprehensive_stats.get("analysis_timestamp", ""),
                "key_metrics_analyzed": self.key_metrics
            },
            "comprehensive_statistics": comprehensive_stats,
            "batch_comparison": batch_comparison,
            "all_individual_results": self.all_results,
            "summary_by_batch": self.batch_summaries
        }

        # 保存详细报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✅ 综合分析报告已保存: {output_file}")

        # 生成摘要报告
        self._generate_summary_report(report)

        return report

    def _generate_summary_report(self, comprehensive_report: Dict[str, Any]):
        """
        生成简化的摘要报告
        Args:
            comprehensive_report: 详细分析报告
        """
        stats = comprehensive_report["comprehensive_statistics"]["overall_statistics"]

        print("\n" + "="*80)
        print("🎯 实验一（Baseline）综合性能报告")
        print("="*80)
        print(f"📊 总样本数: {comprehensive_report['experiment_info']['total_samples']}")
        print(f"📈 分析批次: {comprehensive_report['experiment_info']['batches_analyzed']}")
        print("\n📋 关键指标统计:")
        print("-" * 80)

        for metric, values in stats.items():
            print(f"\n🔍 {metric.upper()}:")
            print(f"   平均值: {values['mean']:.4f}")
            print(f"   中位数: {values['median']:.4f}")
            print(f"   标准差: {values['std']:.4f}")
            print(f"   最小值: {values['min']:.4f}")
            print(f"   最大值: {values['max']:.4f}")
            print(f"   Q25: {values['q25']:.4f} | Q75: {values['q75']:.4f}")

        # 批次一致性分析
        consistency = comprehensive_report["comprehensive_statistics"]["batch_consistency"]
        print(f"\n🔄 批次一致性分析:")
        print("-" * 80)
        for metric, consist_data in consistency.items():
            print(f"\n📏 {metric.upper()}:")
            print(f"   批次间标准差: {consist_data['std']:.4f}")
            print(f"   变异系数: {consist_data['cv']:.4f}")
            print(f"   一致性评分: {consist_data['consistency_score']:.4f}")
            print(f"   批次均值: {[f'{x:.4f}' for x in consist_data['batch_means']]}")

        print("\n" + "="*80)

def main():
    """主函数"""
    print("🚀 开始实验一（Baseline）综合分析...")
    print("="*80)

    # 创建分析器
    analyzer = BaselineResultsAnalyzer()

    # 加载数据
    if not analyzer.load_batch_results():
        print("❌ 数据加载失败，请检查文件路径和格式")
        return

    # 生成综合分析报告
    report = analyzer.save_comprehensive_report()

    print("\n✅ 实验一（Baseline）综合分析完成！")
    print("📁 详细结果已保存到 baseline_comprehensive_analysis.json")

if __name__ == "__main__":
    main()