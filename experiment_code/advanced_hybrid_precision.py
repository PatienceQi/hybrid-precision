"""
高级Hybrid Precision算法
引入信息熵、互信息、自适应权重等高级概念
实现多维度置信度评估和统计显著性检验
"""

import numpy as np
import scipy.stats as stats
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

@dataclass
class AdvancedHybridConfig:
    """高级Hybrid Precision配置"""
    # 基础权重配置
    base_dense_weight: float = 0.7
    base_sparse_weight: float = 0.3

    # 置信度权重配置
    entropy_weight: float = 0.3
    mi_weight: float = 0.3
    statistical_weight: float = 0.2
    domain_weight: float = 0.2

    # 统计检验参数
    confidence_level: float = 0.95
    min_samples_for_stats: int = 10

    # 自适应参数
    complexity_threshold_high: float = 0.7
    complexity_threshold_low: float = 0.3
    max_weight_adjustment: float = 0.15

    # 不确定性参数
    uncertainty_penalty_factor: float = 0.2
    score_diff_threshold: float = 0.5

class InformationEntropyCalculator:
    """信息熵计算器"""

    def __init__(self):
        self.epsilon = 1e-10  # 避免log(0)

    def compute_entropy(self, scores: List[float]) -> float:
        """计算信息熵"""
        if not scores:
            return 0.0

        # 归一化分数到概率分布
        scores_array = np.array(scores)
        probabilities = scores_array / (np.sum(scores_array) + self.epsilon)
        probabilities = np.clip(probabilities, self.epsilon, 1.0)

        # 计算香农熵
        entropy = -np.sum(probabilities * np.log2(probabilities))

        # 归一化到[0, 1]范围
        max_entropy = np.log2(len(scores))
        normalized_entropy = entropy / (max_entropy + self.epsilon) if max_entropy > 0 else 0.0

        return normalized_entropy

    def compute_joint_entropy(self, scores1: List[float], scores2: List[float]) -> float:
        """计算联合熵"""
        if len(scores1) != len(scores2):
            raise ValueError("Score lists must have the same length")

        # 计算联合概率分布
        joint_probs = []
        for s1, s2 in zip(scores1, scores2):
            # 简单的联合概率估计
            joint_prob = (s1 * s2) / (np.sum(scores1) * np.sum(scores2) + self.epsilon)
            joint_probs.append(max(joint_prob, self.epsilon))

        joint_probs = np.array(joint_probs)
        joint_probs = joint_probs / (np.sum(joint_probs) + self.epsilon)

        # 计算联合熵
        joint_entropy = -np.sum(joint_probs * np.log2(joint_probs))

        # 归一化
        max_joint_entropy = np.log2(len(scores1))
        return joint_entropy / (max_joint_entropy + self.epsilon) if max_joint_entropy > 0 else 0.0

    def compute_mutual_information(self, scores1: List[float], scores2: List[float]) -> float:
        """计算互信息"""
        entropy1 = self.compute_entropy(scores1)
        entropy2 = self.compute_entropy(scores2)
        joint_entropy = self.compute_joint_entropy(scores1, scores2)

        # 互信息 = H(X) + H(Y) - H(X,Y)
        mutual_info = entropy1 + entropy2 - joint_entropy

        # 归一化到[0, 1]
        return max(0.0, min(1.0, mutual_info))

    def compute_entropy_confidence(self, dense_scores: List[float], sparse_scores: List[float]) -> float:
        """基于信息熵计算置信度"""
        entropy_dense = self.compute_entropy(dense_scores)
        entropy_sparse = self.compute_entropy(sparse_scores)

        # 熵越小，置信度越高
        avg_entropy = (entropy_dense + entropy_sparse) / 2.0
        confidence = 1.0 - avg_entropy

        return max(0.0, min(1.0, confidence))

class QueryComplexityAnalyzer:
    """查询复杂度分析器"""

    def __init__(self):
        self.complexity_factors = {
            'length': 0.25,
            'vocabulary': 0.25,
            'semantic': 0.25,
            'syntactic': 0.25
        }

    def analyze_complexity(self, question: str) -> float:
        """分析问题复杂度"""
        if not question.strip():
            return 0.5  # 默认中等复杂度

        # 1. 长度复杂度
        words = question.lower().split()
        length_score = min(len(words) / 30.0, 1.0)  # 30词以上为最高复杂度

        # 2. 词汇复杂度（词汇多样性）
        unique_words = set(words)
        vocab_diversity = len(unique_words) / (len(words) + 1e-10)
        vocab_score = vocab_diversity

        # 3. 语义复杂度（基于实体和专业术语）
        semantic_score = self._compute_semantic_complexity(question)

        # 4. 句法复杂度（基于句子结构和从句）
        syntactic_score = self._compute_syntactic_complexity(question)

        # 综合复杂度
        total_complexity = (
            self.complexity_factors['length'] * length_score +
            self.complexity_factors['vocabulary'] * vocab_score +
            self.complexity_factors['semantic'] * semantic_score +
            self.complexity_factors['syntactic'] * syntactic_score
        )

        return max(0.0, min(1.0, total_complexity))

    def _compute_semantic_complexity(self, text: str) -> float:
        """计算语义复杂度"""
        # 简单的实体识别和专业术语检测
        entities = self._extract_entities(text)
        technical_terms = self._detect_technical_terms(text)

        entity_score = min(len(entities) / 10.0, 1.0)
        technical_score = min(len(technical_terms) / 5.0, 1.0)

        return (entity_score + technical_score) / 2.0

    def _compute_syntactic_complexity(self, text: str) -> float:
        """计算句法复杂度"""
        # 简单的句法分析
        sentences = text.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])

        # 检测从句（基于连接词）
        conjunctions = ['and', 'but', 'or', 'because', 'since', 'although', 'while', 'if']
        conjunction_count = sum(text.lower().count(conj) for conj in conjunctions)

        length_score = min(avg_sentence_length / 25.0, 1.0)
        conjunction_score = min(conjunction_count / 5.0, 1.0)

        return (length_score + conjunction_score) / 2.0

    def _extract_entities(self, text: str) -> List[str]:
        """简单的实体提取"""
        # 大写单词（可能是专有名词）
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)

        # 数字（可能是日期、数量）
        numbers = re.findall(r'\b\d+\b', text)

        return list(set(capitalized_words + numbers))

    def _detect_technical_terms(self, text: str) -> List[str]:
        """检测技术术语"""
        # 常见的技术词汇后缀
        tech_suffixes = ['-tion', '-sion', '-ment', '-ness', '-ity', '-ics', '-logy']

        technical_terms = []
        words = text.lower().split()

        for word in words:
            for suffix in tech_suffixes:
                if word.endswith(suffix):
                    technical_terms.append(word)
                    break

        return list(set(technical_terms))

class StatisticalSignificanceTester:
    """统计显著性检验器"""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def perform_significance_test(self, dense_scores: List[float], sparse_scores: List[float]) -> Dict:
        """执行统计显著性检验"""
        if len(dense_scores) != len(sparse_scores) or len(dense_scores) < 2:
            return {
                'significance': 0.5,
                'p_value': 1.0,
                'confidence_interval': (0.0, 1.0),
                'effect_size': 0.0,
                'test_type': 'insufficient_data'
            }

        # 配对t检验
        try:
            t_stat, p_value = stats.ttest_rel(dense_scores, sparse_scores)

            # 计算效应量 (Cohen's d)
            diff_scores = np.array(dense_scores) - np.array(sparse_scores)
            effect_size = np.mean(diff_scores) / (np.std(diff_scores, ddof=1) + 1e-10)

            # 计算置信区间
            confidence_interval = self._compute_confidence_interval(dense_scores, sparse_scores)

            # 显著性判断
            significance = 1.0 - p_value  # 越高表示越显著

            return {
                'significance': max(0.0, min(1.0, significance)),
                'p_value': p_value,
                'confidence_interval': confidence_interval,
                'effect_size': effect_size,
                'test_type': 'paired_t_test',
                't_statistic': t_stat
            }
        except Exception as e:
            return {
                'significance': 0.5,
                'p_value': 1.0,
                'confidence_interval': (0.0, 1.0),
                'effect_size': 0.0,
                'test_type': 'error',
                'error': str(e)
            }

    def _compute_confidence_interval(self, dense_scores: List[float], sparse_scores: List[float]) -> Tuple[float, float]:
        """计算置信区间"""
        diff_scores = np.array(dense_scores) - np.array(sparse_scores)
        mean_diff = np.mean(diff_scores)
        std_diff = np.std(diff_scores, ddof=1)
        n = len(diff_scores)

        # t分布临界值
        t_critical = stats.t.ppf(1 - self.alpha/2, n-1)
        margin_error = t_critical * (std_diff / np.sqrt(n))

        ci_lower = mean_diff - margin_error
        ci_upper = mean_diff + margin_error

        return (max(-1.0, ci_lower), min(1.0, ci_upper))

class AdaptiveWeightOptimizer:
    """自适应权重优化器"""

    def __init__(self, config: AdvancedHybridConfig):
        self.config = config

    def optimize_weights(self, dense_scores: List[float], sparse_scores: List[float],
                        query_complexity: float, domain_confidence: float) -> Tuple[float, float]:
        """优化权重配置"""

        # 基础权重
        base_dense = self.config.base_dense_weight
        base_sparse = self.config.base_sparse_weight

        # 查询复杂度调整
        complexity_adjustment = self._compute_complexity_adjustment(query_complexity)

        # 分数差异调整
        score_adjustment = self._compute_score_adjustment(dense_scores, sparse_scores)

        # 领域置信度调整
        domain_adjustment = self._compute_domain_adjustment(domain_confidence)

        # 应用调整
        adjusted_dense = base_dense + complexity_adjustment + score_adjustment + domain_adjustment
        adjusted_sparse = base_sparse - complexity_adjustment - score_adjustment - domain_adjustment

        # 确保权重在合理范围内
        adjusted_dense = max(0.1, min(0.9, adjusted_dense))
        adjusted_sparse = max(0.1, min(0.9, adjusted_sparse))

        # 归一化
        total = adjusted_dense + adjusted_sparse
        final_dense = adjusted_dense / total
        final_sparse = adjusted_sparse / total

        return final_dense, final_sparse

    def _compute_complexity_adjustment(self, query_complexity: float) -> float:
        """计算复杂度调整"""
        if query_complexity > self.config.complexity_threshold_high:
            # 复杂查询，增加语义权重
            adjustment = self.config.max_weight_adjustment
        elif query_complexity < self.config.complexity_threshold_low:
            # 简单查询，增加关键词权重
            adjustment = -self.config.max_weight_adjustment
        else:
            adjustment = 0.0

        return adjustment

    def _compute_score_adjustment(self, dense_scores: List[float], sparse_scores: List[float]) -> float:
        """计算分数差异调整"""
        if len(dense_scores) != len(sparse_scores) or not dense_scores:
            return 0.0

        # 计算平均分数差异
        avg_dense = np.mean(dense_scores)
        avg_sparse = np.mean(sparse_scores)
        score_diff = abs(avg_dense - avg_sparse)

        if score_diff > self.config.score_diff_threshold:
            # 差异大，相信高分者
            if avg_dense > avg_sparse:
                adjustment = 0.05  # 增加稠密权重
            else:
                adjustment = -0.05  # 增加稀疏权重
        else:
            adjustment = 0.0

        return adjustment

    def _compute_domain_adjustment(self, domain_confidence: float) -> float:
        """计算领域置信度调整"""
        # 领域置信度影响权重调整
        adjustment = (domain_confidence - 0.5) * 0.1
        return max(-0.05, min(0.05, adjustment))

class AdvancedHybridPrecision:
    """高级Hybrid Precision主类"""

    def __init__(self, config: Optional[AdvancedHybridConfig] = None):
        self.config = config or AdvancedHybridConfig()
        self.entropy_calculator = InformationEntropyCalculator()
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.statistical_tester = StatisticalSignificanceTester(self.config.confidence_level)
        self.weight_optimizer = AdaptiveWeightOptimizer(self.config)

    def calculate_advanced_hybrid_precision(self,
                                          question: str,
                                          contexts: List[str],
                                          dense_scores: List[float],
                                          sparse_scores: List[float],
                                          reference_text: str = "") -> Dict:
        """计算高级Hybrid Precision"""

        try:
            # 1. 计算信息熵置信度
            entropy_conf = self.entropy_calculator.compute_entropy_confidence(dense_scores, sparse_scores)

            # 2. 计算互信息置信度
            mi_conf = self.entropy_calculator.compute_mutual_information(dense_scores, sparse_scores)

            # 3. 计算查询复杂度
            query_complexity = self.complexity_analyzer.analyze_complexity(question)

            # 4. 计算领域置信度（基于参考文本）
            domain_confidence = self._compute_domain_confidence(question, reference_text)

            # 5. 统计显著性检验
            statistical_result = self.statistical_tester.perform_significance_test(dense_scores, sparse_scores)
            statistical_conf = statistical_result['significance']

            # 6. 自适应权重优化
            adaptive_dense_weight, adaptive_sparse_weight = self.weight_optimizer.optimize_weights(
                dense_scores, sparse_scores, query_complexity, domain_confidence
            )

            # 7. 高级融合分数计算
            advanced_scores = []
            for i in range(len(contexts)):
                # 基础融合
                base_fusion = adaptive_dense_weight * dense_scores[i] + adaptive_sparse_weight * sparse_scores[i]

                # 多维度置信度加权
                total_confidence = (
                    entropy_conf * self.config.entropy_weight +
                    mi_conf * self.config.mi_weight +
                    statistical_conf * self.config.statistical_weight +
                    domain_confidence * self.config.domain_weight
                )

                confidence_weighted = base_fusion * total_confidence

                # 不确定性惩罚
                uncertainty_penalty = self._compute_uncertainty_penalty(
                    dense_scores[i], sparse_scores[i]
                )

                final_score = confidence_weighted * (1 - uncertainty_penalty)
                advanced_scores.append(max(0.0, min(1.0, final_score)))

            # 8. 计算最终精度
            advanced_precision = float(np.mean(advanced_scores))

            # 9. 生成分析报告
            analysis_report = self._generate_analysis_report(
                entropy_conf, mi_conf, query_complexity, domain_confidence,
                adaptive_dense_weight, adaptive_sparse_weight, statistical_result
            )

            return {
                'advanced_hybrid_precision': advanced_precision,
                'confidence_metrics': {
                    'entropy_confidence': entropy_conf,
                    'mutual_information_confidence': mi_conf,
                    'statistical_significance': statistical_conf,
                    'domain_confidence': domain_confidence
                },
                'adaptive_weights': {
                    'dense_weight': adaptive_dense_weight,
                    'sparse_weight': adaptive_sparse_weight
                },
                'query_complexity': query_complexity,
                'statistical_analysis': statistical_result,
                'advanced_scores': advanced_scores,
                'analysis_report': analysis_report,
                'interpretation': self._generate_interpretation(analysis_report)
            }

        except Exception as e:
            # 降级到基础Hybrid Precision
            basic_precision = np.mean([
                0.7 * dense_scores[i] + 0.3 * sparse_scores[i]
                for i in range(len(contexts))
            ])

            return {
                'advanced_hybrid_precision': basic_precision,
                'error': str(e),
                'fallback': True,
                'confidence_metrics': {'error': str(e)},
                'adaptive_weights': {'dense_weight': 0.7, 'sparse_weight': 0.3}
            }

    def _compute_domain_confidence(self, question: str, reference_text: str) -> float:
        """计算领域置信度"""
        if not reference_text.strip():
            return 0.5  # 默认中等置信度

        # 简单的领域相关性计算
        question_words = set(question.lower().split())
        reference_words = set(reference_text.lower().split())

        # 计算词汇重叠度
        overlap = len(question_words.intersection(reference_words))
        total_unique = len(question_words.union(reference_words))

        jaccard_similarity = overlap / (total_unique + 1e-10)

        # 转换为置信度
        domain_confidence = 0.3 + 0.7 * jaccard_similarity  # 基础置信度0.3

        return max(0.0, min(1.0, domain_confidence))

    def _compute_uncertainty_penalty(self, dense_score: float, sparse_score: float) -> float:
        """计算不确定性惩罚"""
        # 分数差异越大，不确定性越高
        score_diff = abs(dense_score - sparse_score)

        # 基础惩罚
        base_penalty = score_diff * self.config.uncertainty_penalty_factor

        # 如果两个分数都很低，增加惩罚
        if dense_score < 0.3 and sparse_score < 0.3:
            low_score_penalty = 0.1
        else:
            low_score_penalty = 0.0

        total_penalty = base_penalty + low_score_penalty

        return min(0.3, total_penalty)  # 最大惩罚不超过30%

    def _generate_analysis_report(self, entropy_conf: float, mi_conf: float,
                                query_complexity: float, domain_confidence: float,
                                dense_weight: float, sparse_weight: float,
                                statistical_result: Dict) -> Dict:
        """生成详细分析报告"""

        # 权重分析
        weight_analysis = {
            'final_dense_weight': dense_weight,
            'final_sparse_weight': sparse_weight,
            'weight_adjustment': dense_weight - 0.7,  # 相对于基础权重的调整
            'adjustment_reasons': []
        }

        if abs(dense_weight - 0.7) > 0.01:
            if dense_weight > 0.7:
                weight_analysis['adjustment_reasons'].append("复杂查询或稠密检索表现更好")
            else:
                weight_analysis['adjustment_reasons'].append("简单查询或稀疏检索表现更好")

        # 置信度分析
        confidence_analysis = {
            'overall_confidence': (entropy_conf + mi_conf + statistical_result['significance'] + domain_confidence) / 4.0,
            'entropy_reliability': 'high' if entropy_conf > 0.7 else 'medium' if entropy_conf > 0.4 else 'low',
            'mutual_information_strength': 'strong' if mi_conf > 0.6 else 'moderate' if mi_conf > 0.3 else 'weak',
            'statistical_significance': 'significant' if statistical_result['p_value'] < 0.05 else 'not_significant'
        }

        # 查询特征分析
        query_analysis = {
            'complexity_level': 'high' if query_complexity > 0.7 else 'medium' if query_complexity > 0.3 else 'low',
            'domain_relevance': 'high' if domain_confidence > 0.7 else 'medium' if domain_confidence > 0.4 else 'low'
        }

        return {
            'weight_analysis': weight_analysis,
            'confidence_analysis': confidence_analysis,
            'query_analysis': query_analysis,
            'statistical_details': statistical_result
        }

    def _generate_interpretation(self, analysis_report: Dict) -> str:
        """生成解释性文本"""

        weight_analysis = analysis_report['weight_analysis']
        confidence_analysis = analysis_report['confidence_analysis']
        query_analysis = analysis_report['query_analysis']

        # 构建解释文本
        interpretation_parts = []

        # 权重解释
        if abs(weight_analysis['weight_adjustment']) > 0.01:
            direction = "增加" if weight_analysis['weight_adjustment'] > 0 else "减少"
            interpretation_parts.append(
                f"基于查询特征分析，算法{direction}了语义相似性权重至{weight_analysis['final_dense_weight']:.3f}"
            )

        # 置信度解释
        if confidence_analysis['overall_confidence'] > 0.7:
            interpretation_parts.append("多维度置信度评估显示高可靠性")
        elif confidence_analysis['overall_confidence'] < 0.4:
            interpretation_parts.append("评估结果存在较大不确定性，建议谨慎解读")

        # 查询复杂度解释
        interpretation_parts.append(f"查询复杂度级别：{query_analysis['complexity_level']}")

        return "；".join(interpretation_parts) + "。"

# 便捷函数
def calculate_advanced_hybrid_precision(question: str, contexts: List[str],
                                      dense_scores: List[float], sparse_scores: List[float],
                                      reference_text: str = "") -> Dict:
    """便捷函数：计算高级Hybrid Precision"""
    calculator = AdvancedHybridPrecision()
    return calculator.calculate_advanced_hybrid_precision(
        question, contexts, dense_scores, sparse_scores, reference_text
    )