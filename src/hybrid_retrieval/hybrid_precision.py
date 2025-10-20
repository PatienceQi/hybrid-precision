"""
Hybrid Precision Evaluator

This module implements the core Hybrid Precision evaluation method for hybrid retrieval systems.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.stats import ttest_rel
from .information_theory import InformationTheoryMetrics
from .adaptive_weights import AdaptiveWeightOptimizer


class HybridPrecisionEvaluator:
    """
    Hybrid Precision evaluator using information theory-driven multi-dimensional confidence assessment.

    This class implements the core evaluation method that combines information entropy, mutual information,
    and statistical significance to provide specialized evaluation for hybrid retrieval systems.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.1):
        """
        Initialize the Hybrid Precision evaluator.

        Args:
            alpha: Weight for complexity adjustment
            beta: Weight for score difference adjustment
            gamma: Weight for domain confidence adjustment
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.info_theory = InformationTheoryMetrics()
        self.weight_optimizer = AdaptiveWeightOptimizer(alpha, beta, gamma)

    def evaluate(self, dense_scores: List[float], sparse_scores: List[float],
                 queries: List[str] = None) -> Dict[str, float]:
        """
        Evaluate hybrid retrieval performance using the Hybrid Precision method.

        Args:
            dense_scores: Scores from dense retriever
            sparse_scores: Scores from sparse retriever
            queries: Optional list of queries for complexity analysis

        Returns:
            Dictionary containing evaluation metrics and confidence scores
        """
        if len(dense_scores) != len(sparse_scores):
            raise ValueError("Dense and sparse scores must have the same length")

        # Convert to numpy arrays for efficient computation
        dense_array = np.array(dense_scores)
        sparse_array = np.array(sparse_scores)

        # Calculate information theory metrics
        entropy_conf = self.info_theory.calculate_entropy_confidence(dense_array, sparse_array)
        mutual_info_conf = self.info_theory.calculate_mutual_information_confidence(dense_array, sparse_array)
        statistical_conf = self.info_theory.calculate_statistical_significance(dense_array, sparse_array)

        # Calculate adaptive weights
        if queries:
            complexity_scores = [self._analyze_query_complexity(q) for q in queries]
            complexity_conf = np.mean(complexity_scores)
        else:
            complexity_conf = 0.5

        # Calculate final hybrid scores
        final_dense_weight, final_sparse_weight = self.weight_optimizer.optimize_weights(
            dense_array, sparse_array, complexity_conf
        )

        # Calculate base hybrid score
        base_scores = final_dense_weight * dense_array + final_sparse_weight * sparse_array

        # Apply confidence weighting
        confidence_weighted = base_scores * (entropy_conf + mutual_info_conf + statistical_conf) / 3

        # Apply uncertainty penalty
        uncertainty_penalty = self._calculate_uncertainty_penalty(dense_array, sparse_array)
        final_scores = confidence_weighted * (1 - uncertainty_penalty)

        return {
            "hybrid_precision": float(np.mean(final_scores)),
            "entropy_confidence": float(entropy_conf),
            "mutual_information_confidence": float(mutual_info_conf),
            "statistical_confidence": float(statistical_conf),
            "adaptive_weights": {
                "dense": float(final_dense_weight),
                "sparse": float(final_sparse_weight)
            },
            "uncertainty_penalty": float(uncertainty_penalty)
        }

    def _analyze_query_complexity(self, query: str) -> float:
        """Analyze query complexity for adaptive weighting."""
        # Simple complexity analysis based on query length and structure
        length_factor = min(len(query) / 100, 1.0)

        # Check for complex query indicators
        complex_indicators = ["and", "or", "but", "however", "moreover"]
        structure_factor = sum(1 for indicator in complex_indicators if indicator in query.lower()) / len(complex_indicators)

        return (length_factor + structure_factor) / 2

    def _calculate_uncertainty_penalty(self, dense_scores: np.ndarray, sparse_scores: np.ndarray) -> float:
        """Calculate uncertainty penalty based on score differences."""
        # Calculate normalized score differences
        score_diff = np.abs(dense_scores - sparse_scores)
        max_diff = np.max(score_diff) if np.max(score_diff) > 0 else 1
        normalized_diff = score_diff / max_diff

        # Return mean normalized difference as penalty
        return float(np.mean(normalized_diff))

    def batch_evaluate(self, results_list: List[Dict[str, List[float]]]) -> List[Dict[str, float]]:
        """
        Batch evaluate multiple hybrid retrieval results.

        Args:
            results_list: List of dictionaries containing dense and sparse scores

        Returns:
            List of evaluation results
        """
        return [
            self.evaluate(res["dense_scores"], res["sparse_scores"], res.get("queries"))
            for res in results_list
        ]

    def get_evaluation_report(self, results: Dict[str, float]) -> str:
        """Generate a detailed evaluation report."""
        report = f"""
Hybrid Precision Evaluation Report
==================================

Overall Hybrid Precision: {results['hybrid_precision']:.4f}

Confidence Metrics:
- Entropy Confidence: {results['entropy_confidence']:.4f}
- Mutual Information Confidence: {results['mutual_information_confidence']:.4f}
- Statistical Confidence: {results['statistical_confidence']:.4f}

Adaptive Weights:
- Dense Weight: {results['adaptive_weights']['dense']:.4f}
- Sparse Weight: {results['adaptive_weights']['sparse']:.4f}

Uncertainty Penalty: {results['uncertainty_penalty']:.4f}
"""
        return report.strip()