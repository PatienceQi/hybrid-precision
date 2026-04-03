"""
Adaptive Weight Optimization for Hybrid Retrieval

This module implements adaptive weight optimization based on query complexity,
score differences, and domain confidence for dynamic weight adjustment in
hybrid retrieval.
"""

import numpy as np
from typing import List, Tuple


class AdaptiveWeightOptimizer:
    """
    Adaptive weight optimizer for hybrid retrieval based on multiple factors.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.1, gamma: float = 0.1):
        """
        Initialize the adaptive weight optimizer.

        Args:
            alpha: Weight for complexity adjustment
            beta: Weight for score difference adjustment
            gamma: Weight for domain confidence adjustment
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.base_dense_weight = 0.7
        self.base_sparse_weight = 0.3

    def optimize_weights(
        self,
        dense_scores: np.ndarray,
        sparse_scores: np.ndarray,
        complexity_confidence: float = 0.5,
    ) -> Tuple[float, float]:
        """
        Optimize weights based on multiple factors.

        Args:
            dense_scores: Scores from dense retriever
            sparse_scores: Scores from sparse retriever
            complexity_confidence: Query complexity confidence (0-1)

        Returns:
            Tuple of (final_dense_weight, final_sparse_weight)
        """
        # Calculate score difference adjustment
        score_diff_adj = self._calculate_score_difference_adjustment(
            dense_scores, sparse_scores
        )

        # Calculate complexity adjustment
        complexity_adj = self._calculate_complexity_adjustment(complexity_confidence)

        # Calculate domain confidence adjustment (simplified for now)
        domain_adj = self._calculate_domain_adjustment()

        # Calculate final weights
        final_dense = (
            self.base_dense_weight + complexity_adj + score_diff_adj + domain_adj
        )
        final_sparse = (
            self.base_sparse_weight - complexity_adj - score_diff_adj - domain_adj
        )

        # Ensure weights are within valid range [0.1, 0.9]
        final_dense = np.clip(final_dense, 0.1, 0.9)
        final_sparse = np.clip(final_sparse, 0.1, 0.9)

        # Normalize to ensure they sum to 1
        total = final_dense + final_sparse
        final_dense = final_dense / total
        final_sparse = final_sparse / total

        return float(final_dense), float(final_sparse)

    def _calculate_score_difference_adjustment(
        self, dense_scores: np.ndarray, sparse_scores: np.ndarray
    ) -> float:
        """Calculate adjustment based on score differences."""
        # Calculate normalized score differences
        score_diff = np.abs(dense_scores - sparse_scores)
        mean_diff = np.mean(score_diff)
        std_diff = np.std(score_diff)

        # Normalize to [0, 1] range
        if std_diff > 0:
            normalized_diff = mean_diff / (mean_diff + std_diff)
        else:
            normalized_diff = 0.0

        # Apply beta weight
        return self.beta * (normalized_diff - 0.5)  # Center around 0

    def _calculate_complexity_adjustment(self, complexity_confidence: float) -> float:
        """Calculate adjustment based on query complexity."""
        # Complexity confidence in [0, 1]
        # For high complexity, slightly favor dense retrieval
        # For low complexity, balance between both
        adjustment = self.alpha * (complexity_confidence - 0.5)  # Center around 0
        return adjustment

    def _calculate_domain_adjustment(self) -> float:
        """Calculate domain-specific adjustment (simplified implementation)."""
        # This could be expanded based on domain detection
        # For now, return a small neutral adjustment
        return self.gamma * 0.0

    def optimize_weights_iterative(
        self,
        dense_scores_list: List[np.ndarray],
        sparse_scores_list: List[np.ndarray],
        complexity_list: List[float],
    ) -> List[Tuple[float, float]]:
        """
        Optimize weights iteratively for multiple queries.

        Args:
            dense_scores_list: List of dense retrieval scores arrays
            sparse_scores_list: List of sparse retrieval scores arrays
            complexity_list: List of complexity scores

        Returns:
            List of optimized weight tuples
        """
        return [
            self.optimize_weights(dense, sparse, comp)
            for dense, sparse, comp in zip(
                dense_scores_list, sparse_scores_list, complexity_list
            )
        ]

    def get_optimization_report(self, dense_weight: float, sparse_weight: float) -> str:
        """Generate a detailed optimization report."""
        report = f"""
Adaptive Weight Optimization Report
===================================

Final Weights:
- Dense Weight: {dense_weight:.4f}
- Sparse Weight: {sparse_weight:.4f}

Base Weights:
- Dense Base: {self.base_dense_weight:.4f}
- Sparse Base: {self.base_sparse_weight:.4f}

Adjustment Parameters:
- Alpha (Complexity): {self.alpha:.4f}
- Beta (Score Difference): {self.beta:.4f}
- Gamma (Domain): {self.gamma:.4f}
"""
        return report.strip()

    def reset_parameters(
        self, alpha: float = None, beta: float = None, gamma: float = None
    ):
        """Reset optimization parameters."""
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma

    def set_base_weights(self, dense_weight: float, sparse_weight: float):
        """Set base weights for dense and sparse retrievers."""
        self.base_dense_weight = dense_weight
        self.base_sparse_weight = sparse_weight

    def analyze_weight_sensitivity(
        self,
        dense_scores: np.ndarray,
        sparse_scores: np.ndarray,
        complexity_range: Tuple[float, float] = (0.0, 1.0),
        n_points: int = 11,
    ) -> List[Tuple[float, float, float]]:
        """
        Analyze weight sensitivity across different complexity levels.

        Args:
            dense_scores: Dense retrieval scores
            sparse_scores: Sparse retrieval scores
            complexity_range: Range of complexity values to test
            n_points: Number of complexity points to test

        Returns:
            List of (complexity, dense_weight, sparse_weight) tuples
        """
        complexities = np.linspace(complexity_range[0], complexity_range[1], n_points)
        results = []

        for complexity in complexities:
            dense_weight, sparse_weight = self.optimize_weights(
                dense_scores, sparse_scores, complexity
            )
            results.append((float(complexity), dense_weight, sparse_weight))

        return results
