"""
Tests for Adaptive Weight Optimization
"""

import pytest
import numpy as np
from src.hybrid_retrieval.adaptive_weights import AdaptiveWeightOptimizer


class TestAdaptiveWeightOptimizer:
    """Test cases for AdaptiveWeightOptimizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = AdaptiveWeightOptimizer(alpha=0.1, beta=0.1, gamma=0.1)
        self.dense_scores = np.array([0.8, 0.7, 0.9, 0.6, 0.85])
        self.sparse_scores = np.array([0.75, 0.72, 0.88, 0.65, 0.82])

    def test_initialization(self):
        """Test proper initialization."""
        assert self.optimizer.alpha == 0.1
        assert self.optimizer.beta == 0.1
        assert self.optimizer.gamma == 0.1
        assert self.optimizer.base_dense_weight == 0.7
        assert self.optimizer.base_sparse_weight == 0.3

    def test_optimize_weights_basic(self):
        """Test basic weight optimization."""
        dense_weight, sparse_weight = self.optimizer.optimize_weights(
            self.dense_scores, self.sparse_scores
        )

        assert 0.1 <= dense_weight <= 0.9
        assert 0.1 <= sparse_weight <= 0.9
        assert abs(dense_weight + sparse_weight - 1.0) < 0.01

    def test_optimize_weights_with_complexity(self):
        """Test weight optimization with complexity confidence."""
        complexity_confidence = 0.8
        dense_weight, sparse_weight = self.optimizer.optimize_weights(
            self.dense_scores, self.sparse_scores, complexity_confidence
        )

        assert 0.1 <= dense_weight <= 0.9
        assert 0.1 <= sparse_weight <= 0.9
        assert abs(dense_weight + sparse_weight - 1.0) < 0.01

    def test_score_difference_adjustment(self):
        """Test score difference adjustment calculation."""
        # Create scores with significant differences
        diff_dense = np.array([0.9, 0.8, 0.95])
        diff_sparse = np.array([0.5, 0.4, 0.45])

        adjustment = self.optimizer._calculate_score_difference_adjustment(diff_dense, diff_sparse)

        assert isinstance(adjustment, float)
        # Adjustment should be relatively small due to beta weight
        assert abs(adjustment) < 0.1

    def test_complexity_adjustment(self):
        """Test complexity adjustment calculation."""
        high_complexity = 0.9
        low_complexity = 0.1

        high_adj = self.optimizer._calculate_complexity_adjustment(high_complexity)
        low_adj = self.optimizer._calculate_complexity_adjustment(low_complexity)

        assert isinstance(high_adj, float)
        assert isinstance(low_adj, float)
        # High complexity should favor dense retrieval (positive adjustment)
        assert high_adj > low_adj

    def test_domain_adjustment(self):
        """Test domain adjustment calculation."""
        adjustment = self.optimizer._calculate_domain_adjustment()

        assert isinstance(adjustment, float)
        # Default implementation should return 0
        assert adjustment == 0.0

    def test_optimize_weights_iterative(self):
        """Test iterative weight optimization."""
        dense_scores_list = [
            np.array([0.8, 0.7, 0.9]),
            np.array([0.6, 0.85, 0.7]),
            np.array([0.9, 0.8, 0.75])
        ]
        sparse_scores_list = [
            np.array([0.75, 0.72, 0.88]),
            np.array([0.65, 0.82, 0.72]),
            np.array([0.85, 0.78, 0.73])
        ]
        complexity_list = [0.5, 0.7, 0.3]

        results = self.optimizer.optimize_weights_iterative(
            dense_scores_list, sparse_scores_list, complexity_list
        )

        assert len(results) == 3
        for dense_weight, sparse_weight in results:
            assert 0.1 <= dense_weight <= 0.9
            assert 0.1 <= sparse_weight <= 0.9
            assert abs(dense_weight + sparse_weight - 1.0) < 0.01

    def test_optimization_report_generation(self):
        """Test optimization report generation."""
        dense_weight, sparse_weight = 0.65, 0.35
        report = self.optimizer.get_optimization_report(dense_weight, sparse_weight)

        assert isinstance(report, str)
        assert "Adaptive Weight Optimization Report" in report
        assert f"Dense Weight: {dense_weight:.4f}" in report
        assert f"Sparse Weight: {sparse_weight:.4f}" in report

    def test_reset_parameters(self):
        """Test parameter reset functionality."""
        original_alpha = self.optimizer.alpha
        new_alpha = 0.2

        self.optimizer.reset_parameters(alpha=new_alpha)

        assert self.optimizer.alpha == new_alpha
        assert self.optimizer.alpha != original_alpha

    def test_set_base_weights(self):
        """Test base weights setting."""
        new_dense_weight = 0.6
        new_sparse_weight = 0.4

        self.optimizer.set_base_weights(new_dense_weight, new_sparse_weight)

        assert self.optimizer.base_dense_weight == new_dense_weight
        assert self.optimizer.base_sparse_weight == new_sparse_weight

    def test_weight_sensitivity_analysis(self):
        """Test weight sensitivity analysis."""
        results = self.optimizer.analyze_weight_sensitivity(
            self.dense_scores, self.sparse_scores,
            complexity_range=(0.0, 1.0), n_points=5
        )

        assert len(results) == 5
        for complexity, dense_weight, sparse_weight in results:
            assert 0.0 <= complexity <= 1.0
            assert 0.1 <= dense_weight <= 0.9
            assert 0.1 <= sparse_weight <= 0.9
            assert abs(dense_weight + sparse_weight - 1.0) < 0.01

    def test_edge_case_identical_scores(self):
        """Test behavior with identical scores."""
        identical_scores = np.array([0.8, 0.7, 0.9, 0.6, 0.85])

        dense_weight, sparse_weight = self.optimizer.optimize_weights(
            identical_scores, identical_scores
        )

        assert 0.1 <= dense_weight <= 0.9
        assert 0.1 <= sparse_weight <= 0.9
        assert abs(dense_weight + sparse_weight - 1.0) < 0.01

    def test_edge_case_uniform_scores(self):
        """Test behavior with uniform scores."""
        uniform_dense = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        uniform_sparse = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        dense_weight, sparse_weight = self.optimizer.optimize_weights(
            uniform_dense, uniform_sparse
        )

        assert 0.1 <= dense_weight <= 0.9
        assert 0.1 <= sparse_weight <= 0.9
        assert abs(dense_weight + sparse_weight - 1.0) < 0.01

    def test_edge_case_single_element(self):
        """Test behavior with single element scores."""
        single_dense = np.array([0.8])
        single_sparse = np.array([0.75])

        dense_weight, sparse_weight = self.optimizer.optimize_weights(
            single_dense, single_sparse
        )

        assert 0.1 <= dense_weight <= 0.9
        assert 0.1 <= sparse_weight <= 0.9
        assert abs(dense_weight + sparse_weight - 1.0) < 0.01

    def test_edge_case_zero_std_deviation(self):
        """Test behavior when standard deviation is zero."""
        uniform_dense = np.array([0.8, 0.8, 0.8])
        varying_sparse = np.array([0.75, 0.85, 0.70])

        # Should not crash when dense scores have zero std deviation
        dense_weight, sparse_weight = self.optimizer.optimize_weights(
            uniform_dense, varying_sparse
        )

        assert 0.1 <= dense_weight <= 0.9
        assert 0.1 <= sparse_weight <= 0.9
        assert abs(dense_weight + sparse_weight - 1.0) < 0.01

    def test_weight_bounds_enforcement(self):
        """Test that weights are properly bounded."""
        # Create extreme conditions that might push weights outside bounds
        extreme_dense = np.array([0.99, 0.98, 0.97])
        extreme_sparse = np.array([0.01, 0.02, 0.03])
        high_complexity = 1.0

        dense_weight, sparse_weight = self.optimizer.optimize_weights(
            extreme_dense, extreme_sparse, high_complexity
        )

        assert 0.1 <= dense_weight <= 0.9
        assert 0.1 <= sparse_weight <= 0.9

    def test_negative_score_differences(self):
        """Test behavior with negative score adjustments."""
        # Create scores where sparse is significantly better than dense
        better_sparse = np.array([0.5, 0.4, 0.45])
        worse_dense = np.array([0.9, 0.8, 0.95])

        dense_weight, sparse_weight = self.optimizer.optimize_weights(
            worse_dense, better_sparse
        )

        assert 0.1 <= dense_weight <= 0.9
        assert 0.1 <= sparse_weight <= 0.9
        assert abs(dense_weight + sparse_weight - 1.0) < 0.01

    def test_parameter_validation(self):
        """Test parameter validation in reset_parameters."""
        # Should not raise any errors
        self.optimizer.reset_parameters(alpha=0.5, beta=0.3, gamma=0.2)
        assert self.optimizer.alpha == 0.5
        assert self.optimizer.beta == 0.3
        assert self.optimizer.gamma == 0.2

        # Test partial parameter reset
        self.optimizer.reset_parameters(beta=0.4)
        assert self.optimizer.alpha == 0.5  # Should remain unchanged
        assert self.optimizer.beta == 0.4
        assert self.optimizer.gamma == 0.2  # Should remain unchanged