"""
Tests for Hybrid Precision Evaluator
"""

import pytest
import numpy as np
from src.hybrid_retrieval.hybrid_precision import HybridPrecisionEvaluator


class TestHybridPrecisionEvaluator:
    """Test cases for HybridPrecisionEvaluator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = HybridPrecisionEvaluator(alpha=0.1, beta=0.1, gamma=0.1)
        self.dense_scores = [0.8, 0.7, 0.9, 0.6, 0.85]
        self.sparse_scores = [0.75, 0.72, 0.88, 0.65, 0.82]
        self.queries = [
            "test query 1",
            "test query 2",
            "complex query with multiple terms",
        ]

    def test_initialization(self):
        """Test proper initialization of evaluator."""
        assert self.evaluator.alpha == 0.1
        assert self.evaluator.beta == 0.1
        assert self.evaluator.gamma == 0.1
        assert hasattr(self.evaluator, "info_theory")
        assert hasattr(self.evaluator, "weight_optimizer")

    def test_evaluate_basic(self):
        """Test basic evaluation functionality."""
        results = self.evaluator.evaluate(self.dense_scores, self.sparse_scores)

        assert isinstance(results, dict)
        assert "hybrid_precision" in results
        assert "entropy_confidence" in results
        assert "mutual_information_confidence" in results
        assert "statistical_confidence" in results
        assert "adaptive_weights" in results
        assert "uncertainty_penalty" in results

    def test_evaluate_with_queries(self):
        """Test evaluation with query complexity analysis."""
        results = self.evaluator.evaluate(
            self.dense_scores, self.sparse_scores, self.queries
        )

        assert isinstance(results, dict)
        assert "hybrid_precision" in results
        assert all(
            0 <= v <= 1
            for v in [
                results["entropy_confidence"],
                results["mutual_information_confidence"],
                results["statistical_confidence"],
            ]
        )

    def test_mismatched_scores_length(self):
        """Test error handling for mismatched score lengths."""
        with pytest.raises(
            ValueError, match="Dense and sparse scores must have the same length"
        ):
            self.evaluator.evaluate([0.8, 0.7], [0.75, 0.72, 0.88])

    def test_query_complexity_analysis(self):
        """Test query complexity analysis."""
        simple_query = "short query"
        complex_query = "complex query with multiple terms and conjunctions"

        simple_complexity = self.evaluator._analyze_query_complexity(simple_query)
        complex_complexity = self.evaluator._analyze_query_complexity(complex_query)

        assert 0 <= simple_complexity <= 1
        assert 0 <= complex_complexity <= 1
        assert complex_complexity >= simple_complexity

    def test_uncertainty_penalty_calculation(self):
        """Test uncertainty penalty calculation."""
        dense_array = np.array(self.dense_scores)
        sparse_array = np.array(self.sparse_scores)

        penalty = self.evaluator._calculate_uncertainty_penalty(
            dense_array, sparse_array
        )

        assert 0 <= penalty <= 1
        assert isinstance(penalty, float)

    def test_batch_evaluation(self):
        """Test batch evaluation functionality."""
        results_list = [
            {"dense_scores": [0.8, 0.7, 0.9], "sparse_scores": [0.75, 0.72, 0.88]},
            {"dense_scores": [0.6, 0.85, 0.7], "sparse_scores": [0.65, 0.82, 0.72]},
        ]

        batch_results = self.evaluator.batch_evaluate(results_list)

        assert len(batch_results) == 2
        assert all(isinstance(result, dict) for result in batch_results)
        assert all("hybrid_precision" in result for result in batch_results)

    def test_evaluation_report_generation(self):
        """Test evaluation report generation."""
        results = self.evaluator.evaluate(self.dense_scores, self.sparse_scores)
        report = self.evaluator.get_evaluation_report(results)

        assert isinstance(report, str)
        assert "Hybrid Precision Evaluation Report" in report
        assert f"Overall Hybrid Precision: {results['hybrid_precision']:.4f}" in report

    def test_confidence_metrics_range(self):
        """Test that confidence metrics are within valid range."""
        results = self.evaluator.evaluate(self.dense_scores, self.sparse_scores)

        confidence_metrics = [
            results["entropy_confidence"],
            results["mutual_information_confidence"],
            results["statistical_confidence"],
        ]

        for metric in confidence_metrics:
            assert 0 <= metric <= 1, f"Confidence metric {metric} out of range [0, 1]"

    def test_adaptive_weights_sum_to_one(self):
        """Test that adaptive weights sum to approximately 1."""
        results = self.evaluator.evaluate(self.dense_scores, self.sparse_scores)

        dense_weight = results["adaptive_weights"]["dense"]
        sparse_weight = results["adaptive_weights"]["sparse"]

        weight_sum = dense_weight + sparse_weight
        assert (
            abs(weight_sum - 1.0) < 0.01
        ), f"Weights sum to {weight_sum}, expected ~1.0"

    def test_edge_case_identical_scores(self):
        """Test behavior with identical dense and sparse scores."""
        identical_scores = [0.8, 0.7, 0.9, 0.6, 0.85]
        results = self.evaluator.evaluate(identical_scores, identical_scores)

        assert isinstance(results, dict)
        assert "hybrid_precision" in results
        # With identical scores, uncertainty penalty should be low
        assert results["uncertainty_penalty"] < 0.1

    def test_edge_case_uniform_scores(self):
        """Test behavior with uniform scores."""
        uniform_dense = [0.5, 0.5, 0.5, 0.5, 0.5]
        uniform_sparse = [0.5, 0.5, 0.5, 0.5, 0.5]

        results = self.evaluator.evaluate(uniform_dense, uniform_sparse)

        assert isinstance(results, dict)
        assert "hybrid_precision" in results
