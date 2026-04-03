"""
Tests for Information Theory Metrics
"""

import numpy as np
from src.hybrid_retrieval.information_theory import InformationTheoryMetrics


class TestInformationTheoryMetrics:
    """Test cases for InformationTheoryMetrics class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = InformationTheoryMetrics()
        self.dense_scores = np.array([0.8, 0.7, 0.9, 0.6, 0.85])
        self.sparse_scores = np.array([0.75, 0.72, 0.88, 0.65, 0.82])

    def test_initialization(self):
        """Test proper initialization."""
        assert isinstance(self.metrics, InformationTheoryMetrics)

    def test_entropy_confidence_calculation(self):
        """Test entropy-based confidence calculation."""
        confidence = self.metrics.calculate_entropy_confidence(
            self.dense_scores, self.sparse_scores
        )

        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)

    def test_mutual_information_confidence_calculation(self):
        """Test mutual information confidence calculation."""
        confidence = self.metrics.calculate_mutual_information_confidence(
            self.dense_scores, self.sparse_scores
        )

        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)

    def test_statistical_significance_calculation(self):
        """Test statistical significance confidence calculation."""
        confidence = self.metrics.calculate_statistical_significance(
            self.dense_scores, self.sparse_scores
        )

        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)

    def test_normalize_to_distribution(self):
        """Test score normalization to probability distribution."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        distribution = self.metrics._normalize_to_distribution(scores)

        assert isinstance(distribution, np.ndarray)
        assert len(distribution) == len(scores)
        assert np.isclose(np.sum(distribution), 1.0, rtol=1e-5)
        assert all(d > 0 for d in distribution)  # All probabilities should be positive

    def test_create_joint_distribution(self):
        """Test joint distribution creation."""
        joint_dist = self.metrics._create_joint_distribution(
            self.dense_scores, self.sparse_scores
        )

        assert isinstance(joint_dist, np.ndarray)
        assert joint_dist.ndim == 2
        assert np.isclose(np.sum(joint_dist), 1.0, rtol=1e-5)
        assert all(prob >= 0 for row in joint_dist for prob in row)

    def test_kl_divergence_calculation(self):
        """Test KL divergence calculation."""
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.1, 0.4, 0.5])

        kl_div = self.metrics.calculate_kl_divergence(p, q)

        assert kl_div >= 0  # KL divergence is always non-negative
        assert isinstance(kl_div, float)

    def test_js_divergence_calculation(self):
        """Test Jensen-Shannon divergence calculation."""
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.1, 0.4, 0.5])

        js_div = self.metrics.calculate_js_divergence(p, q)

        assert 0 <= js_div <= 1  # JS divergence is bounded between 0 and 1
        assert isinstance(js_div, float)

    def test_identical_distributions(self):
        """Test behavior with identical distributions."""
        identical_scores = np.array([0.8, 0.7, 0.9, 0.6, 0.85])

        entropy_conf = self.metrics.calculate_entropy_confidence(
            identical_scores, identical_scores
        )
        mutual_info_conf = self.metrics.calculate_mutual_information_confidence(
            identical_scores, identical_scores
        )

        assert 0 <= entropy_conf <= 1
        assert 0 <= mutual_info_conf <= 1

    def test_uniform_distributions(self):
        """Test behavior with uniform distributions."""
        uniform_dense = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        uniform_sparse = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        entropy_conf = self.metrics.calculate_entropy_confidence(
            uniform_dense, uniform_sparse
        )
        mutual_info_conf = self.metrics.calculate_mutual_information_confidence(
            uniform_dense, uniform_sparse
        )

        assert 0 <= entropy_conf <= 1
        assert 0 <= mutual_info_conf <= 1

    def test_single_element_scores(self):
        """Test behavior with single element scores."""
        single_dense = np.array([0.8])
        single_sparse = np.array([0.75])

        entropy_conf = self.metrics.calculate_entropy_confidence(
            single_dense, single_sparse
        )
        statistical_conf = self.metrics.calculate_statistical_significance(
            single_dense, single_sparse
        )

        assert entropy_conf == 1.0  # Single element should have maximum confidence
        assert 0 <= statistical_conf <= 1

    def test_information_report_generation(self):
        """Test information theory report generation."""
        report = self.metrics.get_information_report(
            self.dense_scores, self.sparse_scores
        )

        assert isinstance(report, str)
        assert "Information Theory Metrics Report" in report
        assert "Entropy-based Confidence:" in report
        assert "Mutual Information Confidence:" in report
        assert "Statistical Significance Confidence:" in report

    def test_edge_case_zero_scores(self):
        """Test behavior with zero scores."""
        zero_scores = np.array([0.0, 0.0, 0.0])

        # Should handle zero scores gracefully with epsilon addition
        distribution = self.metrics._normalize_to_distribution(zero_scores)
        assert all(d > 0 for d in distribution)
        assert np.isclose(np.sum(distribution), 1.0, rtol=1e-5)

    def test_edge_case_negative_scores(self):
        """Test behavior with negative scores."""
        negative_scores = np.array([-0.8, -0.7, -0.9])

        # Should handle negative scores
        distribution = self.metrics._normalize_to_distribution(negative_scores)
        assert np.isclose(np.sum(distribution), 1.0, rtol=1e-5)

    def test_edge_case_large_score_differences(self):
        """Test behavior with large score differences."""
        large_dense = np.array([10.0, 20.0, 30.0])
        small_sparse = np.array([0.1, 0.2, 0.3])

        entropy_conf = self.metrics.calculate_entropy_confidence(
            large_dense, small_sparse
        )
        mutual_info_conf = self.metrics.calculate_mutual_information_confidence(
            large_dense, small_sparse
        )

        assert 0 <= entropy_conf <= 1
        assert 0 <= mutual_info_conf <= 1
