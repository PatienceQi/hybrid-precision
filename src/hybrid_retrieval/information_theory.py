"""
Information Theory Metrics for Hybrid Retrieval Evaluation

This module implements information theory-based metrics including entropy,
mutual information, and statistical significance for confidence assessment
in hybrid retrieval systems.
"""

import numpy as np
from scipy.stats import ttest_rel, entropy
from scipy.special import rel_entr


class InformationTheoryMetrics:
    """
    Information theory metrics for hybrid retrieval evaluation confidence
    assessment.
    """

    def __init__(self):
        """Initialize information theory metrics calculator."""
        pass

    def calculate_entropy_confidence(
        self, dense_scores: np.ndarray, sparse_scores: np.ndarray
    ) -> float:
        """
        Calculate entropy-based confidence for score distributions.

        Entropy measures the uncertainty or randomness in the score
        distributions. Lower entropy indicates more ordered/structured
        distributions, suggesting higher confidence.

        Args:
            dense_scores: Scores from dense retriever
            sparse_scores: Scores from sparse retriever

        Returns:
            Entropy-based confidence score (0-1)
        """
        # Handle edge cases
        if len(dense_scores) == 0 or len(sparse_scores) == 0:
            return 0.0

        if len(dense_scores) == 1 or len(sparse_scores) == 1:
            return 1.0  # Single element has maximum confidence

        # Normalize scores to probability distributions
        dense_norm = self._normalize_to_distribution(dense_scores)
        sparse_norm = self._normalize_to_distribution(sparse_scores)

        # Calculate entropy for each distribution
        entropy_dense = entropy(dense_norm)
        entropy_sparse = entropy(sparse_norm)

        # Normalize entropy (0 = minimum entropy, max = maximum possible)
        max_entropy = np.log(len(dense_scores))
        if max_entropy == 0:
            return 1.0  # Single element case

        # Calculate normalized entropy (0-1)
        norm_entropy_dense = entropy_dense / max_entropy
        norm_entropy_sparse = entropy_sparse / max_entropy

        # Confidence is inverse of normalized entropy
        confidence_dense = max(0.0, min(1.0, 1 - norm_entropy_dense))
        confidence_sparse = max(0.0, min(1.0, 1 - norm_entropy_sparse))

        # Return average confidence
        return float((confidence_dense + confidence_sparse) / 2)

    def calculate_mutual_information_confidence(
        self, dense_scores: np.ndarray, sparse_scores: np.ndarray
    ) -> float:
        """
        Calculate mutual information-based confidence between dense and sparse
        retrievers.

        Mutual information measures the amount of information obtained about
        one random variable through observing the other random variable.
        Higher mutual information suggests better alignment.

        Args:
            dense_scores: Scores from dense retriever
            sparse_scores: Scores from sparse retriever

        Returns:
            Mutual information-based confidence score (0-1)
        """
        # Create joint probability distribution
        joint_dist = self._create_joint_distribution(dense_scores, sparse_scores)

        # Calculate marginal distributions
        marginal_dense = np.sum(joint_dist, axis=1)
        marginal_sparse = np.sum(joint_dist, axis=0)

        # Calculate mutual information
        mi = 0.0
        for i in range(len(marginal_dense)):
            for j in range(len(marginal_sparse)):
                if (
                    joint_dist[i, j] > 0
                    and marginal_dense[i] > 0
                    and marginal_sparse[j] > 0
                ):
                    mi += joint_dist[i, j] * np.log(
                        joint_dist[i, j]
                        / (marginal_dense[i] * marginal_sparse[j])
                    )

        # Normalize mutual information to [0, 1]
        # Use tanh for smooth normalization
        max_mi = np.log(min(len(dense_scores), len(sparse_scores)))
        if max_mi == 0:
            return 0.0

        normalized_mi = np.tanh(mi / max_mi)
        return float(normalized_mi)

    def calculate_statistical_significance(
        self, dense_scores: np.ndarray, sparse_scores: np.ndarray
    ) -> float:
        """
        Calculate statistical significance confidence using paired t-test.

        Args:
            dense_scores: Scores from dense retriever
            sparse_scores: Scores from sparse retriever

        Returns:
            Statistical significance confidence score (0-1)
        """
        # Handle edge cases
        if len(dense_scores) == 0 or len(sparse_scores) == 0:
            return 0.0

        if len(dense_scores) == 1 or len(sparse_scores) == 1:
            return 1.0  # Single element has maximum confidence

        try:
            # Perform paired t-test
            t_stat, p_value = ttest_rel(dense_scores, sparse_scores)

            # Handle NaN p-values (e.g., when all values are identical)
            if np.isnan(p_value):
                return 1.0  # Identical distributions have high confidence

            # Convert p-value to confidence (1 - p_value)
            # Use confidence = 1 - p_value, but cap at 0.999
            confidence = min(max(0.0, 1.0 - p_value), 0.999)

            return float(confidence)

        except Exception:
            # If t-test fails (e.g., insufficient data), return moderate confidence
            return 0.5

    def _normalize_to_distribution(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to probability distribution."""
        # Add small epsilon to avoid zero probabilities
        scores_norm = scores + 1e-10
        return scores_norm / np.sum(scores_norm)

    def _create_joint_distribution(
        self, dense_scores: np.ndarray, sparse_scores: np.ndarray
    ) -> np.ndarray:
        """Create joint probability distribution from dense and sparse scores."""
        # Handle edge cases
        if len(dense_scores) == 0 or len(sparse_scores) == 0:
            return np.zeros((2, 2))  # Return minimal distribution

        # Handle single element case
        if len(dense_scores) == 1 or len(sparse_scores) == 1:
            return np.array([[1.0, 0.0], [0.0, 0.0]])  # Perfect correlation

        # Discretize scores into bins
        n_bins = min(10, len(dense_scores) // 5)  # Adaptive binning
        if n_bins < 2:
            n_bins = 2

        # Create bins for each score type - handle uniform scores
        dense_min, dense_max = np.min(dense_scores), np.max(dense_scores)
        sparse_min, sparse_max = np.min(sparse_scores), np.max(sparse_scores)

        # If all scores are the same, create artificial range
        if dense_min == dense_max:
            dense_min, dense_max = dense_min - 0.001, dense_max + 0.001
        if sparse_min == sparse_max:
            sparse_min, sparse_max = sparse_min - 0.001, sparse_max + 0.001

        dense_bins = np.linspace(dense_min, dense_max, n_bins + 1)
        sparse_bins = np.linspace(sparse_min, sparse_max, n_bins + 1)

        # Digitize scores
        dense_digitized = np.digitize(dense_scores, dense_bins) - 1
        sparse_digitized = np.digitize(sparse_scores, sparse_bins) - 1

        # Clip to valid range
        dense_digitized = np.clip(dense_digitized, 0, n_bins - 1)
        sparse_digitized = np.clip(sparse_digitized, 0, n_bins - 1)

        # Create joint distribution
        joint_dist = np.zeros((n_bins, n_bins))
        for i in range(len(dense_scores)):
            joint_dist[dense_digitized[i], sparse_digitized[i]] += 1

        # Normalize to probability distribution
        if np.sum(joint_dist) > 0:
            joint_dist = joint_dist / np.sum(joint_dist)
        else:
            joint_dist = np.ones((n_bins, n_bins)) / (n_bins * n_bins)

        return joint_dist

    def calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Kullback-Leibler divergence between two distributions.

        Args:
            p: First probability distribution
            q: Second probability distribution

        Returns:
            KL divergence value
        """
        # Ensure distributions are valid
        p = self._normalize_to_distribution(p)
        q = self._normalize_to_distribution(q)

        # Calculate KL divergence
        kl_div = np.sum(rel_entr(p, q))
        return float(kl_div)

    def calculate_js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon divergence between two distributions.

        Args:
            p: First probability distribution
            q: Second probability distribution

        Returns:
            JS divergence value (0-1)
        """
        # Ensure distributions are valid
        p = self._normalize_to_distribution(p)
        q = self._normalize_to_distribution(q)

        # Calculate mixture distribution
        m = 0.5 * (p + q)

        # Calculate JS divergence
        js_div = (
            0.5 * self.calculate_kl_divergence(p, m)
            + 0.5 * self.calculate_kl_divergence(q, m)
        )

        # Normalize to [0, 1] using sqrt
        return float(np.sqrt(js_div)) if js_div >= 0 else 0.0

    def get_information_report(
        self, dense_scores: np.ndarray, sparse_scores: np.ndarray
    ) -> str:
        """Generate a detailed information theory metrics report."""
        entropy_conf = self.calculate_entropy_confidence(
            dense_scores, sparse_scores
        )
        mutual_info_conf = self.calculate_mutual_information_confidence(
            dense_scores, sparse_scores
        )
        statistical_conf = self.calculate_statistical_significance(
            dense_scores, sparse_scores
        )

        report = f"""
Information Theory Metrics Report
================================

Entropy-based Confidence: {entropy_conf:.4f}
Mutual Information Confidence: {mutual_info_conf:.4f}
Statistical Significance Confidence: {statistical_conf:.4f}

Average Confidence: {(entropy_conf + mutual_info_conf + statistical_conf) / 3:.4f}
"""
        return report.strip()
