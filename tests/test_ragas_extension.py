"""
Tests for RAGAS Extension
"""

import pytest
import tempfile
import json
from src.hybrid_retrieval.ragas_extension import RAGASHybridExtension


class TestRAGASHybridExtension:
    """Test cases for RAGASHybridExtension class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extension = RAGASHybridExtension(alpha=0.1, beta=0.1, gamma=0.1)
        self.query = "What are the main components of machine learning?"
        self.retrieved_contexts = [
            "Machine learning consists of algorithms that learn from data.",
            "The main components include data, algorithms, and models.",
            "Supervised and unsupervised learning are key paradigms."
        ]
        self.generated_answer = "Machine learning has three main components: data, algorithms, and models."
        self.reference_answer = "Machine learning consists of data, algorithms, models, and evaluation metrics."
        self.dense_scores = [0.8, 0.7, 0.9]
        self.sparse_scores = [0.75, 0.72, 0.88]

    def test_initialization(self):
        """Test proper initialization of RAGAS hybrid extension."""
        assert hasattr(self.extension, 'hybrid_evaluator')
        assert hasattr(self.extension, 'info_theory')
        assert hasattr(self.extension, 'weight_optimizer')
        assert self.extension.hybrid_evaluator.alpha == 0.1

    def test_evaluate_hybrid_retrieval(self):
        """Test hybrid retrieval evaluation."""
        results = self.extension.evaluate_hybrid_retrieval(
            query=self.query,
            retrieved_contexts=self.retrieved_contexts,
            generated_answer=self.generated_answer,
            reference_answer=self.reference_answer,
            dense_scores=self.dense_scores,
            sparse_scores=self.sparse_scores
        )

        assert isinstance(results, dict)
        assert 'hybrid_precision' in results
        assert 'hybrid_confidence' in results
        assert 'entropy_confidence' in results
        assert 'mutual_information_confidence' in results
        assert 'statistical_confidence' in results
        assert 'dense_weight' in results
        assert 'sparse_weight' in results
        assert 'uncertainty_penalty' in results

    def test_standard_ragas_metrics_calculation(self):
        """Test standard RAGAS metrics calculation."""
        results = self.extension._calculate_standard_ragas_metrics(
            query=self.query,
            contexts=self.retrieved_contexts,
            generated_answer=self.generated_answer,
            reference_answer=self.reference_answer
        )

        assert isinstance(results, dict)
        assert 'context_precision' in results
        assert 'context_recall' in results
        assert 'faithfulness' in results
        assert 'answer_relevancy' in results
        # These are placeholder values in the current implementation
        assert all(v == 0.0 for v in results.values())

    def test_hybrid_metrics_calculation(self):
        """Test hybrid-specific metrics calculation."""
        results = self.extension._calculate_hybrid_metrics(
            dense_scores=self.dense_scores,
            sparse_scores=self.sparse_scores,
            query=self.query
        )

        assert isinstance(results, dict)
        assert 'hybrid_precision' in results
        assert 'hybrid_confidence' in results
        assert 'entropy_confidence' in results
        assert 'mutual_information_confidence' in results
        assert 'statistical_confidence' in results
        assert 'dense_weight' in results
        assert 'sparse_weight' in results
        assert 'uncertainty_penalty' in results

        # Check confidence ranges
        assert 0 <= results['hybrid_confidence'] <= 1
        assert 0 <= results['entropy_confidence'] <= 1
        assert 0 <= results['mutual_information_confidence'] <= 1
        assert 0 <= results['statistical_confidence'] <= 1

    def test_compare_hybrid_vs_standard(self):
        """Test comparison between hybrid and standard retrieval results."""
        dense_results = {'context_precision': 0.7}
        sparse_results = {'context_precision': 0.6}
        hybrid_results = {'hybrid_precision': 0.8}

        comparison = self.extension.compare_hybrid_vs_standard(
            dense_results, sparse_results, hybrid_results
        )

        assert isinstance(comparison, dict)
        assert 'improvement_vs_dense' in comparison
        assert 'improvement_vs_sparse' in comparison
        assert 'hybrid_confidence_score' in comparison

        # Check improvement calculations
        expected_dense_improvement = ((0.8 - 0.7) / 0.7) * 100
        expected_sparse_improvement = ((0.8 - 0.6) / 0.6) * 100
        assert abs(comparison['improvement_vs_dense'] - expected_dense_improvement) < 0.1
        assert abs(comparison['improvement_vs_sparse'] - expected_sparse_improvement) < 0.1

    def test_compare_hybrid_vs_standard_missing_metrics(self):
        """Test comparison with missing metrics."""
        # Test with missing context_precision
        dense_results = {'other_metric': 0.7}
        sparse_results = {'context_precision': 0.6}
        hybrid_results = {'hybrid_precision': 0.8}

        comparison = self.extension.compare_hybrid_vs_standard(
            dense_results, sparse_results, hybrid_results
        )

        assert isinstance(comparison, dict)
        # Should only calculate improvement_vs_sparse since dense_results lacks context_precision
        assert 'improvement_vs_dense' not in comparison
        assert 'improvement_vs_sparse' in comparison

    def test_hybrid_report_generation(self):
        """Test hybrid evaluation report generation."""
        results = {
            'hybrid_precision': 0.85,
            'hybrid_confidence': 0.78,
            'entropy_confidence': 0.8,
            'mutual_information_confidence': 0.75,
            'statistical_confidence': 0.79,
            'dense_weight': 0.65,
            'sparse_weight': 0.35,
            'uncertainty_penalty': 0.15
        }

        report = self.extension.generate_hybrid_report(results)

        assert isinstance(report, str)
        assert "Hybrid Retrieval Evaluation Report (RAGAS Extension)" in report
        assert f"Hybrid Precision Score: {results['hybrid_precision']:.4f}" in report
        assert f"Hybrid Confidence: {results['hybrid_confidence']:.4f}" in report

    def test_get_recommendations_high_confidence(self):
        """Test recommendations for high confidence results."""
        results = {
            'hybrid_confidence': 0.85,
            'uncertainty_penalty': 0.1,
            'dense_weight': 0.6,
            'sparse_weight': 0.4
        }

        recommendations = self.extension.get_recommendations(results)

        assert isinstance(recommendations, list)
        # Should return default recommendation since confidence is high
        assert recommendations == ["Hybrid retrieval configuration looks good"]

    def test_get_recommendations_low_confidence(self):
        """Test recommendations for low confidence results."""
        results = {
            'hybrid_confidence': 0.3,
            'uncertainty_penalty': 0.4,
            'dense_weight': 0.8,
            'sparse_weight': 0.2
        }

        recommendations = self.extension.get_recommendations(results)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 1  # Should have multiple recommendations
        assert any("confidence" in rec.lower() for rec in recommendations)
        assert any("uncertainty" in rec.lower() for rec in recommendations)
        assert any("weight imbalance" in rec.lower() for rec in recommendations)

    def test_export_and_load_results(self):
        """Test results export and load functionality."""
        results = {
            'hybrid_precision': 0.85,
            'hybrid_confidence': 0.78,
            'entropy_confidence': 0.8,
            'mutual_information_confidence': 0.75,
            'statistical_confidence': 0.79,
            'dense_weight': 0.65,
            'sparse_weight': 0.35,
            'uncertainty_penalty': 0.15
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name

        try:
            # Export results
            self.extension.export_results(results, temp_filename)

            # Load results
            loaded_results = self.extension.load_results(temp_filename)

            assert isinstance(loaded_results, dict)
            assert loaded_results == results

        finally:
            import os
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def test_edge_case_empty_contexts(self):
        """Test behavior with empty contexts."""
        results = self.extension.evaluate_hybrid_retrieval(
            query=self.query,
            retrieved_contexts=[],
            generated_answer=self.generated_answer,
            reference_answer=self.reference_answer,
            dense_scores=[],
            sparse_scores=[]
        )

        assert isinstance(results, dict)
        # Should still return valid results even with empty inputs
        assert 'hybrid_precision' in results

    def test_edge_case_single_context(self):
        """Test behavior with single context."""
        results = self.extension.evaluate_hybrid_retrieval(
            query=self.query,
            retrieved_contexts=["Single context"],
            generated_answer=self.generated_answer,
            reference_answer=self.reference_answer,
            dense_scores=[0.8],
            sparse_scores=[0.75]
        )

        assert isinstance(results, dict)
        assert 'hybrid_precision' in results
        assert 'hybrid_confidence' in results

    def test_edge_case_identical_scores(self):
        """Test behavior with identical dense and sparse scores."""
        identical_scores = [0.8, 0.7, 0.9]
        results = self.extension.evaluate_hybrid_retrieval(
            query=self.query,
            retrieved_contexts=self.retrieved_contexts,
            generated_answer=self.generated_answer,
            reference_answer=self.reference_answer,
            dense_scores=identical_scores,
            sparse_scores=identical_scores
        )

        assert isinstance(results, dict)
        assert 'hybrid_precision' in results
        assert 'uncertainty_penalty' in results
        # Low uncertainty penalty expected for identical scores
        assert results['uncertainty_penalty'] < 0.1

    def test_static_load_results_method(self):
        """Test static load_results method."""
        results = {
            'test_metric': 0.5,
            'another_metric': 0.8
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(results, f)
            temp_filename = f.name

        try:
            # Test static method
            loaded_results = RAGASHybridExtension.load_results(temp_filename)

            assert isinstance(loaded_results, dict)
            assert loaded_results == results

        finally:
            import os
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def test_empty_comparison_results(self):
        """Test comparison with empty or missing results."""
        empty_results = {}
        hybrid_results = {'hybrid_precision': 0.8}

        comparison = self.extension.compare_hybrid_vs_standard(
            empty_results, empty_results, hybrid_results
        )

        assert isinstance(comparison, dict)
        # Should only contain hybrid confidence score
        assert 'hybrid_confidence_score' in comparison
        assert len(comparison) == 1

    def test_recommendations_with_moderate_results(self):
        """Test recommendations with moderate confidence and penalty values."""
        results = {
            'hybrid_confidence': 0.6,
            'uncertainty_penalty': 0.2,
            'dense_weight': 0.55,
            'sparse_weight': 0.45
        }

        recommendations = self.extension.get_recommendations(results)

        assert isinstance(recommendations, list)
        # Should have some recommendations but not the default "looks good"
        assert recommendations != ["Hybrid retrieval configuration looks good"]
        assert len(recommendations) >= 1