"""
Hybrid Retrieval Evaluation Method

This package implements the Hybrid Precision evaluation method for hybrid
retrieval systems, using information theory-driven multi-dimensional
confidence assessment framework.

Author: Jingxuan Qi
Email: 1312750677@qq.com
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Jingxuan Qi"
__email__ = "1312750677@qq.com"
__license__ = "MIT"

from .hybrid_precision import HybridPrecisionEvaluator
from .information_theory import InformationTheoryMetrics
from .adaptive_weights import AdaptiveWeightOptimizer
from .ragas_extension import RAGASHybridExtension

__all__ = [
    "HybridPrecisionEvaluator",
    "InformationTheoryMetrics",
    "AdaptiveWeightOptimizer",
    "RAGASHybridExtension"
]
