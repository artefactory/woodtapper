"""
Example-based explanations for classification tree-based models.

This module provides wrapper classes that extend scikit-learn classifiers
with example-based explainability capabilities through the ExplanationMixin.
"""

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from .base import ExplanationMixin


class RandomForestClassifierExplained(ExplanationMixin, RandomForestClassifier):
    """Random Forest classifier with example-based explanations.

    Extends RandomForestClassifier with methods for generating interpretable
    instance-level explanations.
    """


class ExtraTreesClassifierExplained(ExplanationMixin, ExtraTreesClassifier):
    """Extremely Randomized Trees classifier with example-based explanations.

    Extends ExtraTreesClassifier with methods for generating interpretable
    instance-level explanations.
    """


class GradientBoostingClassifierExplained(ExplanationMixin, GradientBoostingClassifier):
    """Gradient Boosting classifier with example-based explanations.

    Extends GradientBoostingClassifier with methods for generating interpretable
    instance-level explanations.
    """
