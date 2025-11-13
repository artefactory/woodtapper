"""

ExampleExplanation for classification.

"""

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from .base import ExplanationMixin


class RandomForestClassifierExplained(ExplanationMixin, RandomForestClassifier):
    """ExplanationExample RandomForestClassifier"""


class ExtraTreesClassifierExplained(ExplanationMixin, ExtraTreesClassifier):
    """ExplanationExample ExtraTreesClassifier"""


class GradientBoostingClassifierExplained(ExplanationMixin, GradientBoostingClassifier):
    """ExplanationExample GradientBoostingClassifier"""
