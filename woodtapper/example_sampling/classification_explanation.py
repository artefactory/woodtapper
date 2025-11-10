"""

ExampleExplanation for classification.

"""

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from .base import ExplanationMixin


class RandomForestClassifierExplained(ExplanationMixin, RandomForestClassifier):
    """ExplanationExample RandomForestClassifier"""


class ExtraTreesClassifierExplained(ExplanationMixin, ExtraTreesClassifier):
    """ExplanationExample ExtraTreesClassifier"""
