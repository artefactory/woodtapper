"""

ExampleExplanation for regression.

"""

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from .base import ExplanationMixin


class RandomForestRegressorExplained(ExplanationMixin, RandomForestRegressor):
    """ExplanationExample RandomForestRegressor"""


class ExtraTreesRegressorExplained(ExplanationMixin, ExtraTreesRegressor):
    """ExplanationExample ExtraTreesRegressor"""


class GradientBoostingRegressorExplained(ExplanationMixin, GradientBoostingRegressor):
    """ExplanationExample GradientBoostingRegressor"""
