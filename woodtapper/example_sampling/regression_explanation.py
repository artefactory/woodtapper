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
    """ExplanationExample RandomForestRegressor

    Extends RandomForestRegressor with methods for generating interpretable
    instance-level explanations.
    """


class ExtraTreesRegressorExplained(ExplanationMixin, ExtraTreesRegressor):
    """ExplanationExample ExtraTreesRegressor

    Extends ExtraTreesRegressor with methods for generating interpretable
    instance-level explanations.
    """


class GradientBoostingRegressorExplained(ExplanationMixin, GradientBoostingRegressor):
    """ExplanationExample GradientBoostingRegressor

    Extends GradientBoostingRegressor with methods for generating interpretable
    instance-level explanations.
    """
