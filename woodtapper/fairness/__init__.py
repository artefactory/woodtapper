"""Fairness module for WoodTapper."""

from .classification import OTFairBoostClassifier
from .regression import OTFairBoostRegressor

__all__ = ["OTFairBoostClassifier", "OTFairBoostRegressor"]
