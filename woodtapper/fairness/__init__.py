"""Fairness module for WoodTapper."""

from .classification import OTFairBoostClassifier, OTXFairBoostClassifier
from .regression import OTFairBoostRegressor

__all__ = ["OTFairBoostClassifier", "OTXFairBoostClassifier", "OTFairBoostRegressor"]
