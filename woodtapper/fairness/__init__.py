"""Fairness module for WoodTapper."""

from .classification import OTFairBoostClassifier, OTXFairBoostClassifier
from .regression import OTFairBoostRegressor, OTXFairBoostRegressor

__all__ = [
    "OTFairBoostClassifier",
    "OTXFairBoostClassifier",
    "OTFairBoostRegressor",
    "OTXFairBoostRegressor",
]
