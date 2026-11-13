"""Fairness module for WoodTapper."""

from .classification import (
    OTFairBoostClassifier,
    OTHGBFairBoostClassifier,
    OTXFairBoostClassifier,
)
from .regression import (
    OTFairBoostRegressor,
    OTHGBFairBoostRegressor,
    OTXFairBoostRegressor,
)

__all__ = [
    "OTFairBoostClassifier",
    "OTHGBFairBoostClassifier",
    "OTXFairBoostClassifier",
    "OTFairBoostRegressor",
    "OTHGBFairBoostRegressor",
    "OTXFairBoostRegressor",
]
