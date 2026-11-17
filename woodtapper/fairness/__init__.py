"""Fairness module for WoodTapper.

``OTHGBFairBoost*`` estimators only rely on scikit-learn. ``OTFairBoost*`` estimators
require the optional ``lightgbm`` dependency and ``OTXFairBoost*`` estimators require
the optional ``xgboost`` dependency; they are imported lazily on first access.
"""

import importlib

from .hgb_fairboost import OTHGBFairBoostClassifier, OTHGBFairBoostRegressor

_LAZY_ESTIMATORS = {
    "OTFairBoostClassifier": "lgbm_fairboost",
    "OTFairBoostRegressor": "lgbm_fairboost",
    "OTXFairBoostClassifier": "xgb_fairboost",
    "OTXFairBoostRegressor": "xgb_fairboost",
}

__all__ = [
    "OTFairBoostClassifier",
    "OTHGBFairBoostClassifier",
    "OTXFairBoostClassifier",
    "OTFairBoostRegressor",
    "OTHGBFairBoostRegressor",
    "OTXFairBoostRegressor",
]


def __getattr__(name):
    if name in _LAZY_ESTIMATORS:
        module = importlib.import_module(f".{_LAZY_ESTIMATORS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
