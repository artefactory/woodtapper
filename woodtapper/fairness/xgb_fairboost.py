"""XGBoost-based fairness-aware estimators (requires the optional ``xgboost`` dependency)."""

import warnings

import numpy as np

from .objective import build_fair_loss
from .utils import _import_optional_dependency, _sigmoid

xgb = _import_optional_dependency("xgboost", extra="xgboost")
from xgboost.sklearn import _cls_predict_proba  # noqa: E402


class OTXFairBoostClassifier(xgb.XGBClassifier):
    def __init__(
        self,
        lambda_fairness_value,
        n_steps_cdf=1024,
        fairness_mode="Demographic_Parity",
        objective="binary:logistic",
        min_child_weight=0.0,
        reg_lambda=0.0,
        **kwargs,
    ):
        super().__init__(
            objective=objective,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            **kwargs,
        )
        self.lambda_fairness_value = lambda_fairness_value
        self.n_steps_cdf = n_steps_cdf
        self.fairness_mode = fairness_mode

    def get_xgb_params(self):
        """Get XGBoost parameters excluding fairness-specific ones."""
        params = super().get_xgb_params()
        for name in ("lambda_fairness_value", "n_steps_cdf", "fairness_mode"):
            params.pop(name, None)
        return params

    def fit(
        self,
        X,
        y,
        sensitive_attribute,
        sample_weight=None,
        base_margin=None,
        eval_set=None,
        verbose=None,
        xgb_model=None,
        sample_weight_eval_set=None,
        base_margin_eval_set=None,
    ):
        partitions_target = [(np.flatnonzero(y == 0), np.flatnonzero(y == 1))]
        fobj = build_fair_loss(
            sensitive_attribute=sensitive_attribute,
            partitions_target=partitions_target,
            lambda_fairness_value=self.lambda_fairness_value,
            fairness_mode=self.fairness_mode,
            n_steps_cdf=self.n_steps_cdf,
            is_classification=True,
        )
        # if self.objective is not None:
        #    raise ValueError(
        #        f"An explicit objective ('{self.objective}') was passed,but OTFairBoostClassifier overrides it with its custom fair loss objective."
        #    )
        if self.min_child_weight != 0.0:
            raise ValueError(
                "OTXFairBoostClassifier requires min_child_weight to be 0.0. Otherwise, the fairness objective may not be correctly applied."
            )
        if self.reg_lambda != 0.0:
            warnings.warn(
                "reg_lambda not set to 0.0, with min_child_weight=0.0, this may lead to the fairness term to dominate. Thus, produce low predictive performances."
            )
        self.set_params(objective=fobj)
        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            base_margin=base_margin,
            eval_set=eval_set,
            verbose=verbose,
            xgb_model=xgb_model,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin_eval_set=base_margin_eval_set,
        )
        return self

    def predict(
        self,
        X,
        output_margin=False,
        validate_features=True,
        base_margin=None,
        iteration_range=None,
    ):
        # with self._verbosity_context():
        class_probs = self.predict_proba(
            X=X,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
        )
        probs_class_1 = class_probs[:, 1]
        column_indexes = np.repeat(0, class_probs.shape[0])
        column_indexes[probs_class_1 > 0.5] = 1

        return column_indexes

    def predict_proba(
        self,
        X,
        validate_features=True,
        base_margin=None,
        iteration_range=None,
    ):
        """
        Modify slightly LGBM method to obtain real probas with our custom loss.
        # Link source: https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py
        """
        class_probs = super(xgb.XGBClassifier, self).predict(
            X=X,
            output_margin=True,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
        )
        class_probs = _sigmoid(class_probs)
        return _cls_predict_proba(self.n_classes_, class_probs, np.vstack)


class OTXFairBoostRegressor(xgb.XGBRegressor):
    def __init__(
        self,
        lambda_fairness_value,
        n_steps_cdf=1024,
        fairness_mode="Demographic_Parity",
        objective="reg:squarederror",
        min_child_weight=0.0,
        reg_lambda=0.0,
        **kwargs,
    ):
        super().__init__(
            objective=objective,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            **kwargs,
        )
        self.lambda_fairness_value = lambda_fairness_value
        self.n_steps_cdf = n_steps_cdf
        self.fairness_mode = fairness_mode

    def get_xgb_params(self):
        """Get XGBoost parameters excluding fairness-specific ones."""
        params = super().get_xgb_params()
        for name in ("lambda_fairness_value", "n_steps_cdf", "fairness_mode"):
            params.pop(name, None)
        return params

    def fit(
        self,
        X,
        y,
        sensitive_attribute,
        sample_weight=None,
        base_margin=None,
        eval_set=None,
        verbose=None,
        xgb_model=None,
        sample_weight_eval_set=None,
        base_margin_eval_set=None,
    ):
        partitions_target = [(np.flatnonzero(y == 0), np.flatnonzero(y == 1))]
        fobj = build_fair_loss(
            sensitive_attribute=sensitive_attribute,
            lambda_fairness_value=self.lambda_fairness_value,
            fairness_mode=self.fairness_mode,
            n_steps_cdf=self.n_steps_cdf,
            is_classification=False,
            partitions_target=partitions_target,
        )
        # if self.objective is not None:
        #    raise ValueError(
        #        f"An explicit objective ('{self.objective}') was passed,but OTFairBoostClassifier overrides it with its custom fair loss objective."
        #    )
        if self.min_child_weight != 0.0:
            raise ValueError(
                "OTXFairBoostRegressor requires min_child_weight to be 0.0. Otherwise, the fairness objective may not be correctly applied."
            )
        if self.reg_lambda != 0.0:
            warnings.warn(
                "reg_lambda not set to 0.0, with min_child_weight=0.0, this may lead to the fairness term to dominate. Thus, produce low predictive performances."
            )
        self.set_params(objective=fobj)
        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            base_margin=base_margin,
            eval_set=eval_set,
            verbose=verbose,
            xgb_model=xgb_model,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin_eval_set=base_margin_eval_set,
        )
        return self

    def predict(
        self,
        X,
        output_margin=False,
        validate_features=True,
        base_margin=None,
        iteration_range=None,
    ):
        """
        Modify slightly LGBM method to obtain real probas with our custom loss.
        # Link source: https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py
        """
        res = super(xgb.XGBRegressor, self).predict(
            X=X,
            output_margin=output_margin,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
        )
        return res
