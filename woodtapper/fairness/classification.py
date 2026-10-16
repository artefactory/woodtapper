import lightgbm as lgb
import numpy as np

from .objective import build_fair_loss
from .utils import _sigmoid


class OTFairBoostClassifier(lgb.LGBMClassifier):
    """
    OTFairBoostClassifier that incorporates fairness constraints into the LightGBM classifier.
    This classifier extends the LightGBM classifier by adding a custom fair loss objective that takes into account a sensitive attribute to enforce fairness constraints during training.
    Parameters
    ----------
    lambda_fairness_value : float
        The weight of the fairness constraint in the custom loss function.
    n_steps_cdf : int, default=1024
        The number of steps to use for approximating the cumulative distribution function (CDF) in the fairness loss.
    fairness_mode : str, default="Demographic_Parity"
        The fairness criterion to enforce. Options include "Demographic_Parity" and "Equalized_Opportunity".
    All other parameters are inherited from lightgbm.LGBMClassifier.
    """

    def __init__(
        self,
        lambda_fairness_value,
        n_steps_cdf=1024,
        fairness_mode="Demographic_Parity",
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        subsample_for_bin=200000,
        objective=None,
        class_weight=None,
        min_split_gain=0.0,
        min_child_weight=1e-3,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=None,
        n_jobs=None,
        importance_type="split",
        verbosity=-1,
    ):
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            importance_type=importance_type,
            verbosity=verbosity,
        )
        self.lambda_fairness_value = lambda_fairness_value
        self.n_steps_cdf = n_steps_cdf
        self.fairness_mode = fairness_mode
        self._initial_user_objective = objective  # Track whether the user explicitly supplied an objective at initialization

    def fit(
        self,
        X,
        y,
        sensitive_attribute,
        sample_weight=None,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_class_weight=None,
        eval_init_score=None,
        eval_metric=None,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
        eval_X=None,
        eval_y=None,
    ):
        """
        Fit the OTFairBoostClassifier model.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        sensitive_attribute : array-like of shape (n_samples,)
            The sensitive attribute used for enforcing fairness.
        sample_weight : array-like of shape (n_samples,), optional (default=None)
            Weights of the training samples.
        init_score : array-like of shape (n_samples,), optional (default=None)
            Initial scores of the training samples.
        eval_set : list of (X, y) tuples, optional (default=None)
            A list of validation sets for early stopping.
        eval_names : list of str, optional (default=None)
            Names of the validation sets.
        eval_sample_weight : list of array-like, optional (default=None)
            Sample weights for the validation sets.
        eval_class_weight : list of dict, optional (default=None)
            Class weights for the validation sets.
        eval_init_score : list of array-like, optional (default=None)
            Initial scores for the validation sets.
        eval_metric : str, callable, list of str or list of callable, optional (default=None)
            Evaluation metric(s) for the validation sets.
        feature_name : list of str or "auto", optional (default="auto")
            Names of features.
        categorical_feature : list of str, list of int or "auto", optional (default="auto")
            Categorical features.
        callbacks : list of callable, optional (default=None)
            List of callback functions.
        init_model : str, Booster, or LGBMModel, optional (default=None)
            Model used for initialization.
        eval_X : array-like or sparse matrix of shape (n_samples, n_features), optional (default=None)
            Validation features.
        eval_y : array-like of shape (n_samples,), optional (default=None)
            Validation target values.

        Returns
        -------
        self : object
            Fitted estimator.
        Notes
        -----
        This method overrides the default objective with a custom fair loss objective built using the sensitive attribute and fairness parameters.
        """
        fobj = build_fair_loss(
            sensitive_attribute=sensitive_attribute,
            lambda_fairness_value=self.lambda_fairness_value,
            fairness_mode=self.fairness_mode,
            n_steps_cdf=self.n_steps_cdf,
            is_classification=True,
        )
        if self._initial_user_objective is not None:
            raise ValueError(
                f"An explicit objective ('{self._initial_user_objective}') was passed,but OTFairBoostClassifier overrides it with its custom fair loss objective."
            )
        self.set_params(objective=fobj)
        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_class_weight=eval_class_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
            eval_X=eval_X,
            eval_y=eval_y,
        )

    def predict(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        validate_features=False,
        **kwargs,
    ):
        """
        Modify slightly LGBM method to obtain rzal probas with our custom loss.
        """
        result = self.predict_proba(
            X=X,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=validate_features,
            **kwargs,
        )
        # if callable(self._objective) or raw_score or pred_leaf or pred_contrib:
        if raw_score or pred_leaf or pred_contrib:
            return result
        else:
            class_index = np.argmax(result, axis=1)
            return self._le.inverse_transform(class_index)

    def predict_proba(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        validate_features=False,
        **kwargs,
    ):
        """
        Modify slightly LGBM method to obtain rzal probas with our custom loss.
        """
        result = super(lgb.LGBMClassifier, self).predict(
            X=X,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=validate_features,
            **kwargs,
        )
        # print("result before sigmoid: ",result)
        result = _sigmoid(result)
        # print("result after sigmoid: ",result)
        # if callable(self._objective) and not (raw_score or pred_leaf or pred_contrib):
        #    _log_warning(
        #        "Cannot compute class probabilities or labels "
        #        "due to the usage of customized objective function.\n"
        #        "Returning raw scores instead."
        #    )
        #    return result
        # if self.__is_multiclass or raw_score or pred_leaf or pred_contrib:  # type: ignore [operator]
        if raw_score or pred_leaf or pred_contrib:
            raise ValueError(
                "Cannot compute raw_score, pred_leaf, or pred_contrib with OTFairBoostClassifier."
            )

        return np.vstack((1.0 - result, result)).transpose()
