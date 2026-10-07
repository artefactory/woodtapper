import lightgbm as lgb

from .objective import build_fair_loss


class OTFairBoostRegressor(lgb.LGBMRegressor):
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
        eval_init_score=None,
        eval_metric=None,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
        eval_X=None,
        eval_y=None,
    ):
        fobj = build_fair_loss(
            sensitive_attribute=sensitive_attribute,
            lambda_fairness_value=self.lambda_fairness_value,
            fairness_mode=self.fairness_mode,
            n_steps_cdf=self.n_steps_cdf,
            is_classification=False,
        )
        if self._initial_user_objective is not None:
            raise ValueError(
                f"An explicit objective ('{self._initial_user_objective}') was passed,but OTFairBoostClassifier overrides it with its custom fair loss objective."
            )
        self.set_params(objective=fobj)
        return super().fit(X, y)

    def predict_(
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
        result = super(lgb.LGBMRegressor, self).predict(
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
        # result = _sigmoid(result)
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

        return result
