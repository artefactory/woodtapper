# Fairness-aware classification

This tutorial compares a standard LightGBM classifier with `OTFairBoostClassifier` on the Adult income dataset. The sensitive attribute (`sex`) is used during fair-model training and evaluation, but is excluded from the predictive features.


## Import modules

```python
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from woodtapper.fairness import OTFairBoostClassifier
from woodtapper.fairness.metrics import derive_fairness_metrics, make_preprocessor_adult
```

## Load and prepare data


```python
adult = fetch_openml("adult", version=2, as_frame=True)

X = adult.data.drop(columns=["sex"])
y = adult.target.map({"<=50K": 0, ">50K": 1}).astype(int)
sensitive_attribute = adult.data["sex"].map({"Male": 0, "Female": 1}).astype(int)

numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_cols = [column for column in X.columns if column not in numeric_cols]

joint_labels = y.astype(str) + "__" + sensitive_attribute.astype(str)
(
	X_train,
	X_test,
	y_train,
	y_test,
	sensitive_attribute_train,
	sensitive_attribute_test,
) = train_test_split(
	X,
	y,
	sensitive_attribute,
	test_size=0.2,
	random_state=0,
	stratify=joint_labels,
)
```

## Train OTFairBoostClassifier


```python
fair_classifier = OTFairBoostClassifier(
	n_estimators=1000,
	lambda_fairness_value=1.0,
	fairness_mode="Demographic_Parity",
	learning_rate=0.1,
	num_leaves=64,
	min_child_samples=50,
	colsample_bytree=0.9,
	subsample=0.9,
	subsample_freq=1,
	random_state=42,
	n_jobs=8,
	verbosity=-1,
)

preprocessor = make_preprocessor_adult(numeric_cols, categorical_cols)
ot_model_pipeline = Pipeline(
	steps=[("preprocessor", preprocessor), ("classifier", fair_classifier)]
)
ot_model_pipeline.fit(
	X_train,
	y_train,
	classifier__sensitive_attribute=sensitive_attribute_train,
)
```

## Evaluate both models


```python
y_preds = ot_model_pipeline.predict(X_test)
y_preds_probas = ot_model_pipeline.predict_proba(X_test)
pr_auc = average_precision_score(y_true=y_test, y_score=y_preds_probas[:, 1])
roc_auc = roc_auc_score(y_true=y_test, y_score=y_preds_probas[:, 1])
accuracy = accuracy_score(y_true=y_test, y_pred=y_preds)
print("### Predictive Perfs ###")
print("PR AUC:", pr_auc)
print("ROC AUC:", roc_auc)
print("Accuracy:", accuracy)

dp_gap, di, eopp_gap, fpr_gap, ppv_gap, rates_df=derive_fairness_metrics(y_true=y_test.to_numpy().ravel(),
                                                                         y_preds=y_preds,
                                                                         sensitive_attribute=sensitive_attribute_test.to_numpy().ravel(), min_group_n=10)
print("Fairness Perfs")
print("DP GAP:", dp_gap)
print("DI:", di)
print("EOPP GAP:", eopp_gap)
```

!!! example "Output"
```text
### Predictive Perfs ###
PR AUC: 0.7846302715892377
ROC AUC: 0.9054975002990155
Accuracy: 0.8546422356433616
Fairness Perfs
DP GAP: 0.07666049274730133
DI: 0.6381735443399936
EOPP GAP: 0.16029435661722036
```
