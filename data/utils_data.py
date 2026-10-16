import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def make_stratify_labels(y, sensitive_attribute):
    """
    Build stratification labels for CV.
    Stratify on joint labels (y, a) to stabilize
    subgroup proportions across folds. If some joint labels are too rare,
    fallback to stratifying on y only.
    """
    if isinstance(y, pd.Series):
        y_arr = y.to_numpy()
    else:
        y_arr = y

    if isinstance(sensitive_attribute, pd.Series):
        joint = (y.astype(str) + "__" + sensitive_attribute.astype(str)).to_numpy()
    else:
        joint = y.astype(str) + "__" + sensitive_attribute.astype(str)  # .to_numpy()

    # If very rare joint labels exist, stratification can fail; fallback to y-only
    _, counts = np.unique(joint, return_counts=True)
    if counts.size == 0 or np.min(counts) < 2:
        return y_arr

    return joint


def make_preprocessor_adult(numeric_cols, categorical_cols):
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor
