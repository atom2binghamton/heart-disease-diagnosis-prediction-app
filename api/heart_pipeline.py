"""
Shared ML pipeline components for the heart disease project.
Simplified to use only the top 3 correlated features: ca, cp, and thal.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# =============================================================================
# Custom transformer: ColumnDropper
# =============================================================================

class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Drops patient_id to prevent leakage while keeping it in the DataFrame 
    for DagsHub/MLflow signature consistency.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X
        return X.drop(columns=[col for col in self.columns if col in X.columns])

# =============================================================================
# Building blocks for preprocessing
# =============================================================================

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
)

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

def build_preprocessing():
    """
    Simplified preprocessing: Only uses 'ca', 'cp_id', and 'thal_id'.
    Drops 'patient_id' internally to ensure model doesn't cheat.
    """
    
    # Define the processor for the 3 selected features
    core_preprocessing = ColumnTransformer(
        [
            # Categorical features (3NF IDs)
            ("cat", cat_pipeline, ["cp_id", "thal_id"]),
            # Numerical feature
            ("num", default_num_pipeline, ["ca"]),
        ],
        # Drop EVERYTHING else (age, chol, trestbps, etc.)
        remainder="drop", 
    )

    # Wrap in a Pipeline that drops patient_id before feature selection
    full_pipeline = Pipeline([
        ("drop_id", ColumnDropper(columns=["patient_id"])),
        ("preprocessor", core_preprocessing),
    ])

    return full_pipeline

# =============================================================================
# Estimator factory
# =============================================================================

def make_estimator_for_name(name: str):
    if name == "logistic":
        return LogisticRegression(max_iter=1000)
    elif name == "histgradientboosting":
        return HistGradientBoostingClassifier(random_state=42)
    elif name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            random_state=42,
            n_estimators=100,
            max_depth=3,
            n_jobs=-1,
        )
    elif name == "lightgbm":
        return LGBMClassifier(
            random_state=42,
            n_estimators=100,
            num_leaves=10,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")