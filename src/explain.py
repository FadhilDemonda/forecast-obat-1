"""Interpretabilitas: SHAP (jika tersedia) atau feature importance bawaan XGBoost."""
import numpy as np
import pandas as pd

from .features import FEATURES


def feature_impact(model, X: pd.DataFrame) -> pd.DataFrame:
    features = list(getattr(model, "feature_names_in_", FEATURES))
    X = X[features]
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(X)
        imp = np.abs(sv).mean(axis=0)
        src = "SHAP (mean |SHAP value|)"
    except Exception:
        imp = np.asarray(model.feature_importances_, float)
        src = "XGBoost feature importance (gain)"
    out = pd.DataFrame({"fitur": features, "dampak": imp}).sort_values("dampak", ascending=False)
    out.attrs["source"] = src
    return out.reset_index(drop=True)
