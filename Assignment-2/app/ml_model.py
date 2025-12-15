import os
import joblib
import pandas as pd


def _default_model_path():
    # default path inside package
    return os.path.join(os.path.dirname(__file__), 'models', 'pipeline.pkl')


def load_model(path=None):
    """Load saved sklearn pipeline. If path is None, use default path inside app/models."""
    if path is None:
        path = _default_model_path()
    # If caller passed a relative path, make it absolute relative to project root
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    model = joblib.load(path)
    return model


def predict_income(model, X_df: pd.DataFrame):
    """Return predicted income for a DataFrame of features.

    Returns a float for the single-row case or an array for multi-row input.
    """
    preds = model.predict(X_df)

    return preds
