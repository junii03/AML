import os
import pandas as pd
from flask import current_app, render_template, request

from .ml_model import load_model, predict_income


def _project_root():
    # one level above the app package
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


@current_app.route('/', methods=['GET', 'POST'])
def index():
    """Show form and handle predictions."""
    project_root = _project_root()
    data_csv = os.path.join(project_root, 'DATA.csv')

    # Load dataset for UI options
    df = pd.read_csv(data_csv)

    # All columns except the target
    feature_cols = [c for c in df.columns if c != 'Income']

    # determine numeric vs categorical columns
    numeric_cols = ['Age', 'Capital Gain', 'Capital Loss', 'Hours Per Week']
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # prepare choices for selects
    choices = {col: sorted(df[col].dropna().unique().tolist()) for col in categorical_cols}

    prediction = None
    if request.method == 'POST':
        # gather inputs
        input_data = {}
        for col in feature_cols:
            val = request.form.get(col)
            if col in numeric_cols:
                # Avoid calling float() on None. Treat empty strings as missing.
                if val is None or val == '':
                    input_data[col] = None
                else:
                    try:
                        input_data[col] = float(val)
                    except (ValueError, TypeError):
                        input_data[col] = None
            else:
                input_data[col] = val

        # create single-row DataFrame
        X = pd.DataFrame([input_data], columns=feature_cols)

        # load model
        model_path = current_app.config.get('MODEL_PATH')
        try:
            model = load_model(model_path)
            preds = predict_income(model, X)
            # single value expected
            prediction = float(preds[0]) if hasattr(preds, '__len__') else float(preds)
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', numeric_cols=numeric_cols, categorical_cols=categorical_cols, choices=choices, prediction=prediction)
