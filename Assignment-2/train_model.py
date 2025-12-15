import os
import joblib
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    root = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(root, 'DATA.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"DATA.csv not found at {data_path}")

    df = pd.read_csv(data_path)

    if 'Income' not in df.columns:
        raise RuntimeError('DATA.csv must contain Income column as target')

    X = df[[c for c in df.columns if c != 'Income']].copy()
    y = df['Income'].values

    # numeric and categorical split (based on known column names in this dataset)
    numeric_features = ['Age', 'Capital Gain', 'Capital Loss', 'Hours Per Week']
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ],
        remainder='drop'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('svr', SVR())
    ])

    # split to get a quick evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

    print('Training model...')
    pipeline.fit(X_train, y_train)

    # evaluate
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f'Training finished. MAE={mae:.4f}, R²={r2:.4f}')

    # save model
    model_dir = os.path.join(root, 'app', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'pipeline.pkl')
    joblib.dump(pipeline, model_path)
    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
