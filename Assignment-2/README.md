# Income prediction Flask app (SVR)

This project demonstrates a simple Flask app that predicts an income value using an SVM regressor (SVR) trained on the provided `DATA.csv`.

Quick steps

1. Create and activate the virtual environment (optional but recommended):

   python3 -m venv .venv
   source .venv/bin/activate

2. Install dependencies:

   python3 -m pip install -r requirements.txt

3. Train the model (saves pipeline to `app/models/pipeline.pkl`):

   python3 train_model.py

4. Run the app:

   python3 run.py

The web UI is available at <http://127.0.0.1:5001> and shows a simple form for the input features. The form options for categorical inputs are populated from `DATA.csv` so they match the training data.
