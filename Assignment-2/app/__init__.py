import os
from flask import Flask


def create_app():
    """Application factory for the Flask app."""
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
    # store model path in config (relative to app package)
    app.config.setdefault('MODEL_PATH', os.path.join(os.path.dirname(__file__), 'models', 'pipeline.pkl'))

    # import routes so they are registered
    with app.app_context():
        from . import routes  # noqa: F401

    return app
