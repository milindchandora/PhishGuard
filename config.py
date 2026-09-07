"""
config.py
----------
Central configuration. All secrets are read from environment variables
(or a local .env file) so nothing sensitive is hardcoded into source
control.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from a .env file if present


class Config:
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-key-change-me")
    MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/phishing_detector")
    DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
