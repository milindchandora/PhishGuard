"""
app.py
-------
Flask entry point for the Phishing Website Detection System.

Routes:
  GET  /                 Landing page (with quick URL check form)
  GET  /about            About / how it works
  POST /check            Form-based URL check (used by the home page)
  POST /api/predict      JSON API: {"url": "..."} -> prediction + confidence
  GET  /register         Registration form
  POST /register         Create account
  GET  /login            Login form
  POST /login            Authenticate
  GET  /logout           Log out
  GET  /dashboard        Logged-in user's stats + recent scans
  GET  /history           Logged-in user's full scan history

The trained model is loaded once at startup. If it's missing, the app
still runs so the rest of the UI can be demoed, but prediction routes
return a clear error explaining how to train the model.
"""

import os
import re
import sys
from urllib.parse import urlparse

from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify, session,
)
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user, login_required, current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd

from config import Config
import db

sys.path.append(os.path.join(os.path.dirname(__file__), "ml"))
from feature_extraction import extract_features, FEATURE_NAMES  # noqa: E402

app = Flask(__name__, static_folder='public', static_url_path='')
app.config.from_object(Config)

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "warning"

# ---------------------------------------------------------------------
# Load ML model + scaler once at startup
# ---------------------------------------------------------------------
MODEL_PATH = os.path.join(Config.MODEL_DIR, "phishing_model.pkl")
SCALER_PATH = os.path.join(Config.MODEL_DIR, "scaler.pkl")

model = None
scaler = None
model_load_error = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as exc:  # noqa: BLE001
    model_load_error = (
        "ML model not found or failed to load. Run "
        "'python ml/build_sample_dataset.py' then 'python ml/train_model.py' "
        f"to generate it. (Details: {exc})"
    )


# ---------------------------------------------------------------------
# Auth: minimal User wrapper for flask-login
# ---------------------------------------------------------------------
class User(UserMixin):
    def __init__(self, doc):
        self.id = str(doc["_id"])
        self.username = doc["username"]
        self.email = doc.get("email", "")


@login_manager.user_loader
def load_user(user_id):
    try:
        doc = db.find_user_by_id(user_id)
        return User(doc) if doc else None
    except Exception:  # noqa: BLE001 - DB may be unavailable
        return None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
URL_REGEX = re.compile(
    r"^(?:http|https)?://?"
    r"(?:\S+(?::\S*)?@)?"
    r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}"
    r"(?::\d+)?(?:[/?#]\S*)?$"
)


def is_valid_url(raw_url: str) -> bool:
    if not raw_url or len(raw_url) > 2048:
        return False
    candidate = raw_url if "://" in raw_url else "http://" + raw_url
    parsed = urlparse(candidate)
    if not parsed.hostname:
        return False
    return bool(URL_REGEX.match(raw_url)) or "." in parsed.hostname


def run_prediction(raw_url: str):
    """
    Extracts features from raw_url, scales them, and returns
    (label_str, confidence_float, features_dict).
    Raises RuntimeError if the model isn't loaded.
    """
    if model is None or scaler is None:
        raise RuntimeError(model_load_error or "Model not available.")

    features = extract_features(raw_url)
    X = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    label = "Phishing" if int(pred) == 1 else "Legitimate"

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        confidence = float(max(proba))
    else:
        confidence = 1.0  # model has no probability estimate

    return label, round(confidence, 4), features


# ---------------------------------------------------------------------
# Public pages
# ---------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", model_ready=model is not None)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/check", methods=["POST"])
def check():
    raw_url = request.form.get("url", "").strip()

    if not raw_url:
        flash("Please enter a URL to check.", "danger")
        return redirect(url_for("index"))

    if not is_valid_url(raw_url):
        flash("That doesn't look like a valid URL. Please try again.", "danger")
        return redirect(url_for("index"))

    try:
        label, confidence, features = run_prediction(raw_url)
    except RuntimeError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("index"))
    except Exception as exc:  # noqa: BLE001
        flash(f"Something went wrong while analyzing the URL: {exc}", "danger")
        return redirect(url_for("index"))

    # Save to history if logged in and DB is reachable
    if current_user.is_authenticated:
        try:
            db.save_scan(current_user.id, raw_url, label, confidence, features)
        except Exception:  # noqa: BLE001
            flash("Result shown below, but it could not be saved to your "
                  "history (database unavailable).", "warning")

    return render_template(
        "result.html", url=raw_url, prediction=label,
        confidence=confidence, features=features,
    )


# ---------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    raw_url = str(data.get("url", "")).strip()

    if not raw_url:
        return jsonify({"error": "Missing 'url' field."}), 400
    if not is_valid_url(raw_url):
        return jsonify({"error": "Invalid URL format."}), 400

    try:
        label, confidence, _ = run_prediction(raw_url)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify({"url": raw_url, "prediction": label, "confidence": confidence})


# ---------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm_password", "")

    if not username or not email or not password:
        flash("All fields are required.", "danger")
        return redirect(url_for("register"))
    if len(password) < 8:
        flash("Password must be at least 8 characters.", "danger")
        return redirect(url_for("register"))
    if password != confirm:
        flash("Passwords do not match.", "danger")
        return redirect(url_for("register"))

    try:
        if db.username_or_email_exists(username, email):
            flash("Username or email already registered.", "danger")
            return redirect(url_for("register"))

        password_hash = generate_password_hash(password)
        db.create_user(username, email, password_hash)
        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))
    except Exception as exc:  # noqa: BLE001
        flash(f"Registration failed (database unavailable): {exc}", "danger")
        return redirect(url_for("register"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    try:
        doc = db.find_user_by_username(username)
    except Exception as exc:  # noqa: BLE001
        flash(f"Login failed (database unavailable): {exc}", "danger")
        return redirect(url_for("login"))

    if doc and check_password_hash(doc["password_hash"], password):
        login_user(User(doc))
        flash(f"Welcome back, {doc['username']}!", "success")
        return redirect(url_for("dashboard"))

    flash("Invalid username or password.", "danger")
    return redirect(url_for("login"))


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------
# Dashboard + history (require login)
# ---------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    try:
        stats = db.get_user_stats(current_user.id)
        recent = db.get_user_scans(current_user.id, limit=5)
    except Exception:  # noqa: BLE001
        stats = {"total": 0, "safe": 0, "phishing": 0}
        recent = []
        flash("Could not load dashboard data (database unavailable).", "warning")

    return render_template("dashboard.html", stats=stats, recent=recent)


@app.route("/history")
@login_required
def history():
    try:
        scans = db.get_user_scans(current_user.id, limit=200)
    except Exception:  # noqa: BLE001
        scans = []
        flash("Could not load scan history (database unavailable).", "warning")

    return render_template("history.html", scans=scans)


# ---------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_e):
    return render_template("error.html", code=404, message="Page not found."), 404


@app.errorhandler(500)
def server_error(_e):
    return render_template("error.html", code=500, message="Internal server error."), 500


if __name__ == "__main__":
    app.run(debug=Config.DEBUG, host="127.0.0.1", port=5000)
