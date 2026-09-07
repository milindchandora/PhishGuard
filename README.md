# PhishGuard — Phishing Website Detection System Using Machine Learning

A complete final-year major project: a web application that predicts
whether a URL is **Legitimate/Safe** or **Phishing/Malicious**, using
URL-based feature extraction and a trained scikit-learn classifier.

---

## 1. Abstract

Phishing websites impersonate trusted brands to steal credentials and
payment information. PhishGuard detects these sites by analyzing the
*structure* of a URL — length, subdomain depth, special-character
density, presence of an IP address, brand-imitation keywords, and more
— rather than the page content, so a link can be checked instantly and
safely, without visiting it. A Flask web app lets a user paste a URL,
runs it through a trained ML model, and returns a clear Safe/Phishing
verdict with a confidence score. Registered users get a dashboard and
a searchable scan history stored in MongoDB.

## 2. Problem Statement

Manually spotting a phishing URL requires attention most people don't
have time for in the moment they're about to click a link. An
automated, lightweight classifier that scores a URL in milliseconds,
using only information available before the page loads, reduces that
burden.

## 3. Objectives

- Extract a consistent set of numeric features from any URL.
- Train and compare several ML classifiers on those features.
- Select and persist the best-performing model.
- Serve real-time predictions through a web UI and a JSON API.
- Let users track their scan history and see summary statistics.

## 4. Scope

In scope: URL-structure-based classification, a demo dataset and
training pipeline, a Flask web app, MongoDB-backed auth and history,
and a `/api/predict` endpoint. Out of scope: fetching/rendering live
page content, browser-extension integration, and real-time threat-feed
ingestion — noted under Future Scope below.

## 5. Technologies Used

| Layer      | Technology                                          |
|------------|------------------------------------------------------|
| Frontend   | HTML5, CSS3 (custom, no framework), vanilla JS       |
| Backend    | Python 3, Flask, Flask-Login                         |
| ML         | pandas, NumPy, scikit-learn, joblib, matplotlib      |
| Database   | MongoDB (pymongo)                                     |
| Auth       | Werkzeug password hashing, server-side sessions       |

## 6. System Architecture

```
Browser ──HTTP──▶ Flask app (app.py)
                     │
                     ├─▶ feature_extraction.py  (URL → feature dict)
                     │        used identically at train time and
                     │        predict time, so there is no train/serve
                     │        skew
                     │
                     ├─▶ model/phishing_model.pkl + scaler.pkl
                     │        (loaded once at startup)
                     │
                     └─▶ db.py ──▶ MongoDB (users, scans)
```

Training happens offline (`ml/train_model.py`) and writes the model
artifacts the Flask app loads at startup — the web app never trains a
model live.

## 7. Machine Learning Methodology

1. **Load** `data/urls_dataset.csv` (columns: `url`, `label`; label 1 =
   phishing, 0 = legitimate).
2. **Clean**: drop empty/duplicate rows, coerce types.
3. **Feature extraction**: every URL → a 27-dimensional numeric vector
   via `ml/feature_extraction.py`.
4. **Split**: 80/20 stratified train/test split.
5. **Scale**: `StandardScaler` fit on the training set only.
6. **Train & compare**: Logistic Regression, Decision Tree, Random
   Forest, K-Nearest Neighbors, SVM, and Naive Bayes, all on the same
   folds.
7. **Evaluate**: accuracy, precision, recall, F1, confusion matrix for
   each model, computed from real predictions on the held-out test
   set — nothing is hardcoded.
8. **Select**: the model with the highest F1 score is chosen (F1
   balances false positives — blocking a safe site — against false
   negatives — missing a phishing site).
9. **Save**: `model/phishing_model.pkl`, `model/scaler.pkl`,
   `model/feature_names.pkl`, plus `evaluation_report.json/.txt` and
   two comparison charts.

## 8. Feature Extraction

See `ml/feature_extraction.py` for the full list (27 features),
including: URL/hostname/path/query length, counts of dots, hyphens,
digits, special characters and subdomains, presence of an IP address,
HTTPS usage, `@` symbol, `//` redirection tricks, known URL-shortener
domains, suspicious brand-imitation keywords, and Shannon entropy of
the URL string. This exact module is imported by both the training
script and the Flask app, so the features used to predict always match
the features used to train.

## 9. Dataset

`ml/build_sample_dataset.py` generates a **demo** dataset of 1,000
synthetically constructed URLs (500 legitimate-style, 500
phishing-style) so the pipeline runs end to end with no downloads.
**For an actual submission, replace `data/urls_dataset.csv` with a
real published dataset** (e.g. the UCI "Phishing Websites" dataset, a
PhishTank export, or a Kaggle phishing-URL dataset) that has `url` and
`label` columns — everything downstream works unchanged.

## 10. Database Design

**`users`**

| Field         | Type      |
|---------------|-----------|
| `_id`         | ObjectId  |
| `username`    | string    |
| `email`       | string    |
| `password_hash` | string  (never plain text) |
| `created_at`  | datetime  |

**`scans`**

| Field                | Type      |
|----------------------|-----------|
| `_id`                | ObjectId  |
| `user_id`            | string    |
| `url`                | string    |
| `prediction`         | string ("Legitimate" / "Phishing") |
| `confidence`         | float     |
| `extracted_features` | object    |
| `timestamp`          | datetime  |

## 11. Application Workflow

1. User pastes a URL on the home page (or calls `/api/predict`).
2. `app.py` validates the URL format.
3. `feature_extraction.py` extracts the feature vector.
4. The saved scaler transforms it; the saved model predicts a class
   and (where supported) a probability.
5. The result is rendered on `result.html` and, if the user is logged
   in, saved to their scan history in MongoDB.
6. The dashboard aggregates the user's `scans` documents into totals
   and a recent-activity list.

## 12. Folder Structure

```
phishing-detection/
│
├── app.py                    Flask entry point / routes
├── config.py                 Env-var-based configuration
├── db.py                     MongoDB connection + data-access helpers
├── requirements.txt
├── .env.example               Copy to .env and fill in secrets
│
├── model/
│   ├── phishing_model.pkl     Trained classifier (generated)
│   ├── scaler.pkl              Feature scaler (generated)
│   ├── feature_names.pkl       Ordered feature name list (generated)
│   ├── evaluation_report.json  Metrics for every candidate model
│   ├── evaluation_report.txt
│   ├── confusion_matrix.png
│   └── model_comparison.png
│
├── ml/
│   ├── feature_extraction.py   URL → feature vector (train + serve)
│   ├── build_sample_dataset.py Generates a demo dataset
│   ├── train_model.py          Full training pipeline
│   └── evaluate_model.py       Re-evaluate a saved model
│
├── data/
│   └── urls_dataset.csv        Training data (url,label) — generated
│
├── templates/
│   ├── base.html, index.html, result.html, about.html
│   ├── login.html, register.html
│   ├── dashboard.html, history.html
│   └── error.html
│
└── static/
    ├── css/style.css
    ├── js/script.js
    └── images/
```

## 13. Installation (VS Code on macOS)

```bash
# 1. Open the project folder in VS Code
cd phishing-detection

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the environment template and edit it
cp .env.example .env
# open .env and set FLASK_SECRET_KEY to a random string

# 5. (Optional but recommended) install and start MongoDB locally
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
# Without MongoDB running, URL checks still work — only auth,
# dashboard, and history require it.
```

## 14. Training the Model

```bash
# Generate the demo dataset (skip this if you supply a real dataset at
# data/urls_dataset.csv with 'url' and 'label' columns)
python ml/build_sample_dataset.py

# Train, evaluate, and save the best model
python ml/train_model.py
```

This prints per-model accuracy/precision/recall/F1 to the terminal and
writes `model/phishing_model.pkl`, `model/scaler.pkl`,
`model/evaluation_report.json`, `model/confusion_matrix.png`, and
`model/model_comparison.png`.

## 15. Running the Flask Server

```bash
python app.py
# Server starts at http://127.0.0.1:5000
```

Visit `http://127.0.0.1:5000` in a browser. If the model hasn't been
trained yet, the home page shows a banner explaining how to train it —
the app never fabricates a prediction.

## 16. Using the API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

```json
{"url": "https://example.com", "prediction": "Legitimate", "confidence": 0.97}
```

## 17. Test URLs

Try these once the model is trained (results depend on your dataset):

- `https://www.wikipedia.org` — expected: Legitimate
- `https://github.com/login` — expected: Legitimate
- `http://paypal-secure-login.verify-account.info/webscr?cmd=login` — expected: Phishing
- `http://192.168.10.55/appleid/login.php` — expected: Phishing
- `http://bit.ly/3xJ9kLm` — flagged for shortening-service use

## 18. Common Errors and Solutions

| Symptom | Cause | Fix |
|---|---|---|
| Home page shows "ML model hasn't been trained yet" | No `.pkl` files in `model/` | Run `ml/build_sample_dataset.py` then `ml/train_model.py` |
| `FileNotFoundError: data/urls_dataset.csv` | No dataset present | Run `python ml/build_sample_dataset.py`, or add your own CSV with `url,label` columns |
| Login/register/dashboard show a database warning | MongoDB isn't running or `MONGO_URI` is wrong | Start MongoDB locally (`brew services start mongodb-community`) and check `.env` |
| `ModuleNotFoundError` on startup | Dependencies not installed / venv not activated | `source venv/bin/activate && pip install -r requirements.txt` |
| Predictions look wrong on real-world URLs | Demo dataset is synthetic, not real phishing data | Replace `data/urls_dataset.csv` with a real dataset and retrain |

## 19. Future Scope

- Incorporate page-content and certificate-based features alongside
  URL features.
- Add a browser extension for one-click checking while browsing.
- Periodically retrain against a live threat-intelligence feed.
- Add rate limiting and CAPTCHA to the public `/api/predict` endpoint.

## 20. Limitations

- Trained on a fixed dataset, so it can miss novel phishing patterns.
- URL-only analysis can't catch phishing hosted on a compromised but
  otherwise legitimate-looking domain.
- The bundled demo dataset is synthetic; swap in a real dataset before
  treating results as meaningful.

## 21. Conclusion

PhishGuard demonstrates an end-to-end applied ML system: reproducible
feature engineering shared between training and serving, model
comparison grounded in real evaluation metrics, and a usable web
interface with authentication and history — a complete, defensible
final-year major project.
