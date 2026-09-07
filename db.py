"""
db.py
------
MongoDB connection and simple data-access helper functions for the two
collections used by the app:

  users  -> { _id, username, email, password_hash, created_at }
  scans  -> { _id, user_id, url, prediction, confidence,
              extracted_features, timestamp }

Keeping all DB access in one module makes it easy to swap MongoDB out
later (or mock it in tests) without touching route logic.
"""

from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from config import Config

_client = None
_db = None


def get_db():
    """Lazily create and cache the MongoDB connection."""
    global _client, _db
    if _db is None:
        _client = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=5000)
        _db = _client.get_default_database()
    return _db


def db_available() -> bool:
    """Health check used by routes to fail gracefully if Mongo is down."""
    try:
        get_db().command("ping")
        return True
    except PyMongoError:
        return False


# ---------- Users ----------

def create_user(username: str, email: str, password_hash: str):
    db = get_db()
    doc = {
        "username": username,
        "email": email,
        "password_hash": password_hash,
        "created_at": datetime.now(timezone.utc),
    }
    result = db.users.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def find_user_by_username(username: str):
    return get_db().users.find_one({"username": username})


def find_user_by_id(user_id):
    from bson import ObjectId
    return get_db().users.find_one({"_id": ObjectId(user_id)})


def username_or_email_exists(username: str, email: str) -> bool:
    return get_db().users.find_one(
        {"$or": [{"username": username}, {"email": email}]}
    ) is not None


# ---------- Scan history ----------

def save_scan(user_id, url, prediction, confidence, extracted_features):
    db = get_db()
    doc = {
        "user_id": user_id,
        "url": url,
        "prediction": prediction,
        "confidence": confidence,
        "extracted_features": extracted_features,
        "timestamp": datetime.now(timezone.utc),
    }
    db.scans.insert_one(doc)
    return doc


def get_user_scans(user_id, limit: int = 50):
    db = get_db()
    cursor = db.scans.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
    return list(cursor)


def get_user_stats(user_id):
    db = get_db()
    total = db.scans.count_documents({"user_id": user_id})
    safe = db.scans.count_documents({"user_id": user_id, "prediction": "Legitimate"})
    phishing = db.scans.count_documents({"user_id": user_id, "prediction": "Phishing"})
    return {"total": total, "safe": safe, "phishing": phishing}
