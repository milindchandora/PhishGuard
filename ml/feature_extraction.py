"""
feature_extraction.py
----------------------
Extracts numeric features from a raw URL string for use by the
phishing-detection ML model.

IMPORTANT: This exact module is imported by BOTH train_model.py (during
training) and app.py (during live prediction) so that the feature set
used to train the model is always identical to the feature set used to
score a new URL. Never duplicate this logic elsewhere.
"""

import re
import math
from urllib.parse import urlparse

# Feature names, in the exact order the model expects. Keep this list
# and extract_features() in sync at all times.
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_question_marks",
    "num_equal_signs",
    "num_at_symbols",
    "num_ampersands",
    "num_digits",
    "num_special_chars",
    "num_subdomains",
    "has_ip_address",
    "has_https",
    "has_http_in_path",
    "has_at_symbol",
    "has_double_slash_redirect",
    "has_suspicious_words",
    "has_shortening_service",
    "digit_letter_ratio",
    "num_params",
    "domain_has_hyphen",
    "domain_length",
    "url_entropy",
]

# A small, well-known list of URL-shortening services.
SHORTENING_SERVICES = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "shorte.st", "cutt.ly", "rb.gy", "tiny.cc",
}

# Words that frequently appear in phishing URLs, trying to imitate
# legitimate brands or create urgency.
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "account", "update", "secure", "banking",
    "confirm", "password", "webscr", "ebayisapi", "paypal", "suspend",
    "invoice", "wallet", "alert", "unlock", "urgent", "click", "free",
]

IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)


def _shannon_entropy(s: str) -> float:
    """Return the Shannon entropy of a string (bits per character)."""
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return round(entropy, 4)


def _normalize_url(url: str) -> str:
    """Ensure the URL has a scheme so urlparse behaves consistently."""
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url):
        url = "http://" + url
    return url


def extract_features(raw_url: str) -> dict:
    """
    Extract a dictionary of numeric features from a raw URL string.
    Returns a dict keyed by FEATURE_NAMES (always in the same order,
    always the same keys), so it can be safely converted to a
    DataFrame row for the model.
    """
    url = _normalize_url(raw_url)
    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    num_digits = sum(c.isdigit() for c in url)
    num_letters = sum(c.isalpha() for c in url)
    special_chars = re.findall(r"[^a-zA-Z0-9]", url)

    subdomains = hostname.split(".")
    # e.g. "www.example.co.uk" -> subdomains beyond the base domain
    num_subdomains = max(len(subdomains) - 2, 0)

    domain_only = hostname
    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equal_signs": url.count("="),
        "num_at_symbols": url.count("@"),
        "num_ampersands": url.count("&"),
        "num_digits": num_digits,
        "num_special_chars": len(special_chars),
        "num_subdomains": num_subdomains,
        "has_ip_address": 1 if IP_PATTERN.match(hostname) else 0,
        "has_https": 1 if parsed.scheme == "https" else 0,
        "has_http_in_path": 1 if "http" in path.lower() else 0,
        "has_at_symbol": 1 if "@" in url else 0,
        "has_double_slash_redirect": 1 if url.rfind("//") > 7 else 0,
        "has_suspicious_words": 1 if any(w in url.lower() for w in SUSPICIOUS_WORDS) else 0,
        "has_shortening_service": 1 if domain_only.lower() in SHORTENING_SERVICES else 0,
        "digit_letter_ratio": round(num_digits / num_letters, 4) if num_letters else 0.0,
        "num_params": len(query.split("&")) if query else 0,
        "domain_has_hyphen": 1 if "-" in domain_only else 0,
        "domain_length": len(domain_only),
        "url_entropy": _shannon_entropy(url),
    }
    return features


def extract_features_list(raw_url: str):
    """Return the features as an ordered list (matches FEATURE_NAMES)."""
    feats = extract_features(raw_url)
    return [feats[name] for name in FEATURE_NAMES]
