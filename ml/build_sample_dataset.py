"""
build_sample_dataset.py
------------------------
Generates a small DEMO dataset (data/urls_dataset.csv) so the training
pipeline can be run immediately after cloning the project, with no
external downloads required.

>>> THIS IS NOT A SUBSTITUTE FOR A REAL DATASET. <<<
For an actual final-year submission you should replace
data/urls_dataset.csv with a real, published phishing-URL dataset, e.g.:
  - UCI "Phishing Websites" dataset
  - PhishTank verified phishing feed (https://phishtank.org)
  - Kaggle "Phishing Site URLs" dataset
Just make sure the replacement CSV has two columns: "url" and "label"
(label = 1 for phishing, 0 for legitimate), then re-run train_model.py.
Everything downstream (feature extraction, training, evaluation) works
unchanged regardless of which CSV you use.

Run:
    python ml/build_sample_dataset.py
"""

import csv
import os
import random

random.seed(42)

LEGITIMATE_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "amazon.com", "wikipedia.org",
    "twitter.com", "instagram.com", "linkedin.com", "microsoft.com", "apple.com",
    "github.com", "stackoverflow.com", "reddit.com", "netflix.com", "yahoo.com",
    "office.com", "adobe.com", "salesforce.com", "dropbox.com", "spotify.com",
    "paypal.com", "ebay.com", "wordpress.com", "cnn.com", "bbc.com",
    "nytimes.com", "who.int", "un.org", "harvard.edu", "mit.edu",
    "python.org", "mozilla.org", "gov.uk", "nasa.gov", "irs.gov",
    "chase.com", "hdfcbank.com", "icicibank.com", "sbi.co.in", "flipkart.com",
]

LEGIT_PATHS = [
    "", "/", "/about", "/products", "/contact", "/blog/2024/article",
    "/search?q=example", "/user/profile", "/help/faq", "/docs/api",
    "/news/latest", "/store/item?id=1023", "/login", "/account/settings",
]

PHISHING_PATTERNS = [
    "http://{brand}-secure-login.{tld}/verify/account",
    "http://{brand}.{tld}.security-alert.com/signin",
    "http://192.168.{a}.{b}/{brand}/login.php",
    "http://{brand}-update-billing.{tld}/confirm?user=1",
    "http://secure-{brand}.verify-account.{tld}/webscr?cmd=login",
    "http://{brand}support.{tld}-alert.net/unlock/password",
    "http://{brand}.{tld}/{rand}/login-verify-account-{rand}",
    "http://bit.ly/{rand}phish",
    "http://{brand}-{rand}.tk/secure/banking/confirm",
    "http://{rand}.{brand}.{tld}.account-suspended.info/verify",
]

BRANDS = ["paypal", "amazon", "apple", "microsoft", "bankofamerica", "netflix",
          "chase", "wellsfargo", "hdfc", "icici", "facebook", "instagram"]
TLDS = ["com", "net", "info", "xyz", "top", "co", "ru"]


def random_str(n=6):
    letters = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(letters) for _ in range(n))


def build_legit_urls(n):
    urls = []
    for _ in range(n):
        domain = random.choice(LEGITIMATE_DOMAINS)
        path = random.choice(LEGIT_PATHS)
        scheme = "https"
        sub = random.choice(["", "www."])
        urls.append(f"{scheme}://{sub}{domain}{path}")
    return urls


def build_phishing_urls(n):
    urls = []
    for _ in range(n):
        pattern = random.choice(PHISHING_PATTERNS)
        url = pattern.format(
            brand=random.choice(BRANDS),
            tld=random.choice(TLDS),
            rand=random_str(random.randint(4, 10)),
            a=random.randint(0, 255),
            b=random.randint(0, 255),
        )
        urls.append(url)
    return urls


def main():
    n_per_class = 500
    legit = build_legit_urls(n_per_class)
    phishing = build_phishing_urls(n_per_class)

    rows = [(u, 0) for u in legit] + [(u, 1) for u in phishing]
    random.shuffle(rows)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "urls_dataset.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerows(rows)

    print(f"Demo dataset written to {out_path} ({len(rows)} rows).")
    print("Replace this file with a real dataset before final submission.")


if __name__ == "__main__":
    main()
