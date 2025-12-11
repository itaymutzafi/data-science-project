"""Data access module.

This module handles data fetching from Hugging Face dataset financial-news-multisource
"""

import csv
from datasets import load_dataset
from datetime import datetime, date
from huggingface_hub import login
import json
import pandas as pd
from typing import Any, Optional
import feedparser
import os


# Access Token since the dataset is protected
login(os.getenv("HF_TOKEN"))

YEARS_BACK = 5
OUTPUT_CSV = f"news_last_{YEARS_BACK}y.csv"
OUTPUT_XLSX = f"news_last_{YEARS_BACK}y.xlsx"
DATASET = "Brianferrell787/financial-news-multisource"

APPLE_KEYWORDS = ["apple", "aapl"]
MICROSOFT_KEYWORDS = ["microsoft", "MSFT"]
AMAZON_KEYWORDS = ["amazon", "AMZN"]
GOOGLE_KEYWORDS = ["google", "GOOGL"]

KEYWORDS = APPLE_KEYWORDS + MICROSOFT_KEYWORDS + AMAZON_KEYWORDS + GOOGLE_KEYWORDS

DATA_FILE_FORMAT = "data/{0}/*.parquet"
DATA_FILES = [
    DATA_FILE_FORMAT.format("yahoo_finance_articles"),
    DATA_FILE_FORMAT.format("reddit_finance_sp500"),
    DATA_FILE_FORMAT.format("fnspid_news"),
]

OUTPUT_FIELDS = [
    "date",
    "company",
    "text",
    "publication",
    "dataset_source",
    "author",
    "text_type",
    "time_precision",
    "source",
    "dataset",
    "tz_hint",
    "url",
]

def extract_date(row: dict[str, Any]) -> Optional[date]:
    """Return a date object from row['date'], or None if invalid."""
    s = row.get("date")
    try:
        date = datetime.fromisoformat(s.replace("Z", "")).date()
        return pd.to_datetime(date)
    except Exception:
        return None

def detect_company(text_lower: str) -> Optional[str]:
    if any(k in text_lower for k in APPLE_KEYWORDS):
        return "Apple"
    if any(k in text_lower for k in MICROSOFT_KEYWORDS):
        return "Microsoft"
    if any(k in text_lower for k in AMAZON_KEYWORDS):
        return "Amazon"
    if any(k in text_lower for k in GOOGLE_KEYWORDS):
        return "Google"
    return None

def parse_extra_fields(row: dict[str, Any]) -> dict:
    """
    Parse row['extra_fields'] if present.
    Handles both JSON strings and dicts; returns {} on failure.
    """
    raw = row.get("extra_fields")
    extras = {}

    if isinstance(raw, dict):
        extras = raw
    elif isinstance(raw, str):
        try:
            extras = json.loads(raw)
        except Exception:
            extras = {}
    else:
        extras = {}

    # Flatten the fields we want for output
    result = {
        "publication":    extras.get("publication", "") or "",
        "dataset_source": extras.get("dataset_source", "") or "",
        "author":         extras.get("author", "") or "",
        "text_type":      extras.get("text_type", "") or "",
        "time_precision": extras.get("time_precision", "") or "",
        "source":         extras.get("source", "") or "",
        "dataset":        extras.get("dataset", "") or "",
        "tz_hint":        extras.get("tz_hint", "") or "",
        "url":            extras.get("url", "") or "",
    }

    return result

def main():
    # Cutoff date (last N years)
    today = date.today()
    cutoff_date = pd.to_datetime(today.replace(year=today.year - YEARS_BACK))
    print("Cutoff date:", cutoff_date)

    ds = load_dataset(DATASET, split="train", data_files=DATA_FILES, streaming=True)

    min_date = None
    max_date = None
    apple_count = 0
    rows_scanned = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for row in ds:
            rows_scanned += 1

            d = extract_date(row)
            if d is None or d < cutoff_date:
                continue

            text = (row.get("text") or "").strip()
            if not text:
                continue

            company = detect_company(text.lower())
            if company is None:
                continue

            extras = parse_extra_fields(row)

            out_row = {
                "date": d.isoformat(),
                "company": company,
                "text": text,
                "publication": extras["publication"],
                "dataset_source": extras["dataset_source"],
                "author": extras["author"],
                "text_type": extras["text_type"],
                "time_precision": extras["time_precision"],
                "source": extras["source"],
                "dataset": extras["dataset"],
                "tz_hint": extras["tz_hint"],
                "url": extras["url"],
            }

            writer.writerow(out_row)

            apple_count += 1
            if min_date is None or d < min_date:
                min_date = d
            if max_date is None or d > max_date:
                max_date = d

    print("\n--- SUMMARY ---")
    print("Total rows scanned:", rows_scanned)
    print("Apple-related good rows saved:", apple_count)
    print("Earliest Apple date (>= cutoff):", min_date)
    print("Latest Apple date:", max_date)

    # news_df = pd.read_csv(OUTPUT_CSV)

    # Save also to excel
    # news_df["text"] = news_df["text"].astype(str).apply(
    #     lambda x: "".join(ch for ch in x if ord(ch) >= 32)
    # )
    # news_df.to_excel(OUTPUT_XLSX, index=False)

    # Split each company to seperate file
    # print(news_df["company"].value_counts())

    # for company, group in news_df.groupby("company"):
    #     filename = f"news_{company.lower()}.csv"
    #     group.to_csv(filename, index=False)
    #     print(f"Saved {filename} with {len(group)} rows")

if __name__ == "__main__":
    main()

# Util
def get_news_df_from_csv(csv_path):
    df = pd.read_csv(csv_path, dtype=str)
    df["date"] = pd.to_datetime(df["date"])
    return df

# From previous source Google News
def get_google_news_titles(query, days):
    # "when:Xd" limits to last X days
    q = query.replace(" ", "+") + f"+when:{days}d"
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

    feed = feedparser.parse(url)

    rows = []
    for entry in feed.entries:
        rows.append({
            "published": entry.published,
            "date": pd.to_datetime(entry.published[:16]),
            "title": entry.title,
            "link": entry.link,
            "source": entry.source.title if "source" in entry else "unknown"
        })

    return pd.DataFrame(rows)