"""Data access module.

This module handles data fetching from Hugging Face dataset financial-news-multisource
"""

from datasets import load_dataset
from datetime import datetime, date
from huggingface_hub import login
import json
import pandas as pd
from typing import Any, Optional, Union
from pathlib import Path
import feedparser
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import START_DATE, END_DATE, TICKER_TO_COMPANY_MAP, RAW_NEWS_PATH


# Access Token since the dataset is protected, insert here or in .env
# login(os.getenv("HF_TOKEN"))

OUTPUT_XLSX = "data/raw/news_last_5y.xlsx"
DATASET = "Brianferrell787/financial-news-multisource"

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
    for ticker, company in TICKER_TO_COMPANY_MAP.items():
        if any(k in text_lower for k in [ticker.lower(), company.lower()]):
            return company
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
    ds = load_dataset(DATASET, split="train", data_files=DATA_FILES, streaming=True)

    rows_scanned = 0
    rows = []

    for row in ds:
        rows_scanned += 1

        d = extract_date(row)
        if d is None or d < pd.Timestamp(START_DATE) or d > pd.Timestamp(END_DATE):
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
        rows.append(out_row)

    news_df = pd.DataFrame(rows)
    news_df.to_parquet(RAW_NEWS_PATH, index=False)

    print("\n--- SUMMARY ---")
    print("Total rows scanned:", rows_scanned)
    print("Good rows saved:", len(rows))

    # FOR DEBUG:
    # Save also to excel with valid charachters
    # news_df["text"] = news_df["text"].astype(str).apply(
    #     lambda x: "".join(ch for ch in x if ord(ch) >= 32)
    # )
    # news_df.to_excel(OUTPUT_XLSX, index=False)

    # Split each company to seperate file
    # print(news_df["company"].value_counts())

    # for company, group in news_df.groupby("company"):
    #     filename = f"{FOLDER}news_{company.lower()}.parquet"
    #     group.to_parquet(filename, index=False)
    #     print(f"Saved {filename} with {len(group)} rows")

if __name__ == "__main__":
    main()

# Util
def get_news_df_from_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load news data from CSV or Parquet, accepting either str or Path input.
    """
    path_obj = Path(file_path)

    if path_obj.suffix == ".csv":
        df = pd.read_csv(path_obj, dtype=str)
    else:
        df = pd.read_parquet(path_obj)
    df["date"] = pd.to_datetime(df["date"])
    return df

# From previous source Google News
def get_google_news_titles(query: str, days: int) -> pd.DataFrame:
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

    if not rows:
        df = pd.DataFrame(columns=["published", "date", "title", "link", "source"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    return pd.DataFrame(rows)
