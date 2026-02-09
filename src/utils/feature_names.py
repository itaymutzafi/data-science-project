"""Feature naming schema utilities.

Provides a single canonical naming policy and safe aliasing for legacy names.
"""

from __future__ import annotations

import re
from typing import Dict

import pandas as pd


LEGACY_EXACT_ALIASES = {
    "NVIDIA_Segment_Leader": "NVDA_Leader",
    "Days To Nearest Report": "Days_To_Nearest_Report",
}

_PEER_PATTERN = re.compile(
    r"^(?P<ticker>[A-Z]+)\s*-\s*(?P<feature>Close|Volume|Log_Return)$"
)


def canonicalize_feature_name(name: str) -> str:
    """Return canonical feature name for a potentially legacy input name."""
    if name in LEGACY_EXACT_ALIASES:
        return LEGACY_EXACT_ALIASES[name]

    peer_match = _PEER_PATTERN.match(name)
    if peer_match is not None:
        ticker = peer_match.group("ticker")
        feature = peer_match.group("feature")
        return f"Peer_{ticker}_{feature}"

    return name


def canonicalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe column names to canonical feature naming."""
    rename_map: Dict[str, str] = {}
    duplicates_to_drop: list[str] = []

    for col in df.columns:
        canonical = canonicalize_feature_name(col)
        if canonical == col:
            continue

        if canonical in df.columns:
            duplicates_to_drop.append(col)
            continue

        rename_map[col] = canonical

    normalized = df.rename(columns=rename_map)
    if duplicates_to_drop:
        normalized = normalized.drop(columns=duplicates_to_drop)
    return normalized

