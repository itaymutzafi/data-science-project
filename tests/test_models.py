"""Tests for baseline models."""

import pandas as pd
import numpy as np
from src.models.baselines import NaiveBaseline, RandomBaseline

def test_naive_baseline(sample_data):
    """Test Naive Baseline (Zero Strategy)."""
    model = NaiveBaseline(strategy="zero")
    preds = model.predict(sample_data)
    
    assert len(preds) == len(sample_data)
    assert (preds == 0).all()

def test_random_baseline(sample_data):
    """Test Random Baseline statistical properties."""
    model = RandomBaseline(seed=42)
    y = sample_data["Log_Returns"]
    model.fit(y)
    preds = model.predict(sample_data)
    
    assert len(preds) == len(sample_data)
    assert preds.std() > 0  # Should have variance
    assert isinstance(preds, pd.Series)
