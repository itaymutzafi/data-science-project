from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
from typing import Dict
from yfinance import Ticker 
from datetime import date
from pathlib import Path

from src.config import (
    START_DATE,
    END_DATE,
    DEF_SPLITS,
    LEGACY_PROPHET_CACHE_DIR,
    PROPHET_CACHE_DIR,
)
from src.data import fetch_sample_data
from src.features import add_return_features

def _get_prophet_cache_path(ticker: str) -> Path:
    cache_dir = Path(PROPHET_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{ticker}_prophet_predictions.parquet"

def _get_legacy_prophet_cache_path(ticker: str) -> Path:
    return Path(LEGACY_PROPHET_CACHE_DIR) / f"{ticker}_prophet_predictions.parquet"

def _load_prophet_cache(ticker: str) -> pd.DataFrame:
    primary_path = _get_prophet_cache_path(ticker)
    legacy_path = _get_legacy_prophet_cache_path(ticker)

    for cache_path in (primary_path, legacy_path):
        if not cache_path.exists():
            continue
        try:
            df = pd.read_parquet(cache_path)
            if not df.empty:
                df.index = pd.to_datetime(df.index)

            # One-time migration from legacy cache to primary layout.
            if cache_path != primary_path:
                _save_prophet_cache(ticker, df)

            return df
        except Exception as exc:
            print(f"Warning: failed to load prophet cache for {ticker}: {exc}")
    return pd.DataFrame()

def _save_prophet_cache(ticker: str, df: pd.DataFrame) -> None:
    cache_path = _get_prophet_cache_path(ticker)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)

def prophet(stocks_data: Dict[str, pd.DataFrame], n_splits: int = DEF_SPLITS, force_refresh: bool = False):
    cols = ["prophet_prediction_binary", "prophet_prediction_continuous"]
    for stock_name, stock_df in stocks_data.items():
        # cached logic ->
        if not force_refresh:
            cached = _load_prophet_cache(stock_name)
            if not cached.empty:
                stock_df = stock_df.drop(columns=[c for c in cols if c in stock_df.columns])
                stock_df = stock_df.join(cached[cols], how="left")
                stocks_data[stock_name] = stock_df
                print("successfully imported cached prophet data")
                continue
        
        # else - create new prophet columns ->

        ticker = Ticker(stock_name)
        proph_start_time = date(2015, 1, 1)

        work_df = fetch_sample_data(ticker, proph_start_time, END_DATE, save_file = False)
        add_return_features({ticker: work_df})
        work_df.index = work_df.index.tz_localize(None)
        work_df = work_df.reset_index()

        df_prophet = work_df.rename(columns={
        "Date": "ds",
        "Log_Return": "y"
        })

        df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)

        # Prepare data for Prophet
        prophet_data = df_prophet[["ds", "y"]].copy().reset_index(drop=True)

        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        all_predictions = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(prophet_data)):
            train_data = prophet_data.iloc[train_idx].copy()
            val_data = prophet_data.iloc[val_idx].copy()
            
            # Only predict on dates in target range (2020-2025)
            val_data_target = val_data[val_data['ds'] >= pd.Timestamp(START_DATE)]
            
            if len(val_data_target) > 0:
                train_data = train_data[train_data['ds'] < val_data_target['ds'].min()]

            
            if len(train_data) < 365 or len(val_data_target) == 0: 
                continue
            
            model = Prophet()
            try:
                model.fit(train_data)
                
                # Predict for validation dates in target range only
                future = pd.DataFrame({'ds': val_data_target['ds']})
                forecast = model.predict(future)
                
                # Create predictions with dates
                fold_predictions = pd.DataFrame({
                    'ds': val_data_target['ds'].values,
                    'yhat': forecast['yhat'].values
                })
                all_predictions.append(fold_predictions)
                
            except Exception as e:
                print(f"Fold {fold + 1} failed: {e}")
                continue

        # Combine all predictions
        if all_predictions:
            predictions_df = pd.concat(all_predictions, ignore_index=True)
            predictions_df['prophet_prediction_binary'] = (predictions_df['yhat'] > 0).astype(int)
            predictions_df['prophet_prediction_continuous'] = predictions_df['yhat']
            
            # Merge back to original dataframe (use extended_df, not df)
            df_target = work_df[work_df['Date'] >= pd.Timestamp(START_DATE)].copy()
            df_target = df_target.merge(
                predictions_df[['ds', 'prophet_prediction_binary', 'prophet_prediction_continuous']],
                left_on='Date',
                right_on='ds',
                how='left'
            )
            df_target = df_target.drop(columns='ds')

        else:
            df_target = work_df[work_df['Date'] >= pd.Timestamp(START_DATE)].copy()
            df_target['prophet_prediction_binary'] = np.nan  
            df_target['prophet_prediction_continuous'] = np.nan  
        
        df_target_indexed = df_target.set_index("Date")
        df_target_indexed.index = pd.to_datetime(df_target_indexed.index)

        _save_prophet_cache(
            stock_name,
            df_target_indexed[["prophet_prediction_binary", "prophet_prediction_continuous"]]
        )

        stock_df = stock_df.drop(columns=[c for c in cols if c in stock_df.columns])
        stock_df = stock_df.join(
            df_target_indexed[cols],
            how="left"
        )
        stocks_data[stock_name] = stock_df
