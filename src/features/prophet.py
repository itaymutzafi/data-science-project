from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
from typing import Dict
from yfinance import Ticker 
from datetime import date
from src import data, config, features
from src.config import START_DATE, DEF_SPLITS


def prophet(stocks_data: Dict[str, pd.DataFrame], n_splits: int = DEF_SPLITS):
    for stock_name, stock_df in stocks_data.items():
        ticker = Ticker(stock_name)
        proph_start_time = date(2015, 1, 1)

        work_df = data.fetch_sample_data(ticker, proph_start_time, config.END_DATE, save_file = False)
        features.add_return_features({ticker: work_df})
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

        stock_df = stock_df.join(
            df_target_indexed[["prophet_prediction_binary", "prophet_prediction_continuous"]],
            how="left"
        )
        stocks_data[stock_name] = stock_df

