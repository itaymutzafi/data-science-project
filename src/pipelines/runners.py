
import logging
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Dict, List, Optional
from src import config
from src.features import sentiment_analysis, sets
from src.models import experiment

logger = logging.getLogger(__name__)

def load_and_prepare_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Loads stock data and merges it with sentiment features."""
    
    # 1. Load or Generate Sentiment Data
    logger.info("Loading Sentiment Data...")
    if Path(config.SENTIMENT_CACHE).exists():
        sent_df = pd.read_csv(config.SENTIMENT_CACHE)
        sent_df['date'] = pd.to_datetime(sent_df['date'])
    else:
        logger.info("Generating Sentiment Data from raw news...")
        sent_df = sentiment_analysis.generate_daily_sentiment_features(
            config.RAW_NEWS_PATH,
            output_path=config.SENTIMENT_CACHE,
            n_sample_per_day=config.SAMPLES_PER_DAY
        )

    # 2. Download Stock Data
    logger.info(f"Downloading Stock Data for {tickers}...")
    data_dict = {}
    
    try:
        raw_stocks = yf.download(
            tickers, 
            start="2020-01-01", 
            end="2025-01-01", 
            group_by='ticker', 
            auto_adjust=True,
            progress=False
        )
        
        if raw_stocks.empty:
            logger.error("Downloaded stock data is empty!")
            return {}
            
    except Exception as e:
        logger.error(f"Failed to download stock data: {e}")
        return {}

    # 3. Process and Merge
    logger.info("Merging Stock and Sentiment Data...")
    for ticker in tickers:
        try:
            if isinstance(raw_stocks.columns, pd.MultiIndex):
                df = raw_stocks[ticker].copy()
            else:
                df = raw_stocks.copy() if len(tickers) == 1 else pd.DataFrame()

            if df.empty:
                continue

            df.index = pd.to_datetime(df.index).normalize()
            df = df.sort_index()

            company_name = config.TICKER_TO_COMPANY_MAP.get(ticker)
            if company_name:
                df_merged = sentiment_analysis.integrate_sentiment_data(
                    df,
                    sent_df,
                    company_name=company_name
                )
                data_dict[ticker] = df_merged
            else:
                data_dict[ticker] = df

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")

    return data_dict

def run_full_experiment(
    tickers: List[str] = None, 
    n_subspaces: int = 20, 
    output_file: str = "experiment_results_final.csv"
):
    """
    Runs the full experiment pipeline: load data, generate feature sets, run models, save results.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]

    # 1. Load Data
    data_dict = load_and_prepare_data(tickers)
    if not data_dict:
        logger.critical("No data available.")
        return

    # 2. Define Feature Subspaces
    logger.info("Generating Random Feature Subspaces...")
    random_sets = sets.generate_random_subspaces(n=n_subspaces, min_k=3, max_k=10)
    fset_names = list(random_sets.keys())

    # 3. Run Regression
    logger.info("--- Starting Regression Experiment ---")
    reg_models = [
        "NaiveBaseline", "Ridge", "RandomForest", "RandomForest_Deep",
        "XGB_Conservative", "XGB_Aggressive", "SVR", "LSTM"
    ]
    reg_config = experiment.ExperimentConfig(
        tickers=list(data_dict.keys()),
        feature_sets=fset_names,
        feature_set_params=random_sets,
        models=reg_models,
        target_type="continuous",
        target_horizon=1,
        start_date="2020-01-01",
        end_date="2025-01-01",
        n_splits=3
    )
    reg_runner = experiment.ExperimentRunner(data_dict, reg_config)
    reg_runner.run()
    reg_results = reg_runner.get_results_df()

    # 4. Run Classification
    logger.info("--- Starting Classification Experiment ---")
    cls_models = ["RandomForestClassifier", "LogisticRegression", "XGBClassifier"]
    cls_config = experiment.ExperimentConfig(
        tickers=list(data_dict.keys()),
        feature_sets=fset_names,
        feature_set_params=random_sets,
        models=cls_models,
        target_type="binary",
        target_horizon=1,
        start_date="2020-01-01",
        end_date="2025-01-01",
        n_splits=3
    )
    cls_runner = experiment.ExperimentRunner(data_dict, cls_config)
    cls_runner.run()
    cls_results = cls_runner.get_results_df()

    # 5. Combine and Save
    all_results = pd.concat([reg_results, cls_results], ignore_index=True)
    
    output_path = config.PROCESSED_DATA_DIR / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(output_path, index=False)
    logger.info(f"All results saved to {output_path}")

    return all_results
