
import pandas as pd
import numpy as np
from src.evaluation import get_top_results, REG_METRICS

def test_get_top_results():
    print("Metrics:", REG_METRICS)
    
    # Create dummy data
    models = ["NaiveBaseline", "LinearRegression", "RandomForest"]
    tickers = ["AAPL", "MSFT"]
    feature_sets = [1, 2, 3]
    folds = range(5)
    
    data = []
    for model in models:
        for ticker in tickers:
            for fset in feature_sets:
                for fold in folds:
                    row = {
                        "Model": model,
                        "Ticker": ticker,
                        "FeatureSet": fset,
                        "Fold": fold,
                        "RMSE": np.random.random(),
                        "R2": np.random.random(),
                        "Directional Accuracy": np.random.random()
                    }
                    data.append(row)
                    
    df = pd.DataFrame(data)
    print(f"DataFrame shape: {df.shape}")
    print("Calling get_top_results...")
    
    try:
        get_top_results(REG_METRICS, df)
        print("Success!")
    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_get_top_results()
