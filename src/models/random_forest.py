from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.config import TEST_SIZE
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def random_forest_classifier(df_s: Dict, base_features: List[str], with_peers: bool = True) -> None:
    results = []

    for df in df_s.values():
        df["Target"] = (df["Log_Return"] > 0).astype(int)
        
    for company, df in df_s.items():
        # data processing
        split_idx = int((1 - TEST_SIZE) * len(df))
        peer_features = []
        if with_peers:
            peer_features = [col for col in df.columns if "-" in col]
        features = base_features + peer_features

        # model initilazion
        model = RandomForestClassifier(n_estimators=1000, min_samples_split= 2, random_state=1)
        train_data = df.iloc[:split_idx, :]
        test_data = df.iloc[split_idx:, :]
        model.fit(train_data[features], train_data["Target"])
        preds = model.predict(test_data[features])
        preds = pd.Series(preds, index = test_data.index)

        # model metrics
        accuracy = accuracy_score(test_data["Target"], preds)
        precision = precision_score(test_data["Target"], preds, zero_division=0)
        recall = recall_score(test_data["Target"], preds, zero_division=0)
        f1 = f1_score(test_data["Target"], preds, zero_division=0)
        
        results.append({
            'Company': company,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })

    # plotting 
    results_df = pd.DataFrame(results).set_index("Company")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(results_df.T, annot=True, fmt='.4f', cmap='RdYlGn', 
                cbar=True, linewidths=0.5, ax=ax, 
                xticklabels=results_df.index, yticklabels=results_df.columns, 
                vmin=0.0, vmax=1.0)
    ax.set_title(f"Random Forest Classification Model using features {base_features}", 
                fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()



