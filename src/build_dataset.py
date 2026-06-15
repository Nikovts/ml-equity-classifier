import pandas as pd
from data_fetcher import fetch_market_data
from feature_engineering import calculate_features
from fundamental_engine import fetch_and_process_fundamentals
from target_labeling import generate_target_labels

def execute_full_pipeline(ticker: str, start: str, end: str):
    """
    Orchestrates the entire end-to-end quantitative data pipeline.
    Merges market data, technical indicators, fundamental models, and labels,
    then executes a strict look-ahead protected chronological split.
    """
    print("=== STARTING MASTER QUANT DATA PIPELINE ===")
    
    # 1. Fetch raw underlying assets and index markers
    raw_data = fetch_market_data(ticker, start, end)
    if raw_data.empty:
        print("Pipeline aborted: Raw data download failed.")
        return
    
    # 2. Inject point-in-time corporate financials & DCF evaluations
    fundamental_data = fetch_and_process_fundamentals(ticker, raw_data)
    
    # 3. Inject mathematical momentum, trend, and volatility indicators
    feature_data = calculate_features(fundamental_data)
    
    # 4. Construct look-ahead protected forward return target labels
    final_matrix = generate_target_labels(feature_data, max_horizon=60, alpha_threshold=0.05)
    
    print("\n=== PIPELINE DATA CONSOLIDATION COMPLETE ===")
    print(f"Total clean sample matrix rows: {final_matrix.shape[0]}")
    print(f"Total features matrix columns: {final_matrix.shape[1] - 1}") # Exclude target column
    
    # ----------------------------------------------------
    # 5. Chronological Train / Test Slicing
    # ----------------------------------------------------
    split_index = int(len(final_matrix) * 0.80)
    
    train_set = final_matrix.iloc[:split_index]
    test_set = final_matrix.iloc[split_index:]
    
    print(f"\nSuccessfully segmented matrices chronologically:")
    print(f"├── Training Set Span: {train_set.index.min().strftime('%Y-%m-%d')} to {train_set.index.max().strftime('%Y-%m-%d')} ({len(train_set)} rows)")
    print(f"└── Testing Set Span:  {test_set.index.min().strftime('%Y-%m-%d')} to {test_set.index.max().strftime('%Y-%m-%d')} ({len(test_set)} rows)")
    
    # Export matrices locally for the modeling phase
    train_set.to_csv("data/train_matrix.csv")
    test_set.to_csv("data/test_matrix.csv")
    print("\nMatrices exported successfully to data/ folder. Ready for model training.")

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    execute_full_pipeline("NVDA", "2022-01-01", "2026-06-01")