import numpy as np
import pandas as pd
import sys

def generate_target_labels(feat_df: pd.DataFrame, horizon: int = 60, threshold: float = 0.05) -> pd.DataFrame:
    """
    Generates forward-looking target classification labels (1, 0, -1) 
    based on relative outperformance over a forward window.
    Strictly truncates the tail of the dataset to eliminate look-ahead bias.
    
    Parameters:
        feat_df (pd.DataFrame): Dataframe containing engineered features and close prices.
        horizon (int): Forward trading days window (e.g., 60 days = ~3 months).
        threshold (float): Percentage outperformance hurdle rate (e.g., 0.05 = 5%).
    """
    df_target = feat_df.copy().sort_index()
    
    print(f"Generating target labels for a forward horizon of {horizon} trading days...")
    
    # 1. Calculate future log returns for the asset and the benchmark index
    # We look FORWARD by using shift(-horizon)
    df_target['future_asset_ret'] = np.log(df_target['asset_close'].shift(-horizon) / df_target['asset_close'])
    df_target['future_spy_ret'] = np.log(df_target['spy_close'].shift(-horizon) / df_target['spy_close'])
    
    # 2. Calculate the future outperformance margin
    df_target['forward_outperformance'] = df_target['future_asset_ret'] - df_target['future_spy_ret']
    
    # 3. Apply mathematical threshold to generate 3 distinct classes
    conditions = [
        (df_target['forward_outperformance'] > threshold),          # Outperform -> Buy
        (df_target['forward_outperformance'] < -threshold)         # Underperform -> Sell
    ]
    choices = [1, -1]
    
    # Default choice is 0 (Match / Hold)
    df_target['target'] = np.select(conditions, choices, default=0)
    
    # 4. CRITICAL: Prevent Look-Ahead Bias
    # The last 'horizon' rows contain future metrics that are unseeable in real-time.
    # We drop them entirely to preserve causal integrity.
    clean_df = df_target.dropna(subset=['forward_outperformance']).copy()
    
    # Drop auxiliary forward columns so the ML model can't accidentally peek at them
    clean_df = clean_df.drop(columns=['future_asset_ret', 'future_spy_ret', 'forward_outperformance'])
    
    print(f"Target labeling complete.")
    print(f"Dropped the last {horizon} rows to mitigate look-ahead bias.")
    print("Class Distribution counts:")
    print(clean_df['target'].value_counts())
    
    return clean_df

if __name__ == "__main__":
    # Local verification routine
    from data_fetcher import fetch_market_data
    from feature_engineering import calculate_features
    
    raw = fetch_market_data("AAPL", "2016-01-01", "2026-01-01")
    if not raw.empty:
        features = calculate_features(raw)
        final_matrix = generate_target_labels(features, horizon=60, threshold=0.05)
        print(f"\nFinal training matrix dimensions: {final_matrix.shape}")