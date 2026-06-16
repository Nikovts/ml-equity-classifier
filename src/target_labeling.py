import pandas as pd
import numpy as np
import sys



def generate_target_labels(df, alpha_threshold=0.05, max_horizon=60):
    """
    Strategy A: The Pre-Earnings Run Engine.
    Dynamically caps the prediction horizon to stop exactly 2 trading days 
    before the next scheduled earnings announcement.
    """
    print("Executing Strategy A: Dynamic Pre-Earnings Target Labeling...")
    df = df.sort_index()
    
    # 1. Reverse-engineer stationary EPS to locate real corporate reporting dates
    if 'ratio_pe' in df.columns and 'asset_close' in df.columns:
        # Avoid division by zero bugs
        safe_pe = df['ratio_pe'].replace(0, np.nan)
        calculated_eps = (df['asset_close'] / safe_pe).ffill().bfill()
        
        # Round to clear out minor floating-point computational noise
        eps_rounded = calculated_eps.round(3)
        
        # Identify the true structural jumps (the actual earnings release days)
        fundamental_changes = eps_rounded != eps_rounded.shift(1)
        fundamental_changes.iloc[0] = False
        earnings_dates = df.index[fundamental_changes].tolist()
        print(f"Cleanly extracted {len(earnings_dates)} historical earnings dates from stationary EPS states.")
    else:
        print("Warning: Fundamental data missing. Falling back to fixed horizons.")
        earnings_dates = []

    targets = []
    dates_list = df.index.tolist()
    df_prices = df['asset_close']
    spy_prices = df['spy_close']
    
    # 2. Dynamic Horizon Loop
    for i, current_date in enumerate(dates_list):
        future_earnings = [d for d in earnings_dates if d > current_date]
        
        if not future_earnings:
            remaining_rows = len(dates_list) - 1 - i
            horizon = min(max_horizon, remaining_rows)
        else:
            next_earnings_date = future_earnings[0]
            try:
                earnings_idx = dates_list.index(next_earnings_date)
                trading_days_to_earnings = earnings_idx - i
                horizon = trading_days_to_earnings - 2
            except ValueError:
                horizon = max_horizon
        
        horizon = min(max_horizon, horizon)
        
        # If we are right on top of an earnings event, classify as Hold (0) due to low clean runway
        if horizon < 3 or (i + horizon) >= len(dates_list):
            targets.append(0)
            continue
            
        # 3. Calculate outperformance alpha over the clean pre-earnings runway
        future_asset_ret = (df_prices.iloc[i + horizon] - df_prices.iloc[i]) / df_prices.iloc[i]
        future_spy_ret = (spy_prices.iloc[i + horizon] - spy_prices.iloc[i]) / spy_prices.iloc[i]
        alpha = future_asset_ret - future_spy_ret
        
        # 4. Scale target thresholds proportionally to the horizon runway length
        scaled_threshold = alpha_threshold * (horizon / max_horizon)
        
        if alpha >= scaled_threshold:
            targets.append(1)   # Outperforming Buy Runway
        elif alpha <= -scaled_threshold:
            targets.append(-1)  # Underperforming Sell Runway
        else:
            targets.append(0)   # Neutral / Benchmark Hold
            
    df['target'] = targets
    
    print(f"Target distribution under Strategy A:")
    print(df['target'].value_counts())
    
    return df


if __name__ == "__main__":
    # Local verification routine
    from data_fetcher import fetch_market_data
    from feature_engineering import calculate_features
    
    raw = fetch_market_data("AAPL", "2016-01-01", "2026-01-01")
    if not raw.empty:
        features = calculate_features(raw)
        final_matrix = generate_target_labels(features, max_horizon=60, alpha_threshold=0.05)
        print(f"\nFinal training matrix dimensions: {final_matrix.shape}")