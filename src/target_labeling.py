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
        # 🛡️ DYNAMIC FIX: Calculate expected quarterly intervals for the dataset length
        # 1105 rows / 63 trading days per quarter = ~17 expected reporting events
        expected_quarters = max(1, len(df) // 63)
        
        # If we find significantly fewer dates than expected (e.g., catching annual updates)
        if len(earnings_dates) < (expected_quarters - 3): 
            print(f"⚠️ Insufficient quarterly steps detected ({len(earnings_dates)} found, expected ~{expected_quarters}).")
            print("   Likely missing/annual data fields. Deploying synthetic 63-day fiscal cycle barriers...")
            earnings_dates = [df.index[x] for x in range(0, len(df), 63)]
        
        print(f"Cleanly extracted {len(earnings_dates)} historical earnings dates from stationary EPS states.")
    else:
        print("Warning: Fundamental data missing. Falling back to fixed horizons.")
        earnings_dates = []

    
    dates_list = df.index.tolist()
    df_prices = df['asset_close']
    spy_prices = df['spy_close']
    
    # 2. Create lists to hold our new structural inputs
    targets = []
    days_to_earnings_feature = []

    for i, current_date in enumerate(dates_list):
        future_earnings = [d for d in earnings_dates if d > current_date]
    
        if not future_earnings:
            remaining_rows = len(dates_list) - 1 - i
            horizon = min(max_horizon, remaining_rows)
            trading_days_to_earnings = max_horizon
        else:
            next_earnings_date = future_earnings[0]
            try:
                earnings_idx = dates_list.index(next_earnings_date)
                trading_days_to_earnings = earnings_idx - i
                horizon = trading_days_to_earnings - 2 # Keep 2 days safety buffer
            except ValueError:
                horizon = max_horizon
                trading_days_to_earnings = max_horizon
            
        horizon = min(max_horizon, horizon)
    
        # Track days to earnings as a raw numeric feature for the model
        days_to_earnings_feature.append(trading_days_to_earnings)
    
        # 1. Check for basic runway clearance
        if horizon < 3 or (i + horizon) >= len(dates_list):
            targets.append(0)
            continue
        
        # 2. Extract the entire window slice of future prices
        asset_window = df_prices.iloc[i + 1 : i + horizon + 1]
        spy_window = spy_prices.iloc[i + 1 : i + horizon + 1]
    
        # 3. IMPLEMENTED: Calculate returns using the average window value vs initial value
        # This represents: (Mean_Future_Price - Initial_Price) / Initial_Price
        asset_average_return = (asset_window.mean() - df_prices.iloc[i]) / df_prices.iloc[i]
        spy_average_return = (spy_window.mean() - spy_prices.iloc[i]) / spy_prices.iloc[i]
    
        # Calculate the net average alpha for this runway period
        mean_alpha = asset_average_return - spy_average_return
    
        # 4. Implement your strict Alpha Threshold Floor to eliminate micro-noise
        MIN_ALPHA_FLOOR = 0.025  # Requires at least 2.5% average outperformance
        scaled_threshold = alpha_threshold * (horizon / max_horizon)
        operational_threshold = max(MIN_ALPHA_FLOOR, scaled_threshold)
    
        # 5. Classify based on the average sustained alpha opportunity
        if mean_alpha >= operational_threshold:
            targets.append(1)   # Sustained Outperforming Buy Runway
        elif mean_alpha <= -operational_threshold:
            targets.append(-1)  # Sustained Underperforming Sell Runway
        else:
            targets.append(0)   # Noise / Benchmark Hold

    # Append the new data directly back to your main dataframe
    df['feature_days_to_earnings'] = days_to_earnings_feature
    df['target'] = targets
    # # 2. Dynamic Horizon Loop
    # for i, current_date in enumerate(dates_list):
    #     future_earnings = [d for d in earnings_dates if d > current_date]
        
    #     if not future_earnings:
    #         remaining_rows = len(dates_list) - 1 - i
    #         horizon = min(max_horizon, remaining_rows)
    #     else:
    #         next_earnings_date = future_earnings[0]
    #         try:
    #             earnings_idx = dates_list.index(next_earnings_date)
    #             trading_days_to_earnings = earnings_idx - i
    #             horizon = trading_days_to_earnings - 2
    #         except ValueError:
    #             horizon = max_horizon
        
    #     horizon = min(max_horizon, horizon)
        
    #     # If we are right on top of an earnings event, classify as Hold (0) due to low clean runway
    #     if horizon < 3 or (i + horizon) >= len(dates_list):
    #         targets.append(0)
    #         continue
            
    #     # 3. Calculate outperformance alpha over the clean pre-earnings runway
    #     future_asset_ret = (df_prices.iloc[i + horizon] - df_prices.iloc[i]) / df_prices.iloc[i]
    #     future_spy_ret = (spy_prices.iloc[i + horizon] - spy_prices.iloc[i]) / spy_prices.iloc[i]
    #     alpha = future_asset_ret - future_spy_ret
        
    #     # 4. Scale target thresholds proportionally to the horizon runway length
    #     scaled_threshold = alpha_threshold * (horizon / max_horizon)
        
    #     if alpha >= scaled_threshold:
    #         targets.append(1)   # Outperforming Buy Runway
    #     elif alpha <= -scaled_threshold:
    #         targets.append(-1)  # Underperforming Sell Runway
    #     else:
    #         targets.append(0)   # Neutral / Benchmark Hold
            
    # df['target'] = targets
    
    print(f"Target distribution :")
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