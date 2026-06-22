import os
import pandas as pd
import numpy as np
import yfinance as yf  # Ensure yfinance is accessible here
from data_fetcher import fetch_market_data
from feature_engineering import calculate_features
from fundamental_engine import fetch_and_process_fundamentals
from target_labeling import generate_target_labels

def apply_rolling_valuation_zscores(df, columns_to_transform=None):
    """Transforms absolute valuation metrics into historical rolling Z-scores."""
    if columns_to_transform is None:
        columns_to_transform = ['ratio_pe', 'ratio_pb', 'ratio_dcf_value']
    transformed_df = df.copy().sort_index()
    for col in columns_to_transform:
        if col in transformed_df.columns:
            rolling_mean = transformed_df[col].rolling(window=252, min_periods=30).mean()
            rolling_std = transformed_df[col].rolling(window=252, min_periods=30).std()
            z_score_series = (transformed_df[col] - rolling_mean) / rolling_std
            transformed_df[col] = z_score_series.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return transformed_df

def execute_full_pipeline(ticker: str, sector_ticker: str, start: str, end: str):
    """
    Orchestrates the quantitative data pipeline with integrated Sector Benchmarking.
    Fully optimized to feed the scalable v5.0 Model Architecture.
    """
    print(f"=== STARTING MASTER QUANT DATA PIPELINE FOR {ticker} ===")
    
    # 1. Fetch raw asset and baseline index markers (SPY, VIX)
    raw_data = fetch_market_data(ticker, start, end)
    if raw_data.empty:
        print("Pipeline aborted: Raw data download failed.")
        return
    
    # 2. Inject point-in-time corporate financials & DCF evaluations
    fundamental_data = fetch_and_process_fundamentals(ticker, raw_data)
    
    # 3. Inject mathematical momentum, trend, and volatility indicators
    feature_data = calculate_features(fundamental_data)
    
    # 4. Construct targets using RAW P/E values before applying Z-scores
    final_matrix = generate_target_labels(feature_data, max_horizon=60, alpha_threshold=0.05)
    # -------------------------------------------------------------------------
    # NEW: PRESERVE RAW FUNDAMENTAL ANCHORS FOR THE REGIME CLASSIFIER
    # -------------------------------------------------------------------------
    if 'ratio_pe' in final_matrix.columns:
        final_matrix['raw_ratio_pe'] = final_matrix['ratio_pe'].copy()
    # 5. Apply 1-Year Rolling Z-Scores to normalize individual asset boundaries
    print(f"🔄 Normalizing raw valuation metrics via 1-Year Rolling Z-Scores...")
    final_matrix = apply_rolling_valuation_zscores(final_matrix)
    
    # -------------------------------------------------------------------------
    # 5.5. BULLETPROOF CROSS-SECTIONAL SECTOR BENCHMARK INTEGRATION
    # -------------------------------------------------------------------------
    print(f"📥 Fetching Sector Benchmark ETF Data ({sector_ticker})...")
    try:
        # Pull raw sector dataframe
        sector_df = yf.download(sector_ticker, start=start, end=end, progress=False)
        
        # Robust multi-index column handling to avoid extraction failures
        if isinstance(sector_df.columns, pd.MultiIndex):
            if 'Close' in sector_df.columns.levels[0] and sector_ticker in sector_df.columns.levels[1]:
                sector_close = sector_df.loc[:, ('Close', sector_ticker)]
            else:
                sector_close = sector_df['Close'].iloc[:, 0]  # Fallback to first available close column
        else:
            sector_close = sector_df['Close']
            
        # Align sector closing prices to our core matrix timeline using mapping
        final_matrix['sector_close'] = final_matrix.index.map(sector_close)
        final_matrix['sector_close'] = final_matrix['sector_close'].ffill().bfill()
        
        print("📐 Calculating asset-to-sector relative momentum indicators...")
        # Extract underlying sector momentum baselines from aligned pricing data
        sector_trend_20d = final_matrix['sector_close'].pct_change(20)
        sector_trend_60d = final_matrix['sector_close'].pct_change(60)
        
        # Safely resolve asset trend column naming dependencies
        if 'Close' in final_matrix.columns:
            asset_trend_20d = final_matrix['Close'].pct_change(20)
            asset_trend_60d = final_matrix['Close'].pct_change(60)
        elif 'feature_trend_20d_velocity' in final_matrix.columns:
            asset_trend_20d = final_matrix['feature_trend_20d_velocity']
            asset_trend_60d = final_matrix['feature_trend_60d'] if 'feature_trend_60d' in final_matrix.columns else final_matrix['trend_60d']
        else:
            asset_trend_20d = final_matrix['trend_20d']
            asset_trend_60d = final_matrix['trend_60d']
        
        # Compute pure relative spread features
        final_matrix['trend_20d_sector_spread'] = asset_trend_20d - sector_trend_20d
        final_matrix['trend_60d_sector_spread'] = asset_trend_60d - sector_trend_60d
        
        # Explicitly clean edge-case mathematical abnormalities (inf, -inf, NaN)
        final_matrix['trend_20d_sector_spread'] = final_matrix['trend_20d_sector_spread'].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        final_matrix['trend_60d_sector_spread'] = final_matrix['trend_60d_sector_spread'].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        # Clean up temporary operational columns to keep dataframe small
        final_matrix = final_matrix.drop(columns=['sector_close'])
        print("✅ Sector spread engineering complete and cleaned.")
        
    except Exception as e:
        print(f"⚠️ Warning: Sector integration failed due to: {e}. Proceeding with baseline feature pool.")
    
    print("\n=== PIPELINE DATA CONSOLIDATION COMPLETE ===")
    print(f"Total clean sample matrix rows: {final_matrix.shape[0]}")
    print(f"Total features matrix columns: {final_matrix.shape[1] - 1}")
    
    # 6. Chronological Train / Test Slicing (Preserves timeline integrity)
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
    os.makedirs("data", exist_ok=True)
    
    # --- PRODUCTION ASSIGNMENT MAP REFERENCE ---
    # Alphabet (GOOG) and Meta (META) -> Communication Services (XLC)
    # Amazon (AMZN) -> Consumer Discretionary (XLY)
    # Apple (AAPL), Microsoft (MSFT) -> Technology (XLK)
    # Micron (MU), Broadcom (AVGO), Nvidia (NVDA) -> Semiconductors (SOXX)
    # Bank of America (BAC) -> Financials (XLF)
    # Let's execute using the optimized 2026 data boundary line
    execute_full_pipeline("NVDA", "SOXX", "2022-01-01", "2026-06-22")