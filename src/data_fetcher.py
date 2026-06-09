import os
import pandas as pd
import yfinance as yf

def fetch_market_data(ticker_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical daily price data for a given asset, 
    the S&P 500 benchmark, and the VIX index.
    Explicitly overrides auto_adjust to capture 'Adj Close'.
    
    Parameters:
        ticker_symbol (str): The stock ticker (e.g., 'AAPL').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: A combined DataFrame with aligned dates.
    """
    print(f"Initializing data download for {ticker_symbol} from {start_date} to {end_date}...")
    
    tickers_to_download = {
        'asset': ticker_symbol,
        'spy': '^GSPC',  
        'vix': '^VIX'    
    }
    
    combined_df = pd.DataFrame()
    
    for label, ticker in tickers_to_download.items():
        try:
            print(f"Fetching data for {ticker} ({label})...")
            # Explicitly set auto_adjust=False to maintain 'Adj Close' availability
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            
            if data.empty:
                raise ValueError(f"No data returned for ticker {ticker}")
            
            # Handle MultiIndex columns natively produced by modern yfinance versions
            if isinstance(data.columns, pd.MultiIndex):
                # Columns are formatted as (Metric, Ticker) e.g., ('Adj Close', 'AAPL')
                adj_close = data[('Adj Close', ticker)]
                volume = data[('Volume', ticker)] if ('Volume', ticker) in data.columns else None
            else:
                adj_close = data['Adj Close']
                volume = data['Volume'] if 'Volume' in data.columns else None
            
            # Form single-level aligned series
            combined_df[f"{label}_close"] = adj_close
            if label == 'asset' and volume is not None:
                combined_df[f"{label}_volume"] = volume
                
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
            
    # Clean and align structural timestamps
    combined_df = combined_df.dropna()
    print(f"Data fetch complete. Total rows captured: {len(combined_df)}")
    return combined_df

if __name__ == "__main__":
    df = fetch_market_data(ticker_symbol="AAPL", start_date="2016-01-01", end_date="2026-01-01")
    if not df.empty:
        print("\nFirst 5 rows of aligned dataset:")
        print(df.head())
