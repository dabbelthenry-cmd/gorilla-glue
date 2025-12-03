"""
Generate historical risk scores for the past 5 years using the actual scoring system.
This script fetches historical market data and calculates risk scores using the same
logic as the main app, then stores them in the database.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from database import init_db, add_risk_score

def calculate_indicators(df):
    """Calculate technical indicators for the dataframe"""
    # SMA
    df['SMA'] = df['Close'].rolling(window=20).mean()
    # EMA
    df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def calculate_risk_score(row, prev_row, position_type="Long"):
    """Calculate risk score using the same logic as the main app"""
    bullish_points = 0
    max_points = 0
    
    # Check if we have valid data
    if pd.isna(row['SMA']) or pd.isna(row['EMA']) or pd.isna(row['MACD']) or pd.isna(row['RSI']):
        return None, "N/A"
    
    # 1. Trend (30 pts) - EMA vs SMA
    max_points += 30
    if row['EMA'] > row['SMA']:
        bullish_points += 30
    
    # 2. Momentum (30 pts) - MACD
    max_points += 30
    if (row['MACD'] > row['Signal_Line']) and (row['MACD'] > 0):
        bullish_points += 30
    
    # 3. Volatility (25 pts) - Bollinger Bands
    max_points += 25
    bb_middle_rising = row['BB_Middle'] > prev_row['BB_Middle']
    if (row['Close'] > row['BB_Upper']) or bb_middle_rising:
        bullish_points += 25
    
    # 4. RSI (15 pts)
    max_points += 15
    rsi_rising = row['RSI'] > prev_row['RSI']
    if (row['RSI'] > 50) and rsi_rising and (row['RSI'] < 70):
        bullish_points += 15
    
    # Calculate scores
    bullish_score = int((bullish_points / max_points) * 100) if max_points > 0 else 0
    bearish_score = 100 - bullish_score
    
    # Determine risk based on position
    if position_type == "Long":
        risk_score = bearish_score
    else:
        risk_score = bullish_score
    
    # Determine action
    if risk_score >= 80:
        action = "Strong Hedge (75-100%)"
    elif risk_score >= 60:
        action = "Moderate Hedge (50-75%)"
    elif risk_score >= 40:
        action = "Partial Hedge (25-50%)"
    else:
        action = "No Hedge / Unwind"
    
    return risk_score, action

def generate_historical_scores(currency_pair="EUR/USD", position_type="Long", years=5):
    """Generate historical risk scores for the specified currency pair"""
    
    print(f"Generating historical scores for {currency_pair} over the past {years} years...")
    
    # Map currency pair to Yahoo Finance ticker
    ticker_map = {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "USD/CHF": "CHF=X",
        "AUD/USD": "AUDUSD=X"
    }
    ticker = ticker_map.get(currency_pair, "EURUSD=X")
    
    # Fetch historical data
    print(f"Fetching {years} years of historical data for {ticker}...")
    df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False)
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if df.empty:
        print(f"No data found for {currency_pair}")
        return
    
    print(f"Downloaded {len(df)} days of data")
    
    # Calculate indicators
    print("Calculating technical indicators...")
    df = calculate_indicators(df)
    
    # Drop rows with NaN values (from indicator calculations)
    df = df.dropna()
    
    print(f"Processing {len(df)} data points after indicator calculation...")
    
    # Sample monthly data points (first trading day of each month)
    df_monthly = df.resample('MS').first()  # MS = Month Start
    
    print(f"Generating scores for {len(df_monthly)} monthly data points...")
    
    # Calculate risk scores for each month
    scores_added = 0
    for i in range(1, len(df_monthly)):
        current_row = df_monthly.iloc[i]
        prev_row = df_monthly.iloc[i-1]
        
        risk_score, action = calculate_risk_score(current_row, prev_row, position_type)
        
        if risk_score is not None:
            # Get the timestamp
            timestamp = df_monthly.index[i].strftime("%Y-%m-%d %H:%M:%S")
            
            # Add to database with correct timestamp
            add_risk_score(currency_pair, risk_score, action, position_type, timestamp=timestamp)
            scores_added += 1
            
            if scores_added % 12 == 0:  # Progress update every year
                print(f"  Added {scores_added} scores... (currently at {timestamp})")
    
    print(f"\n✅ Successfully generated and stored {scores_added} historical risk scores for {currency_pair}")
    print(f"   Date range: {df_monthly.index[1].strftime('%Y-%m-%d')} to {df_monthly.index[-1].strftime('%Y-%m-%d')}")

def clear_risk_scores():
    """Clear all entries from the risk_scores table"""
    import sqlite3
    from database import DB_NAME
    
    print("Clearing existing risk scores...")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM risk_scores")
    conn.commit()
    conn.close()
    print("✅ Risk scores table cleared.")

if __name__ == "__main__":
    # Initialize database
    print("Initializing database...")
    init_db()
    
    # Clear existing data first (to remove the incorrect "current timestamp" entries)
    clear_risk_scores()
    
    # Generate scores for EUR/USD (you can add more pairs if needed)
    currency_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF"]
    
    print("\n" + "="*70)
    print("HISTORICAL RISK SCORE GENERATOR")
    print("="*70 + "\n")
    
    for pair in currency_pairs:
        try:
            generate_historical_scores(pair, position_type="Long", years=5)
            print()
        except Exception as e:
            print(f"❌ Error generating scores for {pair}: {e}\n")
    
    print("="*70)
    print("✅ Historical score generation complete!")
    print("="*70)
    print("\nYou can now refresh your Streamlit app to see the historical data.")
