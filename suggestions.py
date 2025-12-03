import pandas as pd
import numpy as np

def analyze_market_condition(data):
    """
    Analyzes the market condition based on recent data.
    Returns a dictionary with analysis.
    """
    if data is None or data.empty:
        return {"status": "No Data", "signal": "Neutral"}
    
    last_row = data.iloc[-1]
    prev_row = data.iloc[-2]
    
    signal = "Neutral"
    reason = []
    
    # Simple crossover strategy logic
    if 'SMA_20' in last_row and 'SMA_50' in last_row:
        if last_row['SMA_20'] > last_row['SMA_50'] and prev_row['SMA_20'] <= prev_row['SMA_50']:
            signal = "Buy"
            reason.append("Golden Cross (SMA 20 crossed above SMA 50)")
        elif last_row['SMA_20'] < last_row['SMA_50'] and prev_row['SMA_20'] >= prev_row['SMA_50']:
            signal = "Sell"
            reason.append("Death Cross (SMA 20 crossed below SMA 50)")
            
    # RSI Logic
    if 'RSI' in last_row:
        if last_row['RSI'] < 30:
            if signal == "Buy":
                reason.append("RSI Oversold (Confluence)")
            else:
                signal = "Buy"
                reason.append("RSI Oversold")
        elif last_row['RSI'] > 70:
            if signal == "Sell":
                reason.append("RSI Overbought (Confluence)")
            else:
                signal = "Sell"
                reason.append("RSI Overbought")
                
    return {
        "status": "Active",
        "signal": signal,
        "reason": "; ".join(reason) if reason else "No strong signal"
    }

def calculate_hedging_ratio(portfolio_value, beta=1.0):
    """
    Calculates a simple hedging ratio.
    """
    # Placeholder logic
    return portfolio_value * beta
