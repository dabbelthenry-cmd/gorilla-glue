import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import timedelta

class MLEngine:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.mse = 0.0

    def prepare_features(self, df):
        """
        Creates features for the model.
        """
        data = df.copy()
        data['Returns'] = data['Close'].pct_change()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        # Drop NaN values created by rolling windows
        data = data.dropna()
        return data

    def train_and_predict(self, df, forecast_days=30):
        """
        Trains the model on historical data and predicts the next 'forecast_days'.
        Returns a DataFrame with dates, predicted prices, and confidence intervals.
        """
        # Prepare data
        data = self.prepare_features(df)
        
        if len(data) < 100:
            return None # Not enough data
            
        # Feature Engineering
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'Volatility']
        
        # Create target: Predict next day's Close
        data['Target'] = data['Close'].shift(-1)
        data = data.dropna()
        
        X = data[features]
        y = data['Target']
        
        # Train Model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate Training Error (MSE) for Confidence Interval
        predictions = self.model.predict(X)
        self.mse = mean_squared_error(y, predictions)
        std_dev = np.sqrt(self.mse)
        
        # Forecast Loop
        future_dates = []
        future_prices = []
        lower_bounds = []
        upper_bounds = []
        
        last_row = df.iloc[-1].copy()
        current_date = df.index[-1]
        
        # We need to iteratively update the 'last_row' features to predict the next step
        # This is a simplified recursive forecasting approach
        
        # Initialize with the last known data
        current_features = last_row.copy()
        
        # Ensure we have the necessary columns in current_features
        # We might need to re-calculate rolling metrics if we want high accuracy, 
        # but for this demo, we will approximate or just use the last known values for slow-moving indicators
        # and update price-based features.
        
        # Better approach for this demo: 
        # Use the trained model to predict t+1, then assume t+1 is true, predict t+2...
        
        # Re-construct the last feature set from the full dataframe to be safe
        last_features_df = self.prepare_features(df).iloc[[-1]][features]
        current_input = last_features_df.values[0]
        
        # To make it dynamic, we'd need to re-calc indicators. 
        # Simplified: We will project a trend based on the model's prediction of the *next* step 
        # and apply some noise/volatility for the range.
        
        # Actually, let's do a direct prediction for the next step, 
        # and then project the confidence interval widening over time.
        
        next_price = self.model.predict(last_features_df)[0]
        
        for i in range(forecast_days):
            current_date += timedelta(days=1)
            # Skip weekends if needed, but for simplicity we keep all days or check weekday
            if current_date.weekday() >= 5: # 5=Sat, 6=Sun
                continue
                
            # For the sake of the "Simulation", we will project the price 
            # slightly adjusting based on the last trend or just keeping it near the predicted level
            # Since we can't easily re-generate 'High', 'Low', 'Volume' for the next input without a separate model.
            
            # We will use the 'next_price' as the base and add some random drift for the 'simulation' aspect
            # or just repeat the prediction if the model is static.
            
            # Let's use a random walk with drift based on the model's last prediction direction
            # drift = (next_price - last_row['Close']) / last_row['Close']
            
            # Simple projection: The model predicts the *next* day. 
            # We will linearly interpolate or just show the range around that level, 
            # but the user asked for 1 month forward.
            
            # Let's use the volatility to widen the cone.
            uncertainty = std_dev * np.sqrt(i + 1) # Square root of time rule for volatility
            
            # We'll keep the price relatively stable or trending slightly towards the model's 1-day prediction
            # This is a limitation of recursive forecasting without a full vector autoregression.
            # We will assume the 1-day prediction sets the "trend" for the month.
            
            trend = (next_price - last_row['Close'])
            projected_price = last_row['Close'] + (trend * (i + 1))
            
            future_dates.append(current_date)
            future_prices.append(projected_price)
            lower_bounds.append(projected_price - (1.96 * uncertainty)) # 95% Confidence
            upper_bounds.append(projected_price + (1.96 * uncertainty))
            
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': future_prices,
            'Lower_Bound': lower_bounds,
            'Upper_Bound': upper_bounds
        })
        
        return forecast_df
