import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class SalesForecaster:
    def __init__(self):
        self.model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    
    def forecast(self, df: pd.DataFrame, periods: int = 6, date_col='Date', val_col='Revenue') -> pd.DataFrame:
        df = df.dropna(subset=[date_col, val_col]).copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Feature engineering
        df['time_idx'] = np.arange(len(df))
        df['month'] = df[date_col].dt.month
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        
        X = df[['time_idx', 'sin_month', 'cos_month']].values
        y = df[val_col].values
        
        self.model.fit(X, y)
        
        # Future dates
        last_date = df[date_col].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='ME')
        future_idx = np.arange(len(df), len(df) + periods)
        future_months = future_dates.month
        
        X_future = np.column_stack([
            future_idx,
            np.sin(2 * np.pi * future_months / 12),
            np.cos(2 * np.pi * future_months / 12)
        ])
        
        predictions = self.model.predict(X_future)
        
        # Confidence intervals (±15% based on historical std)
        std = df[val_col].std()
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions.clip(0),
            'Upper': (predictions + 1.5 * std).clip(0),
            'Lower': (predictions - 1.5 * std).clip(0)
        })
