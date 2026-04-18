import pandas as pd
import numpy as np

class DataProcessor:
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Drop duplicates
        df.drop_duplicates(inplace=True)
        
        # Parse date columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        # Fill numeric nulls with median
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical nulls
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # Remove outliers (IQR method) for sales/revenue cols
        for col in ['Sales', 'Revenue', 'Profit']:
            if col in df.columns:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                df = df[(df[col] >= Q1) & (df[col] <= Q3)]
        
        return df
