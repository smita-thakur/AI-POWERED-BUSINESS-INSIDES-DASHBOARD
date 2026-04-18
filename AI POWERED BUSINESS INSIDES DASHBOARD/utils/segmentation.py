import pandas as pd
import numpy as np

class CustomerSegmentation:
    def segment(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Ensure required columns
        if 'CustomerID' not in df.columns:
            df['CustomerID'] = np.random.randint(1000, 2000, len(df))
        if 'Revenue' not in df.columns and 'Sales' in df.columns:
            df['Revenue'] = df['Sales']
        if 'Date' not in df.columns:
            df['Date'] = pd.Timestamp.now()
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # RFM Analysis
        snapshot = df['Date'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('CustomerID').agg({
            'Date': lambda x: (snapshot - x.max()).days,       # Recency
            'CustomerID': 'count',                              # Frequency
            'Revenue': 'sum'                                    # Monetary
        }).rename(columns={'Date': 'Recency', 'CustomerID': 'Frequency', 'Revenue': 'Monetary'})
        
        # Score 1-4
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1]).astype(int)
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4]).astype(int)
        rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
        
        # Segment labels
        def label(score):
            if score >= 10: return '🌟 Champions'
            elif score >= 8: return '💎 Loyal Customers'
            elif score >= 5: return '⚡ Potential Loyalists'
            else: return '😴 At-Risk Customers'
        
        rfm['Segment'] = rfm['RFM_Score'].apply(label)
        rfm = rfm.reset_index()
        
        return df.merge(rfm[['CustomerID','Segment','RFM_Score']], on='CustomerID', how='left')
