import sqlite3
import pandas as pd
from datetime import datetime
import os

class DatabaseManager:
    def __init__(self, db_name='business_insights.db'):
        self.db_path = db_name
        self.init_db()
    
    def init_db(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sales table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                sales REAL,
                revenue REAL,
                profit REAL,
                units INTEGER,
                category TEXT,
                region TEXT,
                customer_id INTEGER,
                return_rate REAL,
                discount REAL,
                created_at TEXT
            )
        ''')
        
        # Forecasts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                forecast_date TEXT,
                predicted_revenue REAL,
                upper_bound REAL,
                lower_bound REAL,
                created_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_sales_data(self, df):
        """Save sales data to database"""
        conn = sqlite3.connect(self.db_path)
        df['created_at'] = datetime.now().isoformat()
        df.to_sql('sales', conn, if_exists='append', index=False)
        conn.close()
    
    def load_sales_data(self):
        """Load sales data from database"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM sales", conn)
        conn.close()
        return df
    
    def save_forecast(self, forecast_df):
        """Save forecast to database"""
        conn = sqlite3.connect(self.db_path)
        forecast_df['created_at'] = datetime.now().isoformat()
        forecast_df.to_sql('forecasts', conn, if_exists='append', index=False)
        conn.close()
    
    def get_latest_forecast(self):
        """Get latest forecast"""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM forecasts WHERE created_at = (SELECT MAX(created_at) FROM forecasts)"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
