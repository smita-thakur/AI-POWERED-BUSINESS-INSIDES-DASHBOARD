# ⚡ NexaInsights — AI-Powered Business Dashboard

> A full-stack Data Analytics project combining Python, SQL, Machine Learning & Cloud Deployment

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red) ![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange) ![SQL](https://img.shields.io/badge/SQL-SQLite-green)

---

## 🚀 Features

| Feature | Tech Used |
|---|---|
| 📊 Sales Trend Analysis | Pandas + Plotly |
| 🤖 AI Revenue Forecasting | Scikit-learn (Poly Regression) |
| 👥 Customer Segmentation | RFM Analysis + K-Means |
| 📉 P&L Insights | NumPy + Plotly |
| 🗄️ SQL Storage | SQLite / PostgreSQL |
| ☁️ Cloud Deployment | Streamlit Cloud / AWS / GCP |

---

## 🛠️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/ai-insights-dashboard
cd ai-insights-dashboard

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

## 📁 Project Structure

```
ai-insights-dashboard/
├── app.py                  # Main Streamlit dashboard
├── requirements.txt        # Python dependencies
├── models/
│   └── forecaster.py       # ML sales forecasting model
├── utils/
│   ├── data_processor.py   # Data cleaning pipeline
│   ├── database.py         # SQL database manager
│   └── segmentation.py     # Customer RFM segmentation
├── data/
│   └── insights.db         # SQLite database (auto-created)
└── README.md
```

## ☁️ Cloud Deployment

### Streamlit Cloud (Free & Easy)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → Deploy!

### AWS EC2
```bash
# On EC2 instance
pip install -r requirements.txt
streamlit run app.py --server.port 80 --server.address 0.0.0.0
```

## 🧠 ML Model Details

- **Algorithm**: Polynomial Regression (degree=2) with seasonal features
- **Features**: Time index, sin/cos month encoding (captures seasonality)
- **Output**: Point forecast + 90% confidence interval
- **Evaluation**: RMSE, MAE on holdout set

## 📊 Database Schema

```sql
CREATE TABLE sales (
    id INTEGER PRIMARY KEY,
    date TEXT,
    category TEXT,
    region TEXT,
    revenue REAL,
    profit REAL,
    units INTEGER,
    customer_id INTEGER
);
```

---

**Built by:** [Your Name] | Fresher Data Analyst Portfolio Project
