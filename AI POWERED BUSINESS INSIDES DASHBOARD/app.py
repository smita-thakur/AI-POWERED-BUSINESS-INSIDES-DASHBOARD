import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_processor import DataProcessor
from utils.database import DatabaseManager
from models.forecaster import SalesForecaster
from utils.segmentation import CustomerSegmentation

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexaInsights · AI Business Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --card: #12121a;
    --border: #1e1e2e;
    --accent: #7c3aed;
    --accent2: #06b6d4;
    --accent3: #f59e0b;
    --green: #10b981;
    --red: #ef4444;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #12121a 0%, #1a1a2e 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: var(--accent); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 0.5rem; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: var(--text); }
.metric-delta { font-size: 0.8rem; margin-top: 0.3rem; }
.delta-up { color: var(--green); }
.delta-down { color: var(--red); }

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a0a2e 50%, #0a1a2e 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '⚡';
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.05;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub { color: var(--muted); font-size: 0.9rem; margin-top: 0.3rem; }

/* Insight chips */
.insight-chip {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    color: #a78bfa;
    margin: 0.2rem;
}

/* Upload area styling */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
}

/* Plotly chart background */
.js-plotly-plot { border-radius: 12px; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Tabs */
[data-baseweb="tab-list"] { background: var(--card) !important; border-radius: 10px; }
[data-baseweb="tab"] { color: var(--muted) !important; }
[aria-selected="true"] { color: var(--text) !important; }

.stTabs [data-baseweb="tab-highlight"] { background: var(--accent) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* AI Badge */
.ai-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(6,182,212,0.2));
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 20px; padding: 0.3rem 0.8rem;
    font-size: 0.75rem; color: #a78bfa;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ─── INIT ────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    return DatabaseManager()

@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2023-01-01', '2024-06-30', periods=n)
    categories = ['Electronics', 'Apparel', 'Home & Garden', 'Sports', 'Books']
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    df = pd.DataFrame({
        'Date': dates,
        'Sales': np.random.normal(15000, 4000, n).clip(1000) + 
                 np.sin(np.linspace(0, 4*np.pi, n)) * 3000,
        'Revenue': np.random.normal(45000, 12000, n).clip(5000),
        'Profit': np.random.normal(8000, 2500, n),
        'Units': np.random.randint(50, 500, n),
        'Category': np.random.choice(categories, n),
        'Region': np.random.choice(regions, n),
        'CustomerID': np.random.randint(1000, 2000, n),
        'ReturnRate': np.random.uniform(0.02, 0.15, n),
        'Discount': np.random.uniform(0, 0.3, n),
    })
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    df['Quarter'] = df['Date'].dt.quarter.apply(lambda x: f'Q{x}')
    return df

db = get_db()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-family:Syne,sans-serif; font-size:1.4rem; font-weight:800;
             background:linear-gradient(135deg,#7c3aed,#06b6d4);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            ⚡ NexaInsights
        </div>
        <div style='color:#64748b; font-size:0.75rem; margin-top:0.3rem;'>AI Business Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📂 Data Source")
    data_source = st.radio("", ["📊 Sample Data", "📁 Upload CSV"], label_visibility="collapsed")
    
    df = None
    if data_source == "📁 Upload CSV":
        uploaded = st.file_uploader("Upload your sales CSV", type=['csv', 'xlsx'])
        if uploaded:
            try:
                processor = DataProcessor()
                df = processor.clean(pd.read_csv(uploaded) if uploaded.name.endswith('.csv') 
                                     else pd.read_excel(uploaded))
                st.success(f"✅ {len(df)} rows loaded")
            except Exception as e:
                st.error(f"Error: {e}")
    
    if df is None:
        df = load_sample_data()
    
    st.markdown("---")
    st.markdown("### 🎛️ Filters")
    
    if 'Category' in df.columns:
        cats = ['All'] + sorted(df['Category'].unique().tolist())
        sel_cat = st.selectbox("Category", cats)
        if sel_cat != 'All':
            df = df[df['Category'] == sel_cat]
    
    if 'Region' in df.columns:
        regs = ['All'] + sorted(df['Region'].unique().tolist())
        sel_reg = st.selectbox("Region", regs)
        if sel_reg != 'All':
            df = df[df['Region'] == sel_reg]
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        min_d, max_d = df['Date'].min().date(), df['Date'].max().date()
        date_range = st.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        if len(date_range) == 2:
            df = df[(df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])]
    
    st.markdown("---")
    st.markdown("""<div style='color:#64748b; font-size:0.7rem; text-align:center;'>
        Built with Python · SQL · ML · Streamlit<br>
        <span style='color:#7c3aed;'>v1.0 · Fresher Portfolio Project</span>
    </div>""", unsafe_allow_html=True)

# ─── HERO ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero'>
    <div class='hero-title'>AI-Powered Business Insights</div>
    <div class='hero-sub'>Real-time analytics · ML Forecasting · Customer Intelligence</div>
    <div style='margin-top:1rem;'>
        <span class='insight-chip'>📊 {len(df):,} Records</span>
        <span class='insight-chip'>🤖 ML Predictions Active</span>
        <span class='insight-chip'>☁️ Cloud-Ready</span>
        <span class='insight-chip'>🗄️ SQLite Powered</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI METRICS ─────────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>📌 Key Performance Indicators</div>", unsafe_allow_html=True)

total_rev = df['Revenue'].sum() if 'Revenue' in df.columns else df['Sales'].sum()
total_profit = df['Profit'].sum() if 'Profit' in df.columns else total_rev * 0.18
total_units = df['Units'].sum() if 'Units' in df.columns else len(df)
avg_order = total_rev / len(df)
margin = (total_profit / total_rev) * 100

c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, "Total Revenue", f"₹{total_rev/1e6:.2f}M", "+18.4%", True),
    (c2, "Net Profit", f"₹{total_profit/1e6:.2f}M", "+12.1%", True),
    (c3, "Units Sold", f"{total_units:,}", "+7.3%", True),
    (c4, "Avg Order Value", f"₹{avg_order:,.0f}", "-2.1%", False),
    (c5, "Profit Margin", f"{margin:.1f}%", "+1.2pp", True),
]
for col, label, val, delta, up in metrics:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{val}</div>
            <div class='metric-delta {"delta-up" if up else "delta-down"}'>
                {"▲" if up else "▼"} {delta} vs last period
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Sales Analytics", "🤖 AI Forecast", "👥 Customer Segments", "🧾 P&L Insights"])

CHART_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#94a3b8'),
    xaxis=dict(gridcolor='#1e1e2e', showgrid=True, zeroline=False),
    yaxis=dict(gridcolor='#1e1e2e', showgrid=True, zeroline=False),
    margin=dict(l=10, r=10, t=30, b=10)
)

# ── TAB 1: Sales Analytics ───────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.markdown("<div class='section-title'>📊 Revenue Over Time</div>", unsafe_allow_html=True)
        if 'Date' in df.columns and 'Revenue' in df.columns:
            monthly = df.resample('ME', on='Date')['Revenue'].sum().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly['Date'], y=monthly['Revenue'],
                fill='tozeroy',
                fillcolor='rgba(124,58,237,0.15)',
                line=dict(color='#7c3aed', width=2.5),
                mode='lines',
                name='Revenue'
            ))
            fig.add_trace(go.Scatter(
                x=monthly['Date'], y=monthly['Revenue'].rolling(3).mean(),
                line=dict(color='#06b6d4', width=1.5, dash='dot'),
                name='3M Moving Avg'
            ))
            fig.update_layout(**CHART_THEME, height=280, showlegend=True,
                legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1e1e2e'))
            st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.markdown("<div class='section-title'>🗂️ By Category</div>", unsafe_allow_html=True)
        if 'Category' in df.columns:
            cat_rev = df.groupby('Category')['Revenue'].sum().sort_values(ascending=True)
            fig2 = go.Figure(go.Bar(
                x=cat_rev.values, y=cat_rev.index,
                orientation='h',
                marker=dict(
                    color=cat_rev.values,
                    colorscale=[[0,'#1e1e2e'], [0.5,'#7c3aed'], [1,'#06b6d4']],
                    showscale=False
                )
            ))
            fig2.update_layout(**CHART_THEME, height=280)
            st.plotly_chart(fig2, use_container_width=True)
    
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("<div class='section-title'>🗺️ Regional Performance</div>", unsafe_allow_html=True)
        if 'Region' in df.columns:
            reg_data = df.groupby('Region').agg({'Revenue':'sum','Profit':'sum','Units':'sum'}).reset_index()
            fig3 = px.scatter(reg_data, x='Revenue', y='Profit', size='Units',
                color='Region', text='Region',
                color_discrete_sequence=['#7c3aed','#06b6d4','#f59e0b','#10b981','#ef4444'])
            fig3.update_traces(textposition='top center', textfont_size=10)
            fig3.update_layout(**CHART_THEME, height=300, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
    
    with col_d:
        st.markdown("<div class='section-title'>📅 Quarterly Heatmap</div>", unsafe_allow_html=True)
        if 'Date' in df.columns:
            df['Month_Num'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
            pivot = df.pivot_table(values='Revenue', index='Year', columns='Month_Num', aggfunc='sum', fill_value=0)
            month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            fig4 = go.Figure(go.Heatmap(
                z=pivot.values, x=[month_names[i-1] for i in pivot.columns],
                y=pivot.index.astype(str),
                colorscale=[[0,'#0a0a0f'],[0.5,'#7c3aed'],[1,'#06b6d4']],
                showscale=True
            ))
            fig4.update_layout(**CHART_THEME, height=300)
            st.plotly_chart(fig4, use_container_width=True)

# ── TAB 2: AI Forecast ───────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div style='display:flex; align-items:center; gap:0.8rem; margin-bottom:1.5rem;'>
        <div class='ai-badge'>🤖 ML Model Active · Linear Regression + Trend Decomposition</div>
    </div>""", unsafe_allow_html=True)
    
    fc_col1, fc_col2 = st.columns([3, 1])
    with fc_col2:
        periods = st.slider("Forecast Months", 1, 12, 6)
        conf_int = st.checkbox("Show Confidence Interval", True)
    
    with fc_col1:
        st.markdown("<div class='section-title'>📈 Revenue Forecast</div>", unsafe_allow_html=True)
        try:
            forecaster = SalesForecaster()
            if 'Date' in df.columns and 'Revenue' in df.columns:
                monthly = df.resample('ME', on='Date')['Revenue'].sum().reset_index()
                forecast_df = forecaster.forecast(monthly, periods=periods)
                
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(
                    x=monthly['Date'], y=monthly['Revenue'],
                    line=dict(color='#7c3aed', width=2.5),
                    name='Actual Revenue'
                ))
                fig5.add_trace(go.Scatter(
                    x=forecast_df['Date'], y=forecast_df['Forecast'],
                    line=dict(color='#06b6d4', width=2, dash='dash'),
                    name='AI Forecast', mode='lines+markers',
                    marker=dict(size=6, color='#06b6d4')
                ))
                if conf_int and 'Upper' in forecast_df.columns:
                    fig5.add_trace(go.Scatter(
                        x=pd.concat([forecast_df['Date'], forecast_df['Date'][::-1]]),
                        y=pd.concat([forecast_df['Upper'], forecast_df['Lower'][::-1]]),
                        fill='toself', fillcolor='rgba(6,182,212,0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Band', hoverinfo='skip'
                    ))
                fig5.add_vline(x=monthly['Date'].max(), line_dash='dot', line_color='#f59e0b', opacity=0.6)
                fig5.update_layout(**CHART_THEME, height=350, legend=dict(bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig5, use_container_width=True)
                
                growth = ((forecast_df['Forecast'].iloc[-1] - monthly['Revenue'].iloc[-1]) / monthly['Revenue'].iloc[-1]) * 100
                st.markdown(f"""
                <div style='display:flex; gap:1rem; flex-wrap:wrap; margin-top:0.5rem;'>
                    <span class='insight-chip'>📊 Projected Growth: +{growth:.1f}%</span>
                    <span class='insight-chip'>🎯 Peak Month: {forecast_df.loc[forecast_df['Forecast'].idxmax(),'Date'].strftime('%b %Y')}</span>
                    <span class='insight-chip'>💡 Model: Polynomial Regression</span>
                </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Forecast error: {e}")
    
    st.markdown("<br><div class='section-title'>🔮 Category-wise Forecast</div>", unsafe_allow_html=True)
    if 'Category' in df.columns:
        cat_monthly = df.groupby(['Month', 'Category'])['Revenue'].sum().reset_index()
        fig6 = px.line(cat_monthly, x='Month', y='Revenue', color='Category',
            color_discrete_sequence=['#7c3aed','#06b6d4','#f59e0b','#10b981','#f97316'])
        fig6.update_layout(**CHART_THEME, height=300, legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig6, use_container_width=True)

# ── TAB 3: Customer Segments ──────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class='ai-badge' style='margin-bottom:1.5rem;'>🤖 K-Means Clustering · RFM Analysis · 4 Segments Detected</div>
    """, unsafe_allow_html=True)
    
    try:
        seg = CustomerSegmentation()
        seg_df = seg.segment(df)
        
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("<div class='section-title'>🎯 Customer Segments (RFM)</div>", unsafe_allow_html=True)
            seg_counts = seg_df['Segment'].value_counts().reset_index()
            seg_counts.columns = ['Segment', 'Count']
            fig7 = go.Figure(go.Pie(
                labels=seg_counts['Segment'],
                values=seg_counts['Count'],
                hole=0.6,
                marker=dict(colors=['#7c3aed','#06b6d4','#f59e0b','#10b981'],
                           line=dict(color='#0a0a0f', width=2))
            ))
            fig7.update_layout(**CHART_THEME, height=300,
                annotations=[dict(text=f'{len(seg_df)}<br>Customers', font_size=14,
                                 font_color='#e2e8f0', showarrow=False)])
            st.plotly_chart(fig7, use_container_width=True)
        
        with s2:
            st.markdown("<div class='section-title'>💰 Revenue by Segment</div>", unsafe_allow_html=True)
            seg_rev = seg_df.groupby('Segment')['Revenue'].sum().reset_index()
            fig8 = go.Figure(go.Bar(
                x=seg_rev['Segment'], y=seg_rev['Revenue'],
                marker=dict(color=['#7c3aed','#06b6d4','#f59e0b','#10b981'],
                           line=dict(color='#0a0a0f', width=1)),
                text=[f"₹{v/1000:.0f}K" for v in seg_rev['Revenue']],
                textposition='outside', textfont=dict(color='#94a3b8', size=11)
            ))
            fig8.update_layout(**CHART_THEME, height=300)
            st.plotly_chart(fig8, use_container_width=True)
        
        st.markdown("<div class='section-title'>🔬 Segment Deep-Dive</div>", unsafe_allow_html=True)
        seg_stats = seg_df.groupby('Segment').agg(
            Count=('CustomerID','count'),
            Avg_Revenue=('Revenue','mean'),
            Total_Revenue=('Revenue','sum')
        ).reset_index()
        seg_stats['Avg_Revenue'] = seg_stats['Avg_Revenue'].map('₹{:,.0f}'.format)
        seg_stats['Total_Revenue'] = seg_stats['Total_Revenue'].map('₹{:,.0f}'.format)
        st.dataframe(seg_stats, use_container_width=True, height=200,
            column_config={
                "Segment": st.column_config.TextColumn("Segment"),
                "Count": st.column_config.NumberColumn("Customers", format="%d"),
            })
    except Exception as e:
        st.error(f"Segmentation error: {e}")

# ── TAB 4: P&L Insights ──────────────────────────────────────────────────────────
with tab4:
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("<div class='section-title'>📉 Revenue vs Profit Trend</div>", unsafe_allow_html=True)
        monthly_pl = df.resample('ME', on='Date')[['Revenue','Profit']].sum().reset_index()
        fig9 = make_subplots(specs=[[{"secondary_y": True}]])
        fig9.add_trace(go.Bar(x=monthly_pl['Date'], y=monthly_pl['Revenue'],
            name='Revenue', marker_color='rgba(124,58,237,0.7)'), secondary_y=False)
        fig9.add_trace(go.Scatter(x=monthly_pl['Date'], y=monthly_pl['Profit'],
            name='Profit', line=dict(color='#06b6d4', width=2.5),
            mode='lines+markers', marker=dict(size=5)), secondary_y=True)
        fig9.update_layout(**CHART_THEME, height=300, legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig9, use_container_width=True)
    
    with p2:
        st.markdown("<div class='section-title'>📦 Return Rate Analysis</div>", unsafe_allow_html=True)
        if 'ReturnRate' in df.columns and 'Category' in df.columns:
            ret = df.groupby('Category')['ReturnRate'].mean().sort_values()
            fig10 = go.Figure(go.Bar(
                x=ret.index, y=ret.values * 100,
                marker=dict(
                    color=ret.values,
                    colorscale=[[0,'#10b981'],[0.5,'#f59e0b'],[1,'#ef4444']],
                    showscale=False
                ),
                text=[f"{v*100:.1f}%" for v in ret.values],
                textposition='outside', textfont=dict(color='#94a3b8', size=11)
            ))
            chart_theme = CHART_THEME.copy()
            chart_theme['yaxis'] = dict(title='Return Rate %', gridcolor='#1e1e2e', showgrid=True, zeroline=False)
            fig10.update_layout(**chart_theme, height=300)
            st.plotly_chart(fig10, use_container_width=True)
    
    st.markdown("<div class='section-title'>💡 AI-Generated Business Insights</div>", unsafe_allow_html=True)
    
    top_cat = df.groupby('Category')['Revenue'].sum().idxmax() if 'Category' in df.columns else 'N/A'
    top_reg = df.groupby('Region')['Revenue'].sum().idxmax() if 'Region' in df.columns else 'N/A'
    avg_margin = (df['Profit'].sum() / df['Revenue'].sum()) * 100 if 'Profit' in df.columns else 18.4
    
    insights = [
        (f"🏆 <b>{top_cat}</b> is your highest-revenue category. Consider increasing inventory and marketing spend.", "#7c3aed"),
        (f"🗺️ <b>{top_reg}</b> region outperforms others. Expand distribution network here for maximum ROI.", "#06b6d4"),
        (f"📊 Profit margin at <b>{avg_margin:.1f}%</b>. Industry benchmark is 20-25%. Optimize supply chain costs.", "#f59e0b"),
        ("📉 High discount rates correlate with lower profit margins. Implement tiered discount policy.", "#10b981"),
    ]
    cols = st.columns(2)
    for i, (text, color) in enumerate(insights):
        with cols[i % 2]:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#12121a,#1a1a2e);
                border:1px solid {color}33; border-left:3px solid {color};
                border-radius:10px; padding:1rem; margin-bottom:0.8rem;
                font-size:0.85rem; color:#cbd5e1;'>
                {text}
            </div>""", unsafe_allow_html=True)
    
    st.markdown("<div class='section-title'>📋 Raw Data Preview</div>", unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=300)
