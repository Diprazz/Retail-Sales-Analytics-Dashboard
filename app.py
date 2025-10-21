import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Page Configuration
st.set_page_config(
    page_title="Retail Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load and preprocess the retail data"""
    try:
        df = pd.read_csv('data/superstore_sales.csv')
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        df['Ship_Date'] = pd.to_datetime(df['Ship_Date'])
        
        # Create additional features
        df['Order_Month'] = df['Order_Date'].dt.to_period('M')
        df['Order_Year'] = df['Order_Date'].dt.year
        df['Order_Quarter'] = df['Order_Date'].dt.quarter
        df['Processing_Time'] = (df['Ship_Date'] - df['Order_Date']).dt.days
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    st.markdown('<h1 class="main-header">🏪 Retail Sales Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Business Intelligence & Forecasting Platform")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Date range filter
    min_date = df['Order_Date'].min()
    max_date = df['Order_Date'].max()
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)
    
    # Region filter
    regions = ['All'] + list(df['Region'].unique())
    selected_region = st.sidebar.selectbox("Select Region", regions)
    
    # Category filter
    categories = ['All'] + list(df['Category'].unique())
    selected_category = st.sidebar.selectbox("Select Category", categories)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    
    filtered_df = filtered_df[
        (filtered_df['Order_Date'] >= pd.to_datetime(start_date)) & 
        (filtered_df['Order_Date'] <= pd.to_datetime(end_date))
    ]
    
    # Main Dashboard
    st.markdown('<h2 class="section-header">📈 Key Performance Indicators</h2>', unsafe_allow_html=True)
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = filtered_df['Sales'].sum()
        st.metric("Total Sales", f"${total_sales:,.2f}")
    
    with col2:
        total_profit = filtered_df['Profit'].sum()
        profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
        st.metric("Total Profit", f"${total_profit:,.2f}", f"{profit_margin:.1f}% Margin")
    
    with col3:
        avg_order_value = filtered_df['Sales'].mean()
        st.metric("Average Order Value", f"${avg_order_value:.2f}")
    
    with col4:
        total_orders = filtered_df['Order_ID'].nunique()
        st.metric("Total Orders", f"{total_orders:,}")
    
    # Sales Trends
    st.markdown('<h2 class="section-header">📊 Sales Performance Over Time</h2>', unsafe_allow_html=True)
    
    # Monthly sales trend
    monthly_sales = filtered_df.groupby(filtered_df['Order_Date'].dt.to_period('M')).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order_ID': 'nunique'
    }).reset_index()
    monthly_sales['Order_Date'] = monthly_sales['Order_Date'].dt.to_timestamp()
    
    fig_sales = px.line(monthly_sales, x='Order_Date', y='Sales', 
                       title='Monthly Sales Trend', markers=True)
    st.plotly_chart(fig_sales, use_container_width=True)
    
    # Product Performance
    st.markdown('<h2 class="section-header">🏆 Product & Category Performance</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by Category
        category_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
        fig_category = px.pie(category_sales, values='Sales', names='Category', 
                             title='Sales Distribution by Category')
        st.plotly_chart(fig_category, use_container_width=True)
    
    with col2:
        # Top Products
        top_products = filtered_df.groupby('Product_Name').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Quantity': 'sum'
        }).nlargest(10, 'Sales').reset_index()
        
        fig_products = px.bar(top_products, x='Sales', y='Product_Name', 
                             orientation='h', title='Top 10 Products by Sales')
        st.plotly_chart(fig_products, use_container_width=True)
    
    # Regional Performance
    st.markdown('<h2 class="section-header">🌍 Regional Performance</h2>', unsafe_allow_html=True)
    
    regional_sales = filtered_df.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order_ID': 'nunique'
    }).reset_index()
    
    fig_region = px.bar(regional_sales, x='Region', y='Sales', 
                       color='Profit', title='Sales & Profit by Region')
    st.plotly_chart(fig_region, use_container_width=True)
    
    # Customer Segmentation Preview
    st.markdown('<h2 class="section-header">👥 Customer Insights</h2>', unsafe_allow_html=True)
    
    customer_stats = filtered_df.groupby('Customer_ID').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order_ID': 'nunique',
        'Order_Date': 'max'
    }).reset_index()
    
    customer_stats.columns = ['Customer_ID', 'Total_Spent', 'Total_Profit', 'Order_Count', 'Last_Order']
    customer_stats['Avg_Order_Value'] = customer_stats['Total_Spent'] / customer_stats['Order_Count']
    
    st.dataframe(customer_stats.nlargest(10, 'Total_Spent'), use_container_width=True)

if __name__ == "__main__":
    main()