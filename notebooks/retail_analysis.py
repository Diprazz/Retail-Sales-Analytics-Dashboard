# Retail Sales Analysis - Python Script
# This can be run directly without Jupyter notebook issues

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("🛍️ Retail Sales Analytics")
print("=========================")

# Load data
df = pd.read_csv('data/superstore_sales.csv')
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Total Sales: ${df['Sales'].sum():,.2f}")
print(f"Total Profit: ${df['Profit'].sum():,.2f}")

# Display first rows
print("\nFirst 5 rows:")
print(df.head())

# Basic analysis
print("\n📊 Sales by Category:")
category_sales = df.groupby('Category')['Sales'].sum()
print(category_sales)

print("\n🌍 Sales by Region:")
region_sales = df.groupby('Region')['Sales'].sum()
print(region_sales)

# Create visualizations
plt.figure(figsize=(15, 10))

# Plot 1: Sales by Category
plt.subplot(2, 2, 1)
category_sales.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen'])
plt.title('Sales by Category')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45)

# Plot 2: Monthly trend
plt.subplot(2, 2, 2)
monthly_sales = df.groupby(df['Order_Date'].dt.to_period('M'))['Sales'].sum()
monthly_sales.plot(kind='line', marker='o', color='orange')
plt.title('Monthly Sales Trend')
plt.ylabel('Sales ($)')
plt.grid(True)

# Plot 3: Regional sales
plt.subplot(2, 2, 3)
region_sales.plot(kind='bar', color=['red', 'blue', 'green', 'purple'])
plt.title('Sales by Region')
plt.ylabel('Sales ($)')

# Plot 4: Profit distribution
plt.subplot(2, 2, 4)
plt.hist(df['Profit'], bins=20, color='lightseagreen', alpha=0.7)
plt.title('Profit Distribution')
plt.xlabel('Profit ($)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

print("\n✅ Analysis complete! Run 'streamlit run app.py' for interactive dashboard.")