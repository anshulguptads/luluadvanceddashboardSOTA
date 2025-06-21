import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Lulu Retail AI Command Center", layout="wide")

@st.cache_data
def load_data():
    df_products = pd.read_csv("products_master.csv")
    df_stores = pd.read_csv("stores_master.csv")
    df_calendar = pd.read_csv("calendar_master.csv")
    df_inventory = pd.read_csv("inventory_transactions.csv")
    df_sales = pd.read_csv("sales_transactions.csv")
    df_events = pd.read_csv("event_transactions.csv")
    return df_products, df_stores, df_calendar, df_inventory, df_sales, df_events

df_products, df_stores, df_calendar, df_inventory, df_sales, df_events = load_data()
df_sales['Date'] = pd.to_datetime(df_sales['Date'])
df_inventory['Date'] = pd.to_datetime(df_inventory['Date'])
df_events['Date'] = pd.to_datetime(df_events['Date'])

# Merge for analytics
df_sales = df_sales.merge(df_products[['Product_ID', 'Category', 'Brand', 'Cost']], on='Product_ID', how='left')
df_inventory = df_inventory.merge(df_products[['Product_ID', 'Category', 'Brand']], on='Product_ID', how='left')

# Global filters
st.sidebar.header("Global Filters")
store_list = ['All'] + df_stores['Store_Name'].tolist()
category_list = ['All'] + sorted(df_products['Category'].unique())
store_sel = st.sidebar.selectbox("Store", store_list)
category_sel = st.sidebar.selectbox("Category", category_list)
date_range = st.sidebar.date_input("Date Range", [df_sales['Date'].min(), df_sales['Date'].max()])

# Filter logic
sales_data = df_sales.copy()
inv_data = df_inventory.copy()
event_data = df_events.copy()
if store_sel != 'All':
    store_id = df_stores[df_stores['Store_Name'] == store_sel]['Store_ID'].values[0]
    sales_data = sales_data[sales_data['Store_ID'] == store_id]
    inv_data = inv_data[inv_data['Store_ID'] == store_id]
    event_data = event_data[event_data['Store_ID'] == store_id]
if category_sel != 'All':
    sales_data = sales_data[sales_data['Category'] == category_sel]
    inv_data = inv_data[inv_data['Category'] == category_sel]
    event_data = event_data[event_data['Product_ID'].isin(df_products[df_products['Category']==category_sel]['Product_ID'])]
sales_data = sales_data[(sales_data['Date'] >= pd.to_datetime(date_range[0])) & (sales_data['Date'] <= pd.to_datetime(date_range[1]))]
inv_data = inv_data[(inv_data['Date'] >= pd.to_datetime(date_range[0])) & (inv_data['Date'] <= pd.to_datetime(date_range[1]))]
event_data = event_data[(event_data['Date'] >= pd.to_datetime(date_range[0])) & (event_data['Date'] <= pd.to_datetime(date_range[1]))]

# Tabs
tabs = st.tabs([
    "🏠 Executive Dashboard",
    "📊 Descriptive/Diagnostic",
    "💸 Loss & Event Analysis",
    "🔮 Predictive/Prescriptive",
    "🏬 Store/Brand",
    "🗂️ Raw Data"
])

# Executive Dashboard
with tabs[0]:
    st.header("Executive Command Center")
    total_sales = sales_data['Net_Sales_AED'].sum()
    total_units = sales_data['Units_Sold'].sum()
    promo_sales = sales_data[sales_data['Promotion_Flag']=='Y']['Net_Sales_AED'].sum()
    promo_pct = promo_sales / total_sales * 100 if total_sales > 0 else 0
    unique_skus = sales_data['Product_ID'].nunique()
    stores_count = sales_data['Store_ID'].nunique()
    categories = sales_data['Category'].nunique()
    brands = sales_data['Brand'].nunique()
    outlier_days = sales_data[sales_data['Units_Sold'] > sales_data['Units_Sold'].quantile(0.99)].shape[0]
    out_of_stock = inv_data[inv_data['Closing_Stock'] == 0].shape[0]

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Net Sales (AED)", f"{total_sales:,.0f}")
    col2.metric("Units Sold", f"{total_units:,}")
    col3.metric("Promo Sales (%)", f"{promo_pct:.1f}")
    col4.metric("Active SKUs", unique_skus)
    col5.metric("Categories", categories)
    col6.metric("Brands", brands)
    col7.metric("OOS Incidents", out_of_stock)

    st.subheader("Top/Bottom 10 SKUs by Sales and Margin")
    kpi_df = sales_data.groupby('Product_ID').agg({
        'Net_Sales_AED':'sum',
        'Units_Sold':'sum',
        'Cost':'mean'
    }).reset_index()
    kpi_df = kpi_df.merge(df_products[['Product_ID','Product_Name','Category','Brand','Cost']], on='Product_ID')
    kpi_df['Margin'] = kpi_df['Net_Sales_AED'] - (kpi_df['Units_Sold'] * kpi_df['Cost_y'])
    top10 = kpi_df.sort_values('Net_Sales_AED', ascending=False).head(10)
    bottom10 = kpi_df.sort_values('Net_Sales_AED', ascending=True).head(10)
    st.write("**Top 10 SKUs by Sales**")
    st.dataframe(top10[['Product_Name','Category','Brand','Net_Sales_AED','Margin']])
    st.write("**Bottom 10 SKUs by Sales**")
    st.dataframe(bottom10[['Product_Name','Category','Brand','Net_Sales_AED','Margin']])

    st.subheader("Inventory Days of Supply (SKU-level)")
    avg_daily_sales = sales_data.groupby('Product_ID')['Units_Sold'].mean().reset_index()
    avg_stock = inv_data.groupby('Product_ID')['Closing_Stock'].mean().reset_index()
    dos = avg_stock.merge(avg_daily_sales, on='Product_ID', how='left')
    dos['Days_of_Supply'] = dos['Closing_Stock'] / (dos['Units_Sold']+1)
    dos = dos.merge(df_products[['Product_ID','Product_Name']], on='Product_ID')
    st.dataframe(dos[['Product_Name','Days_of_Supply']].sort_values('Days_of_Supply', ascending=False).head(12))

# Descriptive/Diagnostic
with tabs[1]:
    st.header("Descriptive & Diagnostic Analytics")
    st.subheader("Stock Positioning: Stock Levels and Stock-Outs")
    inv_plot = inv_data.groupby('Date')['Closing_Stock'].sum().reset_index()
    fig1, ax1 = plt.subplots(figsize=(9, 3))
    ax1.plot(inv_plot['Date'], inv_plot['Closing_Stock'], marker='o')
    ax1.set_ylabel("Total Closing Stock")
    ax1.set_title("Total Stock Position Over Time")
    st.pyplot(fig1)
    st.subheader("Sales Tracking: Store/Region/Trend")
    trend = sales_data.groupby(['Date'])['Net_Sales_AED'].sum().reset_index()
    fig2, ax2 = plt.subplots(figsize=(9, 3))
    ax2.plot(trend['Date'], trend['Net_Sales_AED'], marker='.', color="#1b8")
    ax2.set_ylabel("Net Sales (AED)")
    ax2.set_title("Net Sales Trend")
    st.pyplot(fig2)
    st.subheader("Promotion Impact: Uplift Analysis")
    promo = sales_data[sales_data['Promotion_Flag']=='Y']
    nonpromo = sales_data[sales_data['Promotion_Flag']=='N']
    promo_group = promo.groupby('Category')['Net_Sales_AED'].mean()
    nonpromo_group = nonpromo.groupby('Category')['Net_Sales_AED'].mean()
    uplift = ((promo_group - nonpromo_group) / (nonpromo_group+1)) * 100
    st.dataframe(pd.DataFrame({'Promo Sales': promo_group, 'Non-Promo Sales': nonpromo_group, 'Uplift %': uplift.round(2)}))

    st.subheader("Dead Stock & Slow Movers")
    slow = sales_data.groupby('Product_ID')['Units_Sold'].sum().reset_index()
    slow = slow.merge(df_products[['Product_ID','Product_Name','Category']], on='Product_ID')
    st.dataframe(slow[slow['Units_Sold'] == 0][['Product_Name','Category','Units_Sold']])

    st.subheader("Stock-Out Root Cause Diagnostic")
    inv_merged = inv_data.merge(sales_data[['Date','Store_ID','Product_ID','Units_Sold']], on=['Date','Store_ID','Product_ID'], how='left')
    inv_merged['Stockout'] = (inv_merged['Closing_Stock'] == 0)
    high_sales = sales_data['Units_Sold'].quantile(0.90)
    causes = []
    for _, row in inv_merged[inv_merged['Stockout']].head(300).iterrows():
        if row['Units_Sold'] > high_sales:
            cause = "High Demand"
        elif row['Inward_Qty'] == 0:
            cause = "Supply Chain Delay"
        else:
            cause = "Forecast Miss"
        causes.append({'Date': row['Date'], 'Store_ID': row['Store_ID'], 'Product_ID': row['Product_ID'], 'Cause': cause})
    if causes:
        st.dataframe(pd.DataFrame(causes))

# Loss & Event Analysis
with tabs[2]:
    st.header("Loss / Shrinkage / Returns Analytics")
    st.subheader("Loss Event Frequency & Value")
    loss_summary = event_data.groupby('Event_Type').agg({'Qty':'sum','Value_Loss':'sum'}).reset_index()
    st.dataframe(loss_summary)
    st.subheader("Event Trends Over Time")
    event_trend = event_data.groupby(['Date','Event_Type'])['Value_Loss'].sum().unstack(fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(9, 3))
    event_trend.plot(ax=ax3)
    ax3.set_ylabel("Loss Value (AED)")
    st.pyplot(fig3)
    st.subheader("SKU/Store Loss Table (Top 10)")
    top_loss = event_data.groupby(['Product_ID'])['Value_Loss'].sum().sort_values(ascending=False).head(10)
    st.dataframe(top_loss.reset_index().merge(df_products[['Product_ID','Product_Name']], on='Product_ID'))

# Predictive/Prescriptive
with tabs[3]:
    st.header("Predictive & Prescriptive Analytics")
    st.subheader("Sales Forecasting (Prophet)")
    sku_list = df_products['Product_ID']
    selected_sku = st.selectbox("Select SKU for Forecasting", sku_list)
    fc_days = st.slider("Forecast Days", min_value=14, max_value=60, value=30)
    sku_df = sales_data[sales_data['Product_ID'] == selected_sku].groupby('Date')['Units_Sold'].sum().reset_index()
    sku_df.columns = ['ds','y']
    if len(sku_df) > 30:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        m.fit(sku_df)
        future = m.make_future_dataframe(periods=fc_days)
        fc = m.predict(future)
        fig4 = m.plot(fc)
        st.pyplot(fig4)
        # Forecast Accuracy if actuals available
        fc_valid = fc.merge(sku_df, on='ds', how='left').dropna()
        mape = np.mean(np.abs(fc_valid['y'] - fc_valid['yhat'])/(fc_valid['y']+1)) * 100
        rmse = np.sqrt(mean_squared_error(fc_valid['y'], fc_valid['yhat']))
        st.metric("MAPE (In-sample)", f"{mape:.2f}%")
        st.metric("RMSE", f"{rmse:.2f}")
        st.dataframe(fc[['ds','yhat','yhat_lower','yhat_upper']].tail(fc_days))
    else:
        st.warning("Not enough historical data for this SKU for forecasting.")

    st.subheader("Replenishment Alerts (Predicted Stockout Risk)")
    avg_sales = sales_data.groupby('Product_ID')['Units_Sold'].mean().to_dict()
    rep_alerts = []
    for _, row in inv_data.iterrows():
        avg = avg_sales.get(row['Product_ID'], 0)
        doc = row['Closing_Stock'] / (avg+1)
        if doc < 2:
            rep_alerts.append({'Date': row['Date'], 'Store_ID': row['Store_ID'], 'Product_ID': row['Product_ID'], 'Closing_Stock': row['Closing_Stock'], 'Days_of_Cover': doc})
    if rep_alerts:
        st.dataframe(pd.DataFrame(rep_alerts).head(20))

    st.subheader("Promotion Recommendation Engine (Velocity & Seasonality)")
    base = sales_data.groupby(['Product_ID','Promotion_Flag'])['Units_Sold'].sum().unstack(fill_value=0)
    base['Promo_Uplift'] = (base['Y'] - base['N']) / (base['N']+1)
    sales_data['Month'] = sales_data['Date'].dt.month
    seasonality = sales_data.groupby(['Product_ID','Month'])['Units_Sold'].mean().unstack(fill_value=0).std(axis=1)
    base = base.join(seasonality.rename("Seasonality_Std"))
    recs = base.sort_values(['Promo_Uplift','Seasonality_Std'], ascending=False).head(10)
    recs = recs.join(df_products.set_index('Product_ID')[['Product_Name','Category','Brand']])
    st.dataframe(recs)

# Store/Brand
with tabs[4]:
    st.header("Store, Category & Brand Analysis")
    st.subheader("Net Sales by Store")
    store_sales = sales_data.groupby('Store_ID')['Net_Sales_AED'].sum().reset_index()
    store_sales = store_sales.merge(df_stores[['Store_ID','Store_Name']], on='Store_ID')
    fig5, ax5 = plt.subplots(figsize=(9, 4))
    sns.barplot(y='Store_Name', x='Net_Sales_AED', data=store_sales.sort_values('Net_Sales_AED', ascending=False), ax=ax5)
    ax5.set_title("Net Sales by Store")
    st.pyplot(fig5)

    st.subheader("Brand/Category Share of Wallet")
    cat_share = sales_data.groupby('Category')['Net_Sales_AED'].sum()
    brand_share = sales_data.groupby('Brand')['Net_Sales_AED'].sum()
    st.write("**Category Share of Wallet**")
    st.dataframe(cat_share.sort_values(ascending=False))
    st.write("**Brand Share of Wallet**")
    st.dataframe(brand_share.sort_values(ascending=False))

    st.subheader("Category x Store Heatmap")
    cat_store = sales_data.groupby(['Category','Store_ID'])['Net_Sales_AED'].sum().unstack(fill_value=0)
    fig6, ax6 = plt.subplots(figsize=(11, 4))
    sns.heatmap(cat_store, annot=False, fmt=".0f", cmap="YlGnBu", ax=ax6)
    ax6.set_title("Category x Store: Net Sales")
    st.pyplot(fig6)

# Raw Data
with tabs[5]:
    st.header("Raw Data (Download & Explore)")
    with st.expander("Sales Data (first 500 rows)"):
        st.dataframe(sales_data.head(500))
    with st.expander("Inventory Data (first 500 rows)"):
        st.dataframe(inv_data.head(500))
    with st.expander("Event Data (first 500 rows)"):
        st.dataframe(event_data.head(500))
    st.download_button("Download Sales Data (CSV)", sales_data.to_csv(index=False), "sales_data.csv")
    st.download_button("Download Inventory Data (CSV)", inv_data.to_csv(index=False), "inventory_data.csv")
    st.download_button("Download Event Data (CSV)", event_data.to_csv(index=False), "event_data.csv")

st.markdown("""
---
<b>Next-Gen Retail AI Dashboard • Advanced Analytics • All Modules Integrated</b>
""", unsafe_allow_html=True)
