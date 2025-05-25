import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import plotly.express as px
import hashlib
import pickle
from pathlib import Path

st.set_page_config(page_title="AI Agent for Post-Sale Strategy", layout="wide")

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    df['streamer_id_num'] = df['streamer_id'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values(['streamer_id_num', 'date'])
    fields = [
        'streamer_id', 'product_id', 'date', 'price', 'discount_rate', 'views',
        'cost_per_unit', 'shelf_life_days', 'holiday_flag', 'quantity_sold'
    ]
    df = df[fields].replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df['quantity_sold'] > 0]
    return df

@st.cache_data
def calculate_cce_and_features(_snack_df):
    df = _snack_df.copy()
    base_model = LinearRegression().fit(np.log(df[['price']]), np.log(df['quantity_sold']))
    df['baseline_logq'] = base_model.predict(np.log(df[['price']]))
    df['uplift'] = np.log(df['quantity_sold']) - df['baseline_logq']
    cce_avg = df.groupby('streamer_id')['uplift'].mean().reset_index().rename(columns={'uplift':'CCE'})
    if 'conversion_rate' in df.columns:
        df['estimated_CCE'] = (
            df['quantity_sold'] / df['baseline_logq'].apply(np.exp)
        ) * (df['conversion_rate']/df['conversion_rate'].mean()) * np.log(df['views']+1)
        blend = df.groupby('streamer_id')['estimated_CCE'].mean().reset_index()
        cce_avg = cce_avg.merge(blend, on='streamer_id', how='left')
        cce_avg['CCE'] = 0.8*cce_avg['CCE'] + 0.2*cce_avg['estimated_CCE'].fillna(0)
    df = df.merge(cce_avg[['streamer_id','CCE']], on='streamer_id', how='left')
    df['CCE'] = df['CCE'].fillna(0.5)
    df['log_price'] = np.log(df['price'])
    df['inv_price'] = 1/df['price']
    df['price_sq'] = df['price']**2
    df['price_CCE'] = df['price'] * df['CCE']
    return df

@st.cache_resource
def train_xgboost_model(df):
    X = df[['price','log_price','inv_price','price_sq','price_CCE',
            'discount_rate','CCE','views','cost_per_unit','holiday_flag']]
    y = df['quantity_sold']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    with st.spinner("Training XGBoost model... (cached for future runs)"):
        model = XGBRegressor(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
    return model, X.columns.tolist()

@st.cache_data
def generate_predictions(_model, data, feature_columns):
    return _model.predict(data[feature_columns])

@st.cache_data
def calculate_summary_stats(filtered_df):
    tbl = filtered_df.groupby('product_id')[['price','quantity_sold','predicted_sales']].agg(['mean','sum']).reset_index()
    tbl.columns = ['product_id','price_avg','price_total','qty_avg','qty_total','pred_sales_avg','pred_sales_total']
    return tbl

@st.cache_data
def get_file_hash(uploaded_file):
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


def create_visualizations(df, filtered_df):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Streamer CCE Distribution")
        disp = df.sort_values('date').drop_duplicates('streamer_id',keep='last')
        disp['streamer_id_num'] = disp['streamer_id'].str.extract(r'(\d+)').astype(int)
        disp = disp.sort_values('streamer_id_num')
        fig = px.bar(disp, x='streamer_id', y='CCE', title="Average CCE by Influencer")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Sales vs Price")
        sample = filtered_df.sample(min(1000,len(filtered_df)))
        fig = px.scatter(sample, x='price', y='quantity_sold', color='CCE', size='views', title="Price vs Quantity Sold")
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.sidebar.header("Upload Historical Sales Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file to get started")
        return
    progress = st.progress(0)
    status = st.empty()
    try:
        status.text("Loading and preprocessing data...")
        progress.progress(20)
        df = load_and_preprocess_data(uploaded_file)
        status.text("Calculating CCE and features...")
        progress.progress(40)
        df = calculate_cce_and_features(df)
        status.text("Training/loading ML model...")
        progress.progress(60)
        model, feat_cols = train_xgboost_model(df)
        status.text("Setting up interface...")
        progress.progress(80)
        streamers = sorted(df['streamer_id'].unique(), key=lambda x: int(x[1:]))
        col1, col2 = st.columns([1,3])
        with col1:
            st.subheader("Select Influencer")
            sel = st.selectbox("Choose streamer:", streamers)
        with col2:
            st.subheader("Quick Stats")
            metrics = [len(streamers), df['product_id'].nunique(), df['quantity_sold'].sum(), df['price'].mean()]
            mcols = st.columns(4)
            mcols[0].metric("Streamers", metrics[0])
            mcols[1].metric("Products", metrics[1])
            mcols[2].metric("Total Sales", f"{metrics[2]:,.0f}")
            mcols[3].metric("Avg Price", f"${metrics[3]:.2f}")
        filtered = df[df['streamer_id']==sel].copy()
        status.text("Generating predictions...")
        progress.progress(90)
        filtered['predicted_sales'] = generate_predictions(model, filtered, feat_cols)
        progress.progress(100)
        status.empty()
        progress.empty()
        tabs = st.tabs(["Analysis","Predictions","Simulation","Raw Data"])
        with tabs[0]:
            st.subheader(f"Analysis for {sel}")
            summary = calculate_summary_stats(filtered)
            c1, c2 = st.columns([2,1])
            with c1:
                st.markdown(f"""
                - **Total Products:** {len(summary)}
                - **Total Sold:** {summary['qty_total'].sum():.0f}
                - **Predicted Sales:** {summary['pred_sales_total'].sum():.0f}
                - **Avg Price:** ${summary['price_avg'].mean():.2f}
                """
                )
                if summary['pred_sales_total'].sum() > summary['qty_total'].sum():
                    st.success("Sales momentum is improving!")
                else:
                    st.warning("Sales might need improvement.")
            with c2:
                st.subheader("Top Products")
                st.dataframe(filtered.nlargest(5,'quantity_sold')[['product_id','quantity_sold','price']],use_container_width=True)
            create_visualizations(df, filtered)
        with tabs[1]:
            st.subheader("Sales Predictions")
            st.dataframe(filtered[['product_id','price','CCE','quantity_sold','predicted_sales']],use_container_width=True)
        with tabs[2]:
            st.subheader("Simulate Future Scenarios")
            with st.form("sim_form"):
                cols = st.columns(3)
                cols[0].number_input("Price",1.0,1000.0,10.0,key='p')
                cols[0].number_input("Discount",0.0,1.0,0.1,key='d')
                cols[1].number_input("CCE",1.0, key='c')
                cols[1].number_input("Views",1000,100000,5000,key='v')
                cols[2].number_input("Cost",0.01,100.0,1.0,key='u')
                cols[2].selectbox("Holiday?",[0,1],key='h')
                sub = st.form_submit_button("Run Simulation")
            if sub:
                simdf = pd.DataFrame([{
                    'price':st.session_state.p,'log_price':np.log(st.session_state.p),'inv_price':1/st.session_state.p,
                    'price_sq':st.session_state.p**2,'price_CCE':st.session_state.p*st.session_state.c,
                    'discount_rate':st.session_state.d,'CCE':st.session_state.c,'views':st.session_state.v,
                    'cost_per_unit':st.session_state.u,'holiday_flag':st.session_state.h
                }])
                pred = model.predict(simdf[feat_cols])[0]
                st.success(f"Predicted Sales: {int(pred)} units")
                rev = pred*st.session_state.p*(1-st.session_state.d)
                cost = pred*st.session_state.u
                prof = rev-cost
                cols2 = st.columns(3)
                cols2[0].metric("Revenue",f"${rev:.2f}")
                cols2[1].metric("Cost",f"${cost:.2f}")
                cols2[2].metric("Profit",f"${prof:.2f}", delta=f"{(prof/cost*100):.1f}% ROI" if cost>0 else "N/A")
        with tabs[3]:
            st.subheader("Raw Data Snapshot")
            st.dataframe(filtered.head(50),use_container_width=True)
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.info("Please check your CSV format and try again.")

if __name__ == '__main__':
    main()
