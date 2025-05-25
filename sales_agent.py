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
    snack_df = pd.read_csv(uploaded_file)
    snack_df['date'] = pd.to_datetime(snack_df['date'])
    snack_df['streamer_id_num'] = snack_df['streamer_id'].str.extract(r'(\d+)').astype(int)
    snack_df = snack_df.sort_values(by=['streamer_id_num', 'date'])

    fields = ['streamer_id', 'product_id', 'date', 'price', 'discount_rate', 'views',
              'cost_per_unit', 'shelf_life_days', 'holiday_flag', 'quantity_sold', 'CCE']
    snack_df = snack_df[fields]
    snack_df = snack_df.replace([np.inf, -np.inf], np.nan).dropna()
    snack_df = snack_df[snack_df['quantity_sold'] > 0]
    
    return snack_df

@st.cache_data
def calculate_cce_and_features(_snack_df):
    snack_df = _snack_df.copy()
    
    base_model = LinearRegression().fit(np.log(snack_df[['price']]), np.log(snack_df['quantity_sold']))
    snack_df['baseline_logq'] = base_model.predict(np.log(snack_df[['price']]))
    snack_df['uplift'] = np.log(snack_df['quantity_sold']) - snack_df['baseline_logq']

    cce_avg_table = snack_df.groupby('streamer_id')['uplift'].mean().reset_index()
    cce_avg_table = cce_avg_table.rename(columns={'uplift': 'CCE'})

    if 'views' in snack_df.columns and 'conversion_rate' in snack_df.columns:
        snack_df['estimated_CCE'] = (snack_df['quantity_sold'] / snack_df['baseline_logq'].apply(np.exp)) \
            * (snack_df['conversion_rate'] / snack_df['conversion_rate'].mean()) \
            * np.log(snack_df['views'] + 1)

        blend_df = snack_df.groupby('streamer_id')['estimated_CCE'].mean().reset_index()
        cce_avg_table = cce_avg_table.merge(blend_df, on='streamer_id', how='left')
        cce_avg_table['CCE'] = 0.8 * cce_avg_table['CCE'] + 0.2 * cce_avg_table['estimated_CCE'].fillna(0)
        cce_avg_table = cce_avg_table[['streamer_id', 'CCE']]
        snack_df = snack_df.merge(cce_avg_table, on='streamer_id', how='left')

        snack_df['CCE'] = snack_df['CCE'].fillna(0.5)

    snack_df['log_price'] = np.log(snack_df['price'])
    snack_df['inv_price'] = 1 / snack_df['price']
    snack_df['price_sq'] = snack_df['price'] ** 2
    snack_df['price_CCE'] = snack_df['price'] * snack_df['CCE']
    
    return snack_df

@st.cache_resource
def train_xgboost_model(_snack_df):
    X = _snack_df[['price', 'log_price', 'inv_price', 'price_sq', 'price_CCE', 
                   'discount_rate', 'CCE', 'views', 'cost_per_unit', 'holiday_flag']]
    y = _snack_df['quantity_sold']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with st.spinner("Training XGBoost model... (This will be cached for future runs)"):
        xgb_model = XGBRegressor(random_state=42, n_estimators=100)
        xgb_model.fit(X_train, y_train)
    
    return xgb_model, X.columns.tolist()

@st.cache_data
def generate_predictions(_model, _data, feature_columns):
    return _model.predict(_data[feature_columns])

@st.cache_data
def calculate_summary_stats(_filtered_df):
    stream_summary = _filtered_df.groupby('product_id')[['price', 'quantity_sold', 'predicted_sales']].agg(['mean', 'sum']).reset_index()
    stream_summary.columns = ['product_id', 'price_avg', 'price_total', 'qty_avg', 'qty_total', 'pred_sales_avg', 'pred_sales_total']
    return stream_summary

@st.cache_data
def get_file_hash(uploaded_file):
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()

def create_visualizations(snack_df, filtered_df):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Streamer CCE Distribution")
        cce_display = snack_df.sort_values(by='date').drop_duplicates(subset='streamer_id', keep='last')
        cce_display = cce_display[['streamer_id', 'CCE']].copy()
        cce_display['streamer_id_num'] = cce_display['streamer_id'].str.extract(r'(\d+)').astype(int)
        cce_display = cce_display.sort_values(by='streamer_id_num')
        fig_cce = px.bar(cce_display, x="streamer_id", y="CCE", title="Average CCE by Influencer")
        st.plotly_chart(fig_cce, use_container_width=True)
    
    with col2:
        st.subheader("Sales vs Price")
        fig_price = px.scatter(
            filtered_df.sample(min(1000, len(filtered_df))),
            x="price", y="quantity_sold", color="CCE",
            size="views", title="Price vs Quantity Sold"
        )
        st.plotly_chart(fig_price, use_container_width=True)

def main():
    st.title("AI Agent for Post-Sale Strategy Analysis")
    st.sidebar.header("Upload Historical Sales Data")

    uploaded_file = st.sidebar.file_uploader("Upload your historical sales CSV", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a CSV file to get started")
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Loading and preprocessing data...")
        progress_bar.progress(20)
        snack_df = load_and_preprocess_data(uploaded_file)
        
        status_text.text("Calculating CCE and features...")
        progress_bar.progress(40)
        snack_df = calculate_cce_and_features(snack_df)
        
        status_text.text("Training/Loading ML model...")
        progress_bar.progress(60)
        xgb_model, feature_columns = train_xgboost_model(snack_df)
        
        status_text.text("Setting up interface...")
        progress_bar.progress(80)
        
        sorted_streamers = sorted(snack_df['streamer_id'].unique(), key=lambda x: int(x[1:]))
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("Select Influencer")
            selected_streamer = st.selectbox("Choose streamer:", sorted_streamers, key="streamer_select")
        
        with col2:
            st.subheader("Quick Stats")
            total_streamers = len(sorted_streamers)
            total_products = snack_df['product_id'].nunique()
            total_sales = snack_df['quantity_sold'].sum()
            avg_price = snack_df['price'].mean()
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            metrics_col1.metric("Streamers", total_streamers)
            metrics_col2.metric("Products", total_products)
            metrics_col3.metric("Total Sales", f"{total_sales:,.0f}")
            metrics_col4.metric("Avg Price", f"${avg_price:.2f}")
        
        filtered_df = snack_df[snack_df['streamer_id'] == selected_streamer].copy()
        
        status_text.text("Generating predictions...")
        progress_bar.progress(90)
        filtered_df['predicted_sales'] = generate_predictions(xgb_model, filtered_df, feature_columns)
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        tab1, tab2, tab3, tab4 = st.tabs(["Analysis", "Predictions", "Simulation", "Raw Data"])
        
        with tab1:
            st.subheader(f"Analysis for {selected_streamer}")
            
            stream_summary = calculate_summary_stats(filtered_df)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"""
                **Performance Summary:**
                - Total Products Promoted: {len(stream_summary)}
                - Total Quantity Sold: {stream_summary['qty_total'].sum():.0f}
                - Total Predicted Sales: {stream_summary['pred_sales_total'].sum():.0f}
                - Average Price: ${stream_summary['price_avg'].mean():.2f}
                """)
                
                if stream_summary['pred_sales_total'].sum() > stream_summary['qty_total'].sum():
                    st.success("Sales momentum is improving!")
                else:
                    st.warning("Sales might need improvement")
            
            with col2:
                st.subheader("Top Products")
                top_products = filtered_df.nlargest(5, 'quantity_sold')[['product_id', 'quantity_sold', 'price']]
                st.dataframe(top_products, use_container_width=True)
            
            create_visualizations(snack_df, filtered_df)
        
        with tab2:
            st.subheader("Sales Predictions")
            
            prediction_df = filtered_df[['product_id', 'price', 'CCE', 'quantity_sold', 'predicted_sales']].round(2)
            st.dataframe(prediction_df, use_container_width=True)
        
        with tab3:
            st.subheader("Simulate Future Scenarios")
            
            with st.form("simulate_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sim_streamer = st.selectbox("Streamer", sorted_streamers)
                    price_input = st.number_input("Price", min_value=1.0, value=10.0, step=0.5)
                    discount_input = st.number_input("Discount Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
                
                with col2:
                    cce_input = st.number_input("CCE", value=1.0, step=0.1)
                    views_input = st.number_input("Expected Views", min_value=1000, value=5000, step=500)
                    cost_input = st.number_input("Cost per Unit", min_value=0.01, value=1.0, step=0.01)
                
                with col3:
                    holiday_input = st.selectbox("Holiday?", options=[0, 1])
                    st.write("")
                    submitted = st.form_submit_button("Run Simulation", use_container_width=True)
            
            if submitted:
                sim_df = pd.DataFrame([{
                    'price': price_input,
                    'log_price': np.log(price_input),
                    'inv_price': 1 / price_input,
                    'price_sq': price_input ** 2,
                    'price_CCE': price_input * cce_input,
                    'discount_rate': discount_input,
                    'CCE': cce_input,
                    'views': views_input,
                    'cost_per_unit': cost_input,
                    'holiday_flag': holiday_input
                }])
                
                prediction = xgb_model.predict(sim_df[feature_columns])[0]
                
                st.success(f"Predicted Sales: **{int(prediction)} units**")
                
                revenue = prediction * price_input * (1 - discount_input)
                cost = prediction * cost_input
                profit = revenue - cost
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Revenue", f"${revenue:.2f}")
                col2.metric("Cost", f"${cost:.2f}")
                col3.metric("Profit", f"${profit:.2f}", delta=f"{(profit/cost*100):.1f}% ROI" if cost > 0 else "N/A")
        
        with tab4:
            st.subheader("Raw Data")
            st.dataframe(filtered_df.head(50), use_container_width=True)
            
            if st.button("Download Filtered Data"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{selected_streamer}_data.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.info("Please check your CSV file format and try again.")

if __name__ == "__main__":
    main()
