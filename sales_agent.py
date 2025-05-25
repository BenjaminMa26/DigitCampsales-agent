import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import plotly.express as px

st.title("🎯 AI Agent for Post-Sale Strategy Analysis")
st.sidebar.header("📥 Upload Historical Sales Data")

uploaded_file = st.sidebar.file_uploader("Upload your historical sales CSV", type=["csv"])

if uploaded_file:
    snack_df = pd.read_csv(uploaded_file)
    snack_df['date'] = pd.to_datetime(snack_df['date'])
    snack_df['streamer_id_num'] = snack_df['streamer_id'].str.extract(r'(\d+)').astype(int)
    snack_df = snack_df.sort_values(by=['streamer_id_num', 'date'])

    fields = ['streamer_id', 'product_id', 'date', 'price', 'discount_rate', 'views',
              'cost_per_unit', 'shelf_life_days', 'holiday_flag', 'quantity_sold', 'CCE']
    snack_df = snack_df[fields]
    snack_df = snack_df.replace([np.inf, -np.inf], np.nan).dropna()
    snack_df = snack_df[snack_df['quantity_sold'] > 0]

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
        missing_cce_streamers = snack_df[snack_df['CCE'] == 0.5]['streamer_id'].unique()
        if len(missing_cce_streamers) > 0:
            st.warning(f"🔍 CCE defaulted to 0.5 for missing streamers: {', '.join(missing_cce_streamers)}")

    snack_df['log_price'] = np.log(snack_df['price'])
    snack_df['inv_price'] = 1 / snack_df['price']
    snack_df['price_sq'] = snack_df['price'] ** 2
    snack_df['price_CCE'] = snack_df['price'] * snack_df['CCE']

    sorted_streamers = sorted(snack_df['streamer_id'].unique(), key=lambda x: int(x[1:]))
    st.subheader("🔍 Select Influencer")
    selected_streamer = st.selectbox("Select a streamer to analyze:", sorted_streamers)
    filtered_df = snack_df[snack_df['streamer_id'] == selected_streamer]

    st.subheader(" Predict Sales Using XGBoost")
    X = snack_df[['price', 'log_price', 'inv_price', 'price_sq', 'price_CCE', 'discount_rate', 'CCE', 'views', 'cost_per_unit', 'holiday_flag']]
    y = snack_df['quantity_sold']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_model = XGBRegressor().fit(X_train, y_train)

    filtered_df['predicted_sales'] = xgb_model.predict(filtered_df[X.columns])
    st.success(" XGBoost model trained on post-sale data")
    st.dataframe(filtered_df[['streamer_id', 'price', 'CCE', 'quantity_sold', 'predicted_sales']].head(15))

    #报告生成
    st.subheader(f" Automated Summary for {selected_streamer}")

    stream_summary = filtered_df.groupby('product_id')[['price', 'quantity_sold', 'predicted_sales']].agg(['mean', 'sum']).reset_index()
    stream_summary.columns = ['product_id', 'price_avg', 'price_total', 'qty_avg', 'qty_total', 'pred_sales_avg', 'pred_sales_total']

    st.markdown(f"""
    - **Total Products Promoted:** {len(stream_summary)}
    - **Total Quantity Sold:** {stream_summary['qty_total'].sum():.0f}
    - **Total Predicted Sales:** {stream_summary['pred_sales_total'].sum():.0f}
    - **Average Price:** ${stream_summary['price_avg'].mean():.2f}
    """)

    if stream_summary['pred_sales_total'].sum() > stream_summary['qty_total'].sum():
        st.info("📈 Sales momentum is improving – predicted performance exceeds historical average.")
    else:
        st.warning("📉 Sales might need improvement – consider revisiting price or influencer match.")

    # ⬇️ Add influencer summary interpretation
    st.subheader(" Summary Report for Influencer Team")
    summary_table = snack_df.groupby('streamer_id')[['CCE', 'quantity_sold']].agg(['mean', 'count']).reset_index()
    summary_table.columns = ['streamer_id', 'CCE_mean', 'CCE_count', 'quantity_mean', 'quantity_count']
    st.dataframe(summary_table.sort_values(by='CCE_mean', ascending=False))

    top_influencer = summary_table.sort_values(by='CCE_mean', ascending=False).iloc[0]
    st.markdown(f"🎖️ **Top Performer: {top_influencer['streamer_id']}**\n\n"
                f"- Average CCE: {top_influencer['CCE_mean']:.2f}\n"
                f"- Average Quantity Sold: {top_influencer['quantity_mean']:.0f}\n"
                f"- Sessions: {int(top_influencer['CCE_count'])}")

    # 📊 可视化模块
    st.subheader(" Streamer CCE Distribution")
    cce_display = snack_df.sort_values(by='date').drop_duplicates(subset='streamer_id', keep='last')
    cce_display = cce_display[['streamer_id', 'CCE']].copy()
    cce_display['streamer_id_num'] = cce_display['streamer_id'].str.extract(r'(\d+)').astype(int)
    cce_display = cce_display.sort_values(by='streamer_id_num')
    fig_cce = px.bar(cce_display, x="streamer_id", y="CCE", title="Average CCE by Influencer (Latest Session)")
    st.plotly_chart(fig_cce, use_container_width=True)

    st.subheader(" Streamer CCE Trends Over Time")
    cce_trend = snack_df.groupby(['date', 'streamer_id'])['CCE'].mean().reset_index()
    cce_trend['streamer_id_num'] = cce_trend['streamer_id'].str.extract(r'(\d+)').astype(int)
    cce_trend = cce_trend.sort_values(by='streamer_id_num')

    # 显示实际CCE趋势并加回归线
    fig_trend_actual = px.scatter(cce_trend, x="date", y="CCE", color="streamer_id", trendline="ols", title="CCE Trend by Influencer with Trendline")
    st.plotly_chart(fig_trend_actual, use_container_width=True)

    st.subheader(" Sales vs Price (Log-Scale)")
    fig_price = px.scatter(
        snack_df, x="price", y="quantity_sold", color="streamer_id",
        size="CCE", log_x=True, title="Price vs Quantity Sold (by Influencer)"
    )
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader(" Next Session Sales Forecast")
    forecast_summary = filtered_df[['product_id', 'price', 'CCE', 'predicted_sales']].groupby(['product_id', 'price', 'CCE']).sum().reset_index()
    st.dataframe(forecast_summary.sort_values(by='predicted_sales', ascending=False).head(10))

    st.subheader(" Simulate Future Scenarios")
    with st.form("simulate_form"):
        st.markdown("**Input your next session plan to simulate outcome**")
        sim_streamer = st.selectbox("Select Streamer", sorted(snack_df['streamer_id'].unique(), key=lambda x: int(x[1:])))
        price_input = st.number_input("Simulated Price", min_value=1.0, value=10.0, step=0.5)
        discount_input = st.number_input("Simulated Discount Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        cce_input = st.text_input("Influencer CCE", value="1.0")
        views_input = st.number_input("Expected Views", min_value=1000, value=5000, step=500)
        cost_input = st.number_input("Cost per Unit", min_value=0.01, value=1.0, step=0.01)
        holiday_input = st.selectbox("Is it a holiday?", options=[0, 1])
        submitted = st.form_submit_button("Run Simulation")

    if submitted:
        try:
            cce_input_val = float(cce_input)
            sim_df = pd.DataFrame([{
                'price': price_input,
                'log_price': np.log(price_input),
                'inv_price': 1 / price_input,
                'price_sq': price_input ** 2,
                'price_CCE': price_input * cce_input_val,
                'discount_rate': discount_input,
                'CCE': cce_input_val,
                'views': views_input,
                'cost_per_unit': cost_input,
                'holiday_flag': holiday_input
            }])
            prediction = xgb_model.predict(sim_df)[0]
            st.success(f" Predicted Sales for {sim_streamer}'s Plan: {int(prediction)} units")
        except ValueError:
            st.error(" Invalid CCE input. Please enter a numeric value.")

    st.subheader(" Raw Data Snapshot")
    st.dataframe(filtered_df.head(20))
    


else:
    st.stop()
