# 📊 AI Agent for Post-Sale Strategy Analysis

This Streamlit-based app helps TikTok livestream influencers and marketing teams analyze sales performance, model influencer CCE (Celebrity Coefficient Effectiveness), and forecast future results based on pricing and viewership plans.

## 🔧 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sales_agent.git
   cd sales_agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the app:
   ```bash
   streamlit run app.py
   ```

4. Upload your CSV sales data (see `sample_data.csv` for reference).

## 🧪 Features

- Dynamic CCE trend tracking
- XGBoost-based sales prediction
- Custom simulation for future campaigns
- Visual summaries for influencer team

## 📁 Sample Data Format

See `sample_data.csv` for a format example with columns:
- `streamer_id`, `product_id`, `date`, `price`, `discount_rate`, `views`, `conversion_rate`, `cost_per_unit`, `shelf_life_days`, `holiday_flag`, `quantity_sold`
