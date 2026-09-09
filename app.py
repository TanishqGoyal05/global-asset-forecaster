import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Commodity Trend & Procurement Forecaster",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. Data Fetching Engine
# ==========================================
@st.cache_data(show_spinner=False)
def load_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    """Fetches historical OHLCV data and caches it for performance."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except Exception:
        return pd.DataFrame()

def get_usd_inr_rate():
    """Fetches live USD/INR exchange rate."""
    try:
        ticker = yf.Ticker("INR=X")
        hist = ticker.history(period="2d")
        if len(hist) >= 2:
            current = hist['Close'].iloc[-1]
            previous = hist['Close'].iloc[-2]
            return current, (current - previous)
    except Exception:
        pass
    return None, None

# ==========================================
# 3. UI Layout: Header & Live KPI
# ==========================================
st.title("📊 Strategic Procurement: Global Commodity Forecaster")
st.markdown("---")

current_rate, daily_change = get_usd_inr_rate()
if current_rate is not None:
    st.metric(
        label="Live USD/INR Exchange Rate (INR=X)", 
        value=f"₹{current_rate:.4f}", 
        delta=f"{daily_change:.4f}"
    )
    
# ==========================================
# 4. UI Layout: Sidebar Parameters
# ==========================================
st.sidebar.title("Engine Configuration")

commodity_map = {
    "Copper": "HG=F", "Crude Oil": "CL=F", "Aluminum": "ALI=F",
    "Natural Gas": "NG=F", "Gold": "GC=F", "Silver": "SI=F"
}

selected_commodity = st.sidebar.selectbox("Select Asset", options=list(commodity_map.keys()))
ticker = commodity_map[selected_commodity]

today = datetime.today().date()
start_date = st.sidebar.date_input("Start Date", value=today - timedelta(days=5*365))
end_date = st.sidebar.date_input("End Date", value=today)

if start_date > end_date:
    st.sidebar.error("Sequence Error: 'Start Date' must precede 'End Date'.")
    st.stop()

# ==========================================
# 5. Data Execution & Rendering
# ==========================================
df_commodity = load_data(ticker, start_date, end_date)

if df_commodity.empty:
    st.error(f"Data stream unavailable for {ticker}. Please adjust your date range.")
else:
    st.markdown(f"### {selected_commodity} Market Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Recent Pricing Data (Tail)**")
        st.dataframe(df_commodity.tail(5), use_container_width=True)
        
    with col2:
        st.write("**Statistical Summary**")
        if 'Close' in df_commodity.columns:
            stats_df = df_commodity['Close'].describe().to_frame(name="Value")
            st.dataframe(stats_df, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # 6. Phase 2: Technical Shape Analysis
    # ==========================================
    st.subheader("Technical Price Analysis & Trend Shapes")
    
    if all(col in df_commodity.columns for col in ['Open', 'High', 'Low', 'Close']):
        # Calculate Moving Averages
        df_commodity['SMA_50'] = df_commodity['Close'].rolling(window=50).mean()
        df_commodity['SMA_200'] = df_commodity['Close'].rolling(window=200).mean()
        
        # Build the Interactive Chart
        fig = go.Figure()
        
        # Candlestick Trace
        fig.add_trace(go.Candlestick(
            x=df_commodity.index,
            open=df_commodity['Open'], high=df_commodity['High'],
            low=df_commodity['Low'], close=df_commodity['Close'],
            name='Price'
        ))
        
        # 50-Day SMA Trace (Blue)
        fig.add_trace(go.Scatter(
            x=df_commodity.index, y=df_commodity['SMA_50'],
            mode='lines', line=dict(color='#2196F3', width=1.5), name='50-Day SMA'
        ))
        
        # 200-Day SMA Trace (Red)
        fig.add_trace(go.Scatter(
            x=df_commodity.index, y=df_commodity['SMA_200'],
            mode='lines', line=dict(color='#F44336', width=1.5), name='200-Day SMA'
        ))
        
        # Style the layout
        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=True,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Render in Streamlit
        st.plotly_chart(fig, use_container_width=True)


        # ==========================================
    # 7. Phase 3: Algorithmic Price Forecasting
    # ==========================================
    st.markdown("---")
    st.subheader("Predictive Price Modeling")
    
    # Import the statistical modeling library
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    # Sidebar control for the forecast window
    st.sidebar.markdown("### Forecasting Engine")
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30)
    
    # Ensure we have clean data without missing values
    ts_data = df_commodity['Close'].dropna()
    
    if len(ts_data) > forecast_horizon * 2:
        with st.spinner('Calculating algorithmic forecast...'):
            # Fit the Exponential Smoothing model (Additive trend)
            model = ExponentialSmoothing(ts_data.values, trend='add', seasonal=None, initialization_method="estimated")
            fit_model = model.fit()
            
            # Generate future predictions
            forecast = fit_model.forecast(forecast_horizon)
            
            # Generate future dates for the X-axis
            last_date = ts_data.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_horizon + 1)]
            
            # Build the forecast chart
            fig_forecast = go.Figure()
            
            # Plot the last 6 months of historical data for context
            historical_slice = ts_data.tail(180)
            fig_forecast.add_trace(go.Scatter(
                x=historical_slice.index, y=historical_slice, 
                mode='lines', name='Historical Close', line=dict(color='#E0E0E0', width=2)
            ))
            
            # Plot the forecasted data
            fig_forecast.add_trace(go.Scatter(
                x=future_dates, y=forecast, 
                mode='lines', name='Projected Trend', line=dict(color='#FFC107', width=2, dash='dash')
            ))
            
            fig_forecast.update_layout(
                template="plotly_dark",
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Display Forecast KPIs
            # Display Forecast KPIs
            proj_price = forecast[-1]  # Removed .iloc since forecast is now a raw array
            current_price = ts_data.iloc[-1] # ts_data is still a Pandas Series, so .iloc stays
            proj_change = ((proj_price - current_price) / current_price) * 100
            
            st.metric(
                label=f"Projected Price in {forecast_horizon} Days", 
                value=f"{proj_price:.4f}", 
                delta=f"{proj_change:.2f}%"
            )
    else:
        st.warning("Insufficient historical data to generate a reliable forecast.")