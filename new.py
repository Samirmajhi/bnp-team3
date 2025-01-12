import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Predefined username and password
USERNAME = "admin"
PASSWORD = "admin123"

# Initialize session state for login status and navigation
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "login"

# Function to handle login
def login(username, password):
    if username == USERNAME and password == PASSWORD:
        st.session_state["logged_in"] = True
        st.session_state["current_page"] = "welcome"
    else:
        st.error("Incorrect username or password")

# Function to compute returns and volatility
def compute_returns_volatility(top_n, risk_free_rate):
    try:
        fundamentals = pd.read_csv("fundamentals.csv")
        prices = pd.read_csv("prices-split-adjusted.csv")

        fundamentals["Net Income"] = fundamentals["Net Income"].fillna(0)
        top_companies = fundamentals.groupby("Ticker Symbol")["Net Income"].sum().nlargest(top_n).index.tolist()

        filtered_prices = prices[prices["symbol"].isin(top_companies)].copy()
        filtered_prices["daily_return"] = filtered_prices.groupby("symbol")["close"].pct_change()

        volatility = filtered_prices.groupby("symbol")["daily_return"].std() * np.sqrt(252)
        mean_daily_returns = filtered_prices.groupby("symbol")["daily_return"].mean()
        annualized_returns = mean_daily_returns * 252

        filtered_prices["daily_return"] = filtered_prices["daily_return"].dropna()
        skewness = filtered_prices.groupby("symbol")["daily_return"].apply(lambda x: skew(x.dropna()))
        kurt = filtered_prices.groupby("symbol")["daily_return"].apply(lambda x: kurtosis(x.dropna()))

        asr = (
            ((annualized_returns - risk_free_rate) / volatility)
            * (1 + (((skewness * 6) + (kurt * 24) - 3) / 24))
        )

        downside_returns = filtered_prices[filtered_prices["daily_return"] < 0].groupby("symbol")["daily_return"]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_returns - risk_free_rate) / downside_std

        omega_ratio = filtered_prices.groupby("symbol")["daily_return"].apply(
            lambda x: (x[x > risk_free_rate].sum()) / abs(x[x < risk_free_rate].sum())
        )

        results = pd.DataFrame(
            {
                "Annualized Return": annualized_returns,
                "Annualized Volatility": volatility,
                "Adjusted Sharpe Ratio": asr,
                "Sortino Ratio": sortino_ratio,
                "Omega Ratio": omega_ratio,
            }
        ).sort_values(by="Annualized Return", ascending=False)

        return results

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Visualize Dummy Stock Performance Page
def visualize_stock_performance_page():
    prices_df = pd.read_csv("prices-split-adjusted.csv", parse_dates=["date"])
    np.random.seed(42)  
    selected_stocks = np.random.choice(prices_df["symbol"].unique(), 10, replace=False)

    portfolio_df = prices_df[prices_df["symbol"].isin(selected_stocks)].copy()
    portfolio_df.sort_values(by=["symbol", "date"], inplace=True)
    portfolio_df["daily_return"] = portfolio_df.groupby("symbol")["close"].pct_change()

    weights = {stock: 1 / len(selected_stocks) for stock in selected_stocks}
    portfolio_df["weighted_return"] = portfolio_df["symbol"].map(weights) * portfolio_df["daily_return"]
    portfolio_returns = portfolio_df.groupby("date")["weighted_return"].sum()

    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
    portfolio_index = 100 * portfolio_cumulative_returns / portfolio_cumulative_returns.iloc[0]

    trading_days = 252
    annual_return = (1 + portfolio_returns.mean())**trading_days - 1
    annual_volatility = portfolio_returns.std() * np.sqrt(trading_days)

    return_skewness = skew(portfolio_returns.dropna())
    return_kurtosis = kurtosis(portfolio_returns.dropna())

    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(trading_days)

    risk_free_rate = 0.02
    ASR = ((annual_return - risk_free_rate) / annual_volatility) * (1 + (6 * return_skewness) + ((24 * return_kurtosis - 3) / 24))
    sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
    returns_above_threshold = portfolio_returns[portfolio_returns > risk_free_rate]
    returns_below_threshold = portfolio_returns[portfolio_returns < risk_free_rate]
    omega_ratio = returns_above_threshold.sum() / abs(returns_below_threshold.sum())

    portfolio_summary = pd.DataFrame({
        "Stock Symbol": selected_stocks,
        "Equal Weight (%)": [weights[s] * 100 for s in selected_stocks]
    })

    st.subheader("Selected Stocks for Dummy Portfolio")
    st.dataframe(portfolio_summary)

    portfolio_metrics = pd.DataFrame({
        "Metric": ["Annualized Return", "Annualized Volatility", "Adjusted Sharpe Ratio (ASR)", "Sortino Ratio", "Omega Ratio"],
        "Value": [f"{annual_return:.2%}", f"{annual_volatility:.2%}", f"{ASR:.2f}", f"{sortino_ratio:.2f}", f"{omega_ratio:.2f}"]
    })

    st.subheader("Portfolio Performance Metrics")
    st.dataframe(portfolio_metrics)

    st.subheader("Portfolio Performance Over Time")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=portfolio_index.index, y=portfolio_index, label="Dummy Portfolio", color="b")
    plt.axhline(y=100, color="gray", linestyle="--", label="Starting Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Index (Base = 100)")
    plt.title("Portfolio Performance Over Time")
    plt.legend()
    plt.grid(True)
    st.pyplot()

    if st.sidebar.button("Back to Dashboard"):
        st.session_state["current_page"] = "welcome"

# Pages
def login_page():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        login(username, password)


def welcome_page():
    st.markdown("<h1 style='text-align: center;'>Welcome Admin</h1>", unsafe_allow_html=True)
    # HTML and CSS for the container box
    box_style = """
<div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; background-color: #f9f9f9; margin-top: 20px;">
    <h3 style="text-align: center; color: #4CAF50;">Stock Information Dashboard</h3>
    <div style="display: flex; justify-content: space-between; gap: 20px;">
        <div style="flex: 1; padding: 15px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); color: #333;">
            <h4 style="color: #4CAF50;">Stock: AAPL</h4>
            <p><strong>Current Price:</strong> $175.54</p>
            <p><strong>Change:</strong> +1.2%</p>
            <p><strong>Market Cap:</strong> $2.8 Trillion</p>
            <p><strong>P/E Ratio:</strong> 25.5</p>
        </div>
        <div style="flex: 1; padding: 15px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); color: #333;">
            <h4 style="color: #4CAF50;">Stock: MSFT</h4>
            <p><strong>Current Price:</strong> $342.16</p>
            <p><strong>Change:</strong> -0.5%</p>
            <p><strong>Market Cap:</strong> $2.56 Trillion</p>
            <p><strong>P/E Ratio:</strong> 30.3</p>
        </div>
    </div>
    <div style="display: flex; justify-content: space-between; gap: 20px; margin-top: 20px;">
        <div style="flex: 1; padding: 15px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); color: #333;">
            <h4 style="color: #4CAF50;">Stock: GOOGL</h4>
            <p><strong>Current Price:</strong> $138.69</p>
            <p><strong>Change:</strong> +0.8%</p>
            <p><strong>Market Cap:</strong> $1.87 Trillion</p>
            <p><strong>P/E Ratio:</strong> 28.7</p>
        </div>
        <div style="flex: 1; padding: 15px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); color: #333;">
            <h4 style="color: #4CAF50;">Stock: AMZN</h4>
            <p><strong>Current Price:</strong> $125.94</p>
            <p><strong>Change:</strong> +0.3%</p>
            <p><strong>Market Cap:</strong> $1.26 Trillion</p>
            <p><strong>P/E Ratio:</strong> 70.1</p>
        </div>  
    </div>
    <br>
    <h3 style="text-align: center; color: #4CAF50;">Upcoming IPO's</h3>
    <div style="flex: 1; padding: 15px; background-color: #ffffff;text-align:center; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); color: #333;">
    <h4 style="color: #4CAF50;">IPO: EMA Partners India Ltd</h4>
    <p><strong>IPO Date:</strong> 2025-01-17</p>
    <p><strong>Expected Price:</strong> $50 - $60</p>
</div>
<div style="flex: 1; padding: 15px; background-color: #ffffff;text-align:center; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); color: #333;">
    <h4 style="color: #4CAF50;">IPO: Laxmi Dental Ltd</h4>
    <p><strong>IPO Date:</strong> 2025-01-13</p>
    <p><strong>Expected Price:</strong> $30 - $40</p>
</div>
<div style="flex: 1; padding: 15px; background-color: #ffffff;text-align:center; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); color: #333;">
    <h4 style="color: #4CAF50;">IPO: Landmark Immigration Consultants Ltd</h4>
    <p><strong>IPO Date:</strong> 2025-01-16</p>
    <p><strong>Expected Price:</strong> $70 - $90</p>
</div>


</div>
"""

    
    # Render the HTML box with stock information
    st.markdown(box_style, unsafe_allow_html=True)

   

    # Move buttons to the sidebar
    st.sidebar.markdown("<h1 style='text-decoration: underline;'>Dashboard</h2>", unsafe_allow_html=True)
    if st.sidebar.button("Top Companies"):
        st.session_state["current_page"] = "compute"

    if st.sidebar.button("Visualize Dummy Stock Performance"):
        st.session_state["current_page"] = "visualize"

    if st.sidebar.button("Heatmap of Top 50 Most Traded Stocks"):
        st.session_state["current_page"] = "heatmap"

    if st.sidebar.button("P/E & P/B Ratio Analysis"):
        st.session_state["current_page"] = "pe_pb_ratio"
    
    if st.sidebar.button("Compare Stock Price Movement"):
        st.session_state["current_page"] = "compare"

    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["current_page"] = "login"
# Enhanced Stock Comparison Page
def stock_comparison_page():
    st.title("Interactive Stock Price Comparison Dashboard")

    # Load stock prices data
    prices = pd.read_csv("prices-split-adjusted.csv", parse_dates=["date"])

    # User inputs for stock selection
    symbols = prices["symbol"].unique()
    stock1 = st.selectbox("Select the first stock:", symbols)
    stock2 = st.selectbox("Select the second stock:", symbols)

    # User inputs for date range
    min_date, max_date = prices["date"].min(), prices["date"].max()
    date_range = st.date_input("Select the date range:", [min_date, max_date])

    # Ensure valid date range
    if len(date_range) != 2:
        st.error("Please select a valid start and end date.")
        return
    start_date, end_date = date_range

    # Filter data for the selected stocks and date range
    filtered_data = prices[
        (prices["symbol"].isin([stock1, stock2])) &
        (prices["date"] >= pd.Timestamp(start_date)) &
        (prices["date"] <= pd.Timestamp(end_date))
    ]

    # Check if there is data for the given selection
    if filtered_data.empty:
        st.warning("No data available for the selected stocks or date range.")
        return

    # User choice for viewing granularity (day, month, year)
    granularity = st.radio("Select Granularity:", ["Daily", "Monthly", "Yearly"], index=0)

    # Resample data based on granularity
    def resample_data(data, freq):
        return data.resample(freq, on="date").agg({
            "close": "mean",
            "volume": "sum"
        }).reset_index()

    freq_map = {"Daily": "D", "Monthly": "M", "Yearly": "Y"}
    stock1_data = resample_data(filtered_data[filtered_data["symbol"] == stock1], freq_map[granularity])
    stock2_data = resample_data(filtered_data[filtered_data["symbol"] == stock2], freq_map[granularity])

    # Interactive Plot with Plotly
    fig = go.Figure()

    # Add Stock 1
    fig.add_trace(go.Scatter(
        x=stock1_data["date"],
        y=stock1_data["close"],
        mode="lines",
        name=f"{stock1} Close Price",
        line=dict(color="blue")
    ))

    # Add Stock 2
    fig.add_trace(go.Scatter(
        x=stock2_data["date"],
        y=stock2_data["close"],
        mode="lines",
        name=f"{stock2} Close Price",
        line=dict(color="orange")
    ))

    # Customize layout
    fig.update_layout(
        title=f"Stock Price Movement: {stock1} vs {stock2} ({granularity})",
        xaxis_title="Date",
        yaxis_title="Close Price",
        legend_title="Stocks",
        hovermode="x unified",
        template="plotly_white"
    )

    # Display Plotly chart
    st.plotly_chart(fig, use_container_width=True)

    # Back to Dashboard button
    if st.button("Back to Dashboard"):
        st.session_state["current_page"] = "welcome"

def pe_pb_ratio_page():
    # Load datasets
    fundamentals1 = pd.read_csv("fundamentals1.csv")
    prices = pd.read_csv("prices-split-adjusted.csv")

    # Merge datasets on Ticker Symbol / symbol
    df = fundamentals1.merge(prices, left_on='Ticker Symbol', right_on='symbol')

    # Calculate Price-to-Earnings (P/E) and Price-to-Book (P/B) ratios
    df['P/E'] = df['close'] / df['Net Income']
    df['P/B'] = df['close'] / df['Total Equity']

    # Sort data by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['Ticker Symbol', 'date'])

    # Apply Exponential Moving Average (EMA) for smoothing
    df['P/E_EMA'] = df.groupby('Ticker Symbol')['P/E'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
    df['P/B_EMA'] = df.groupby('Ticker Symbol')['P/B'].transform(lambda x: x.ewm(span=10, adjust=False).mean())

    # Anomaly detection (flagging extreme P/E values)
    df['PE_Anomaly'] = np.abs(df['P/E'] - df['P/E_EMA']) > 2 * df['P/E'].std()

    # Function to plot P/E and P/B ratios over time
    def plot_ratios(ticker):
        data = df[df['Ticker Symbol'] == ticker]
        plt.figure(figsize=(12, 6))
        plt.plot(data['date'], data['P/E_EMA'], label='P/E (EMA)', color='blue')
        plt.plot(data['date'], data['P/B_EMA'], label='P/B (EMA)', color='red')
        plt.xlabel('Date')
        plt.ylabel('Ratio')
        plt.title(f'{ticker} - P/E & P/B Ratios Over Time')
        plt.legend()
        st.pyplot()

    # Function to forecast P/E ratio using ARIMA
    def forecast_pe(ticker):
        stock_data = df[df['Ticker Symbol'] == ticker]
        stock_data = stock_data[['date', 'P/E_EMA']].dropna()
        stock_data.set_index('date', inplace=True)

        model = ARIMA(stock_data['P/E_EMA'], order=(5,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=30)  # Predict next 30 days

        plt.figure(figsize=(10, 5))
        plt.plot(stock_data.index, stock_data['P/E_EMA'], label='P/E (EMA)', color='blue')
        plt.plot(pd.date_range(stock_data.index[-1], periods=30, freq='D'), forecast, label='Forecast', color='green')
        plt.xlabel('Date')
        plt.ylabel('P/E Ratio')
        plt.title(f'{ticker} - P/E Ratio Forecast')
        plt.legend()
        st.pyplot()

    # User selects stock ticker for analysis
    selected_stock = st.selectbox("Select Stock Ticker", df['Ticker Symbol'].unique())

    # Display P/E and P/B ratio plot
    if st.button(f"Plot P/E & P/B Ratios for {selected_stock}"):
        plot_ratios(selected_stock)

    # Display ARIMA Forecast for P/E ratio
    if st.button(f"Forecast P/E Ratio for {selected_stock}"):
        forecast_pe(selected_stock)

    # Back to Dashboard button
    if st.button("Back to Dashboard"):
        st.session_state["current_page"] = "welcome"

def compute_page():
    st.title("Top Companies Returns & Volatility")

    st.sidebar.title("Parameters")
    top_n = st.sidebar.number_input("Top N Companies", min_value=1, max_value=100, value=10, step=1)
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=100.0, value=2.0) / 100

    if st.sidebar.button("Compute Metrics"):
        results = compute_returns_volatility(top_n, risk_free_rate)

        if results is not None:
            st.subheader("Top Companies' Performance Metrics")
            st.dataframe(results)
    
            st.subheader("Visualizations")
            st.line_chart(results["Annualized Return"], use_container_width=True)
            st.bar_chart(results[["Adjusted Sharpe Ratio", "Sortino Ratio"]], use_container_width=True)

    if st.sidebar.button("Back to Dashboard"):
        st.session_state["current_page"] = "welcome"


# Navigation Logic
if st.session_state["current_page"] == "login":
    login_page()
elif st.session_state["current_page"] == "welcome":
    welcome_page()
elif st.session_state["current_page"] == "compute":
    compute_page()
elif st.session_state["current_page"] == "visualize":
    visualize_stock_performance_page()
elif st.session_state["current_page"] == "heatmap":
    heatmap_page()
elif st.session_state["current_page"] == "pe_pb_ratio":
    pe_pb_ratio_page()
elif st.session_state["current_page"] == "compare":
    stock_comparison_page()
