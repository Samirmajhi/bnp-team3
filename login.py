import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
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
    # Load stock prices data
    prices_df = pd.read_csv("prices-split-adjusted.csv", parse_dates=["date"])

    # Select 10 random stocks
    np.random.seed(42)  # Ensures reproducibility
    selected_stocks = np.random.choice(prices_df["symbol"].unique(), 10, replace=False)

    # Filter data for selected stocks
    portfolio_df = prices_df[prices_df["symbol"].isin(selected_stocks)].copy()

    # Compute daily returns
    portfolio_df.sort_values(by=["symbol", "date"], inplace=True)
    portfolio_df["daily_return"] = portfolio_df.groupby("symbol")["close"].pct_change()

    # Initialize equal weights for the portfolio
    weights = {stock: 1 / len(selected_stocks) for stock in selected_stocks}

    # Compute portfolio daily returns
    portfolio_df["weighted_return"] = portfolio_df["symbol"].map(weights) * portfolio_df["daily_return"]
    portfolio_returns = portfolio_df.groupby("date")["weighted_return"].sum()

    # Compute cumulative returns
    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()

    # Normalize to an index of 100
    portfolio_index = 100 * portfolio_cumulative_returns / portfolio_cumulative_returns.iloc[0]

    # Compute key investment metrics
    trading_days = 252
    annual_return = (1 + portfolio_returns.mean())**trading_days - 1
    annual_volatility = portfolio_returns.std() * np.sqrt(trading_days)

    # Compute Skewness and Kurtosis
    return_skewness = skew(portfolio_returns.dropna())
    return_kurtosis = kurtosis(portfolio_returns.dropna())

    # Compute downside deviation for Sortino Ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(trading_days)

    # Compute risk-free rate (assumed 2%)
    risk_free_rate = 0.02

    # Compute Adjusted Sharpe Ratio (ASR)
    ASR = ((annual_return - risk_free_rate) / annual_volatility) * (1 + (6 * return_skewness) + ((24 * return_kurtosis - 3) / 24))

    # Compute Sortino Ratio
    sortino_ratio = (annual_return - risk_free_rate) / downside_deviation

    # Compute Omega Ratio (Threshold = Risk-Free Rate)
    returns_above_threshold = portfolio_returns[portfolio_returns > risk_free_rate]
    returns_below_threshold = portfolio_returns[portfolio_returns < risk_free_rate]
    omega_ratio = returns_above_threshold.sum() / abs(returns_below_threshold.sum())

    # Create a DataFrame to display selected stocks and their weights
    portfolio_summary = pd.DataFrame({
        "Stock Symbol": selected_stocks,
        "Equal Weight (%)": [weights[s] * 100 for s in selected_stocks]
    })

    st.subheader("Selected Stocks for Dummy Portfolio")
    st.dataframe(portfolio_summary)

    # Print key portfolio metrics
    portfolio_metrics = pd.DataFrame({
        "Metric": ["Annualized Return", "Annualized Volatility", "Adjusted Sharpe Ratio (ASR)", "Sortino Ratio", "Omega Ratio"],
        "Value": [f"{annual_return:.2%}", f"{annual_volatility:.2%}", f"{ASR:.2f}", f"{sortino_ratio:.2f}", f"{omega_ratio:.2f}"]
    })

    st.subheader("Portfolio Performance Metrics")
    st.dataframe(portfolio_metrics)

    # Plot portfolio performance
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

    # Button to go back to Dashboard
    if st.button("Back to Dashboard"):
        st.session_state["current_page"] = "welcome"

# Function to visualize heatmap of the top 50 most traded stocks
def heatmap_page():
    # Load datasets
    prices = pd.read_csv("prices-split-adjusted.csv")
    securities = pd.read_csv("securities.csv")

    # Ensure 'date' column is in datetime format
    prices['date'] = pd.to_datetime(prices['date'], errors='coerce')

    # Extract years and filter between 2012 and 2017
    years_available = sorted(prices['date'].dt.year.unique())
    years_filtered = [year for year in years_available if 2012 <= year <= 2017]

    # Use Streamlit sidebar to select the year
    selected_year = st.sidebar.selectbox("Select Year", options=years_filtered, index=len(years_filtered) - 1)

    # Filter year of interest
    prices['year'] = prices['date'].dt.year
    df_year = prices[prices['year'] == selected_year]

    # Compute total trading volume per stock
    top_50_stocks = df_year.groupby('symbol')['volume'].sum().nlargest(50).index

    # Filter dataset for top 50 stocks
    df_top50 = df_year[df_year['symbol'].isin(top_50_stocks)]

    # Compute normalized stock price changes (Relative Performance)
    df_pivot = df_top50.pivot_table(index='date', columns='symbol', values='close')
    df_pivot = df_pivot.pct_change().fillna(0)  # Compute daily returns

    # Normalize data for better visualization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = pd.DataFrame(scaler.fit_transform(df_pivot), columns=df_pivot.columns, index=df_pivot.index)

    # Merge with sector data (Optional: Sector-Wise Heatmap)
    sector_mapping = securities.set_index("Ticker symbol")["GICS Sector"].to_dict()
    sector_order = [sector_mapping.get(symbol, "Unknown") for symbol in df_scaled.columns]
    df_scaled.columns = pd.MultiIndex.from_tuples(list(zip(sector_order, df_scaled.columns)))

    # Plot heatmap
    st.subheader(f"Heatmap of Top 50 Most Traded Stocks' Price Performance ({selected_year})")
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df_scaled.T, cmap="RdYlGn", center=0, linewidths=0.5, cbar=True, ax=ax)
    ax.set_title(f"Heatmap of Top 50 Most Traded Stocks' Price Performance ({selected_year})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Symbols (Grouped by Sector)")
    st.pyplot(fig)

    # Back to Dashboard button
    if st.button("Back to Dashboard"):
        st.session_state["current_page"] = "welcome"

def stock_comparison_page():
    st.title("Stock Price Movement Comparison")

    # Load stock price data
    prices = pd.read_csv("prices-split-adjusted.csv", parse_dates=["date"])

    # Sidebar for stock selection
    st.sidebar.title("Stock Selection")
    stock1 = st.sidebar.selectbox("Select First Stock", options=prices["symbol"].unique())
    stock2 = st.sidebar.selectbox("Select Second Stock", options=prices["symbol"].unique())

    # Sidebar for date range selection
    st.sidebar.title("Date Range")
    start_date = st.sidebar.date_input("Start Date", min_value=prices["date"].min(), max_value=prices["date"].max())
    end_date = st.sidebar.date_input("End Date", min_value=prices["date"].min(), max_value=prices["date"].max())

    if start_date > end_date:
        st.error("Start Date cannot be after End Date!")
        return

    # Filter data for the selected stocks and date range
    filtered_data = prices[(prices["symbol"].isin([stock1, stock2])) & 
                           (prices["date"] >= pd.Timestamp(start_date)) & 
                           (prices["date"] <= pd.Timestamp(end_date))]

    # Pivot table to align stock prices
    pivot_df = filtered_data.pivot(index="date", columns="symbol", values="close")

    # Normalize prices to 100 for better comparison
    normalized_prices = pivot_df / pivot_df.iloc[0] * 100

    # Display stock comparison chart    
    st.subheader(f"Stock Price Movement: {stock1} vs {stock2}")
    plt.figure(figsize=(12, 6))
    for stock in [stock1, stock2]:
        plt.plot(normalized_prices.index, normalized_prices[stock], label=stock)
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (Base = 100)")
    plt.title("Stock Price Movement Comparison")
    plt.legend()
    plt.grid(True)
    st.pyplot()

    # Back to Dashboard button
    if st.button("Back to Dashboard"):
        st.session_state["current_page"] = "welcome"

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


# Pages
def login_page():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        login(username, password)

def welcome_page():
    st.title("DASHBOARD")

    if st.button("Top Companies"):
        st.session_state["current_page"] = "compute"

    if st.button("Visualize Dummy Stock Performance"):
        st.session_state["current_page"] = "visualize"

    if st.button("Heatmap of Top 50 Most Traded Stocks"):
        st.session_state["current_page"] = "heatmap"
    
    if st.button("Compare Stock Price Movement"):
        st.session_state["current_page"] = "compare"

    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["current_page"] = "login"

def compute_page():
    st.title("Top Companies Returns & Volatility")

    st.sidebar.title("Parameters")
    top_n = st.sidebar.number_input("Top N Companies", min_value=1, max_value=100, value=10, step=1)
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=100.0, value=2.0) / 100

    if st.sidebar.button("Compute Metrics"):
        results = compute_returns_volatility()

        if results is not None:
            st.subheader("Top Companies' Performance Metrics")
            st.dataframe(results)

            st.subheader("Visualizations")
            st.line_chart(results["Annualized Return"], use_container_width=True)
            st.bar_chart(results[["Adjusted Sharpe Ratio", "Sortino Ratio"]], use_container_width=True)

    if st.button("Back to Dashboard"):
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
elif st.session_state["current_page"] == "compare":
    stock_comparison_page()

