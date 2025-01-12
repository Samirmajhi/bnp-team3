import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# Finnhub API key
FINNHUB_API_KEY = 'cu1eh0pr01qqr3sgko5gcu1eh0pr01qqr3sgko60'

# NewsAPI API key
NEWS_API_KEY = "53cc9c5caef24412838e48c29f42ff63"

# Function: Fetch stock price from Finnhub
def get_stock_price(symbol):
    url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}'
    response = requests.get(url)
    data = response.json()
    if 'c' in data:
        return {
            "current_price": data['c'],
            "high": data['h'],
            "low": data['l'],
            "open": data['o'],
            "previous_close": data['pc']
        }
    return {"error": "Unable to fetch stock price. Check the symbol or API key."}


# Function: Static Upcoming IPO Data
def get_upcoming_ipos():
    return pd.DataFrame(
        [
            {"Company": "EMA Partners India Ltd IPO", "IPO Date": "2025-01-17", "Expected Price": "$50 - $60"},
            {"Company": "Laxmi Dental Ltd IPO", "IPO Date": "2025-01-13", "Expected Price": "$30 - $40"},
            {"Company": "Landmark Immigration Consultants Ltd IPO", "IPO Date": "2025-01-16", "Expected Price": "$70 - $90"},
        ]
    )

# Main Content after "Welcome Admin"
def welcome_page():
    st.title("Welcome Admin")

    # Displaying the current login user's name or message
    st.subheader("Hello, Admin! You are logged in.")

    # Add widgets here
    
    # Trending News Section
    st.subheader("📢 Trending News")
    news = fetch_trending_news()
    if news:
        for article in news:
            st.markdown(
                f"""
                <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 10px; background: #f9f9f9;">
                    <h4 style="color: #4CAF50;">{article['title']}</h4>
                    <a href="{article['url']}" target="_blank" style="text-decoration: none; color: blue;">Read more</a>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.write("No trending news available.")

    st.markdown("---")

    # Stock Market Section
    st.subheader("📈 Stock Market")
    # Dropdown for stock tickers
    stock_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'BRK.A', 'V']
    selected_ticker = st.selectbox("Select a Stock Ticker:", stock_tickers, index=0)

    if selected_ticker:
        stock_info = get_stock_price(selected_ticker)
        if "error" in stock_info:
            st.write(stock_info["error"])
        else:
            st.markdown(f"### Stock: {selected_ticker.upper()}")
            st.write(f"**Current Price:** ${stock_info['current_price']}")
            st.write(f"**Previous Close:** ${stock_info['previous_close']}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("High", f"${stock_info['high']}")
            with col2:
                st.metric("Low", f"${stock_info['low']}")
            st.write(f"**Open:** ${stock_info['open']}")

    st.markdown("---")

    # Upcoming IPO Section
    st.subheader("🚀 Upcoming IPOs")
    ipos = get_upcoming_ipos()
    for _, row in ipos.iterrows():
        st.markdown(
            f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 10px; background: #f9f9f9;">
                <h4 style="color: #4CAF50;">{row['Company']}</h4>
                <p><b>IPO Date:</b> {row['IPO Date']}</p>
                <p><b>Expected Price:</b> {row['Expected Price']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
# Navigation Logic
if st.session_state["current_page"] == "login":
    login_page()
elif st.session_state["current_page"] == "welcome":
    welcome_page()
elif st.session_state["current_page"] == "compute":
    compute_page()
elif st.session_state["current_page"] == "visualize":
    visualize_stock_performance_page()
