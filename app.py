import os
from datetime import datetime, timedelta, time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from ML_engine import MLEngine
from database import init_db, add_risk_score, get_risk_scores
from fundamentals import FundamentalsService
from news_service import NewsService

# RAG Imports
try:
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA
    import openai  # noqa: F401
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gorilla Glue | FX Hedging Dashboard",
    page_icon="🦍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- CUSTOM CSS (Financial Terminal Look) ---
st.markdown(
    """
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    div[data-testid="stMetricLabel"] {
        color: #8b949e;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        color: #f0f6fc;
        font-weight: 600;
    }
    .risk-badge-critical {
        background-color: #7f1d1d;
        color: #fecaca;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        border: 1px solid #f87171;
    }
    .risk-badge-high {
        background-color: #78350f;
        color: #fde68a;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        border: 1px solid #fbbf24;
    }
    .risk-badge-med {
        background-color: #14532d;
        color: #bbf7d0;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        border: 1px solid #4ade80;
    }
    .risk-badge-low {
        background-color: #1f2937;
        color: #9ca3af;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        border: 1px solid #4b5563;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #8b949e;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #58a6ff;
        border-bottom: 2px solid #58a6ff;
    }
    h1, h2, h3 {
        color: #f0f6fc;
        font-family: 'Segoe UI', sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #010409;
        border-right: 1px solid #30363d;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- INITIALIZATION ---
init_db()

TICKER_MAP = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CHF": "CHF=X",
    "AUD/USD": "AUDUSD=X",
}
INDICATOR_WEIGHTS = {
    "Trend": 30,
    "Momentum": 30,
    "Volatility": 25,
    "RSI": 15,
}
MIN_LOOKBACK_DAYS = 60
WARMUP_DAYS = 90


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["SMA"] = data["Close"].rolling(window=20).mean()
    data["EMA"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["BB_Middle"] = data["Close"].rolling(window=20).mean()
    data["BB_Std"] = data["Close"].rolling(window=20).std()
    data["BB_Upper"] = data["BB_Middle"] + (2 * data["BB_Std"])
    data["BB_Lower"] = data["BB_Middle"] - (2 * data["BB_Std"])

    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    exp1 = data["Close"].ewm(span=12, adjust=False).mean()
    exp2 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = exp1 - exp2
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
    return data


def compute_signal_details(last_row: pd.Series, prev_row: pd.Series, selected_indicators: list[str]) -> dict:
    trend_active = "SMA" in selected_indicators and "EMA" in selected_indicators
    trend_bullish = bool(last_row["EMA"] > last_row["SMA"])

    momentum_active = "MACD" in selected_indicators
    momentum_bullish = bool((last_row["MACD"] > last_row["Signal_Line"]) and (last_row["MACD"] > 0))

    volatility_active = "Bollinger Bands" in selected_indicators
    bb_middle_rising = bool(last_row["BB_Middle"] > prev_row["BB_Middle"])
    volatility_bullish = bool((last_row["Close"] > last_row["BB_Upper"]) or bb_middle_rising)

    rsi_active = "RSI" in selected_indicators
    rsi_rising = bool(last_row["RSI"] > prev_row["RSI"])
    rsi_bullish = bool((last_row["RSI"] > 50) and rsi_rising and (last_row["RSI"] < 70))

    return {
        "Trend": {"active": trend_active, "bullish": trend_bullish, "points": INDICATOR_WEIGHTS["Trend"]},
        "Momentum": {"active": momentum_active, "bullish": momentum_bullish, "points": INDICATOR_WEIGHTS["Momentum"]},
        "Volatility": {"active": volatility_active, "bullish": volatility_bullish, "points": INDICATOR_WEIGHTS["Volatility"]},
        "RSI": {"active": rsi_active, "bullish": rsi_bullish, "points": INDICATOR_WEIGHTS["RSI"]},
    }


def compute_risk_score(signal_details: dict, position_type: str) -> dict:
    bullish_points = sum(item["points"] for item in signal_details.values() if item["active"] and item["bullish"])
    max_points = sum(item["points"] for item in signal_details.values() if item["active"])

    bullish_score = int((bullish_points / max_points) * 100) if max_points > 0 else 0
    bearish_score = 100 - bullish_score

    if position_type == "Long":
        risk_score = bearish_score
        risk_label = "Bearish Reversal Risk"
    else:
        risk_score = bullish_score
        risk_label = "Bullish Reversal Risk"

    if max_points == 0:
        action = "N/A"
        badge_class = "risk-badge-low"
    elif risk_score >= 80:
        action = "Strong Hedge (75-100%)"
        badge_class = "risk-badge-critical"
    elif risk_score >= 60:
        action = "Moderate Hedge (50-75%)"
        badge_class = "risk-badge-high"
    elif risk_score >= 40:
        action = "Partial Hedge (25-50%)"
        badge_class = "risk-badge-med"
    else:
        action = "No Hedge / Unwind"
        badge_class = "risk-badge-low"

    return {
        "bullish_points": bullish_points,
        "max_points": max_points,
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "risk_score": risk_score,
        "risk_label": risk_label,
        "action": action,
        "badge_class": badge_class,
    }


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🦍 Gorilla Glue")
    st.caption("FX Hedging Intelligence")
    st.markdown("---")

    st.subheader("⚙️ Configuration")
    currency_pair = st.selectbox("Currency Pair", list(TICKER_MAP.keys()))
    position_type = st.radio(
        "Position Type",
        ["Long", "Short"],
        horizontal=True,
        help="Long: You own the currency. Short: You owe the currency.",
    )

    st.subheader("📈 Indicators")
    selected_indicators = st.multiselect(
        "Active Signals",
        ["SMA", "EMA", "Bollinger Bands", "RSI", "MACD"],
        default=["SMA", "EMA", "Bollinger Bands", "RSI", "MACD"],
    )

    st.subheader("📅 Analysis Date Range")
    default_start = datetime.now().date() - timedelta(days=365)
    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=datetime.now().date())
    st.caption("The score, charts, and forecast now use this selected range.")

    st.markdown("---")
    st.subheader("🤖 AI Assistant")
    api_key = st.text_input("OpenAI API Key", type="password", help="Required for RAG Chatbot")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about hedging..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not api_key:
                st.error("Please enter an OpenAI API Key.")
            elif not RAG_AVAILABLE:
                st.error("RAG dependencies not installed.")
            else:
                try:
                    embeddings = OpenAIEmbeddings()
                    if not os.path.exists("knowledge_base.txt"):
                        with open("knowledge_base.txt", "w") as f:
                            f.write("Gorilla Glue is an FX hedging tool using technical analysis and ML.")

                    loader = TextLoader("knowledge_base.txt")
                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                    texts = text_splitter.split_documents(documents)

                    persist_directory = "./chroma_db"
                    vectordb = Chroma.from_documents(
                        documents=texts,
                        embedding=embeddings,
                        persist_directory=persist_directory,
                    )

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
                        chain_type="stuff",
                        retriever=vectordb.as_retriever(),
                    )

                    response = qa_chain.run(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# --- DATA FETCHING & LOGIC ---
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

selected_start_dt = pd.Timestamp(start_date)
selected_end_dt = pd.Timestamp(end_date)
fetch_start_dt = selected_start_dt - timedelta(days=WARMUP_DAYS)
fetch_end_dt = selected_end_dt + timedelta(days=1)

ticker = TICKER_MAP[currency_pair]
raw_df = fetch_market_data(
    ticker,
    fetch_start_dt.strftime("%Y-%m-%d"),
    fetch_end_dt.strftime("%Y-%m-%d"),
)

if raw_df.empty:
    st.error("Failed to fetch data. Please check your connection or date range.")
    st.stop()

analysis_df = calculate_indicators(raw_df)
display_df = analysis_df[(analysis_df.index >= selected_start_dt) & (analysis_df.index < fetch_end_dt)].copy()

if display_df.empty:
    st.error("No market data was returned for the selected date range.")
    st.stop()

valid_score_df = display_df.dropna(subset=["SMA", "EMA", "BB_Middle", "BB_Upper", "RSI", "MACD", "Signal_Line"]).copy()
if len(valid_score_df) < 2:
    st.error(
        "Not enough usable data in the selected range to calculate the indicators and risk score. "
        "Choose a wider range."
    )
    st.stop()

if len(display_df) < MIN_LOOKBACK_DAYS:
    st.warning(
        f"Selected range is short ({len(display_df)} rows). The dashboard will still run, "
        f"but {MIN_LOOKBACK_DAYS}+ rows is recommended for more stable signals and forecasting."
    )

last_row = valid_score_df.iloc[-1]
prev_row = valid_score_df.iloc[-2]
current_price = last_row["Close"]
price_change = last_row["Close"] - prev_row["Close"]
price_pct = (price_change / prev_row["Close"]) * 100 if prev_row["Close"] != 0 else 0

signal_details = compute_signal_details(last_row, prev_row, selected_indicators)
score_details = compute_risk_score(signal_details, position_type)
risk_score = score_details["risk_score"]
action = score_details["action"]
badge_class = score_details["badge_class"]


# --- DASHBOARD LAYOUT ---
st.markdown(f"## {currency_pair} Executive Dashboard")
st.caption(
    f"Evaluated as of {valid_score_df.index[-1].strftime('%Y-%m-%d')} using data from "
    f"{start_date} to {end_date}."
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Current Price", f"{current_price:.4f}", f"{price_pct:.2f}%")
with kpi2:
    st.metric("Position", position_type)
with kpi3:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 10px; background-color: #161b22; border: 1px solid #30363d; border-radius: 6px;">
            <div style="color: #8b949e; font-size: 0.8rem; margin-bottom: 4px;">RISK SCORE</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #f0f6fc;">{risk_score}/100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 10px; background-color: #161b22; border: 1px solid #30363d; border-radius: 6px;">
            <div style="color: #8b949e; font-size: 0.8rem; margin-bottom: 4px;">RECOMMENDATION</div>
            <div class="{badge_class}">{action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

save_col1, save_col2 = st.columns([1, 3])
with save_col1:
    if st.button("Save Score Snapshot"):
        snapshot_ts = datetime.combine(end_date, time(23, 59, 59)).strftime("%Y-%m-%d %H:%M:%S")
        add_risk_score(currency_pair, risk_score, action, position_type, timestamp=snapshot_ts)
        st.success("Snapshot saved to history.")
with save_col2:
    st.caption("This avoids saving a new database row on every rerun.")

st.markdown("---")

# 1. Main Chart
st.subheader("📊 Market Overview")
fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=display_df.index,
        open=display_df["Open"],
        high=display_df["High"],
        low=display_df["Low"],
        close=display_df["Close"],
        name="Price",
    )
)
if "SMA" in selected_indicators:
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df["SMA"], line=dict(color="orange", width=1), name="SMA 20"))
if "EMA" in selected_indicators:
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df["EMA"], line=dict(color="cyan", width=1), name="EMA 20"))
if "Bollinger Bands" in selected_indicators:
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df["BB_Upper"], line=dict(color="gray", width=0), showlegend=False, name="BB Upper"))
    fig.add_trace(
        go.Scatter(
            x=display_df.index,
            y=display_df["BB_Lower"],
            line=dict(color="gray", width=0),
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.1)",
            name="Bollinger Bands",
        )
    )

fig.update_layout(
    height=500,
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9"),
    xaxis=dict(showgrid=True, gridcolor="#30363d", rangeslider=dict(visible=False)),
    yaxis=dict(showgrid=True, gridcolor="#30363d"),
    hovermode="x unified",
)
st.plotly_chart(fig, width="stretch")

# 2. Analysis Tabs
tab_tech, tab_macro, tab_quant = st.tabs(["📉 Technicals", "🌍 Macro & News", "🧠 Quant & Forecast"])

# --- TAB A: TECHNICALS ---
with tab_tech:
    st.subheader("Technical Signal Breakdown")
    t1, t2, t3, t4 = st.columns(4)

    def signal_card(title: str, active: bool, bullish: bool, points: int):
        status_color = "#238636" if bullish else "#da3633"
        status_text = "BULLISH" if bullish else "BEARISH"
        if not active:
            status_color = "#484f58"
            status_text = "INACTIVE"

        st.markdown(
            f"""
            <div style="background-color: #161b22; padding: 15px; border-radius: 6px; border-left: 4px solid {status_color};">
                <div style="font-weight: bold; color: #f0f6fc;">{title}</div>
                <div style="color: {status_color}; font-size: 0.9rem; margin-top: 5px;">{status_text}</div>
                <div style="color: #8b949e; font-size: 0.8rem; margin-top: 5px;">Weight: {points} pts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t1:
        signal_card("Trend (EMA vs SMA)", signal_details["Trend"]["active"], signal_details["Trend"]["bullish"], signal_details["Trend"]["points"])
    with t2:
        signal_card("Momentum (MACD)", signal_details["Momentum"]["active"], signal_details["Momentum"]["bullish"], signal_details["Momentum"]["points"])
    with t3:
        signal_card("Volatility (BB)", signal_details["Volatility"]["active"], signal_details["Volatility"]["bullish"], signal_details["Volatility"]["points"])
    with t4:
        signal_card("Relative Strength (RSI)", signal_details["RSI"]["active"], signal_details["RSI"]["bullish"], signal_details["RSI"]["points"])

    st.markdown("---")
    st.subheader("💡 Technical Analysis Summary")

    bullish_signals = []
    bearish_signals = []
    if signal_details["Trend"]["active"]:
        if signal_details["Trend"]["bullish"]:
            bullish_signals.append("Strong upward trend (EMA > SMA)")
        else:
            bearish_signals.append("Downward trend (EMA ≤ SMA)")
    if signal_details["Momentum"]["active"]:
        if signal_details["Momentum"]["bullish"]:
            bullish_signals.append("Positive momentum (MACD bullish crossover)")
        else:
            bearish_signals.append("Weak momentum (MACD bearish)")
    if signal_details["Volatility"]["active"]:
        if last_row["Close"] > last_row["BB_Upper"]:
            bullish_signals.append("Price breakout above upper Bollinger Band")
        elif last_row["Close"] < last_row["BB_Lower"]:
            bearish_signals.append("Price breakdown below lower Bollinger Band")
    if signal_details["RSI"]["active"]:
        if last_row["RSI"] > 70:
            bearish_signals.append(f"Overbought RSI ({last_row['RSI']:.1f})")
        elif last_row["RSI"] < 30:
            bullish_signals.append(f"Oversold RSI ({last_row['RSI']:.1f})")

    if position_type == "Long":
        if bearish_signals:
            st.warning(f"**⚠️ Bearish Signals Detected:** {', '.join(bearish_signals)}. Consider hedging your long position.")
        elif bullish_signals:
            st.success(f"**✅ Bullish Signals:** {', '.join(bullish_signals)}. Your long position looks healthy.")
        else:
            st.info("No strong technical confluence right now.")
    else:
        if bullish_signals:
            st.warning(f"**⚠️ Bullish Signals Detected:** {', '.join(bullish_signals)}. Consider hedging your short position.")
        elif bearish_signals:
            st.success(f"**✅ Bearish Signals:** {', '.join(bearish_signals)}. Your short position looks healthy.")
        else:
            st.info("No strong technical confluence right now.")

    st.markdown("---")
    if "RSI" in selected_indicators:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=display_df.index, y=display_df["RSI"], line=dict(color="#a371f7", width=2), name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"),
            title="RSI Momentum",
        )
        st.plotly_chart(fig_rsi, width="stretch")

# --- TAB B: MACRO & NEWS ---
with tab_macro:
    m_col1, m_col2 = st.columns([1, 1])
    base_ccy = currency_pair.split("/")[0]
    quote_ccy = currency_pair.split("/")[1]

    with m_col1:
        st.subheader("Economic Indicators")
        fund_service = FundamentalsService()
        fund_df = fund_service.get_comparison_df(base_ccy, quote_ccy)

        if fund_df is not None:
            st.dataframe(
                fund_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Metric": st.column_config.TextColumn("Indicator", width="medium"),
                    base_ccy: st.column_config.TextColumn(f"{base_ccy}", width="small"),
                    quote_ccy: st.column_config.TextColumn(f"{quote_ccy}", width="small"),
                },
            )
        else:
            st.warning("Data unavailable.")

        st.markdown("---")
        st.subheader("💡 Macro Analysis")
        if fund_df is not None:
            base_data = fund_service.get_fundamentals(base_ccy)
            quote_data = fund_service.get_fundamentals(quote_ccy)
            if base_data and quote_data:
                base_rate = float(base_data.get("Interest Rate", "0").replace("%", ""))
                quote_rate = float(quote_data.get("Interest Rate", "0").replace("%", ""))
                base_gdp = float(base_data.get("GDP Growth (YoY)", "0").replace("%", ""))
                quote_gdp = float(quote_data.get("GDP Growth (YoY)", "0").replace("%", ""))
                macro_insights = []
                if base_rate > quote_rate:
                    macro_insights.append(f"{base_ccy} has higher interest rates ({base_rate}% vs {quote_rate}%), which typically attracts capital and strengthens the currency.")
                elif quote_rate > base_rate:
                    macro_insights.append(f"{quote_ccy} has higher interest rates ({quote_rate}% vs {base_rate}%), which may weaken {base_ccy}.")
                if base_gdp > quote_gdp:
                    macro_insights.append(f"{base_ccy} shows stronger economic growth ({base_gdp}% vs {quote_gdp}% GDP growth).")
                elif quote_gdp > base_gdp:
                    macro_insights.append(f"{quote_ccy} shows stronger economic growth ({quote_gdp}% vs {base_gdp}% GDP growth).")
                if macro_insights:
                    st.info("**Key Insights:** " + " ".join(macro_insights))

    with m_col2:
        st.subheader("Major Headlines")
        news_service = NewsService()
        news_items = news_service.get_combined_news(base_ccy, quote_ccy)
        if news_items:
            for item in news_items:
                badge_color = "#1f6feb" if item["currency"] == base_ccy else "#238636"
                st.markdown(
                    f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: #161b22; border: 1px solid #30363d; margin-bottom: 10px;">
                        <span style="background-color: {badge_color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7em; font-weight: bold;">{item['currency']}</span>
                        <span style="color: #8b949e; font-size: 0.8em; margin-left: 8px;">{item['time']}</span>
                        <div style="margin-top: 5px; font-weight: 500; font-size: 0.9em; color: #c9d1d9;">{item['title']}</div>
                        <div style="color: #8b949e; font-size: 0.75em; margin-top: 2px;">Source: {item['source']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No headlines found.")

# --- TAB C: QUANT & FORECAST ---
with tab_quant:
    st.subheader("ML Price Prediction (30 Days)")
    forecast_source_df = display_df.dropna().copy()
    with st.spinner("Running Random Forest Model..."):
        ml_engine = MLEngine()
        forecast_df = ml_engine.train_and_predict(forecast_source_df, forecast_days=30) if len(forecast_source_df) >= 100 else None

    if forecast_df is not None:
        fig_ml = go.Figure()
        fig_ml.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted_Close"], line=dict(color="#58a6ff", width=2), name="Forecast"))
        fig_ml.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Upper_Bound"], line=dict(width=0), showlegend=False))
        fig_ml.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Lower_Bound"], line=dict(width=0), fill="tonexty", fillcolor="rgba(88, 166, 255, 0.2)", name="95% Conf. Interval"))
        fig_ml.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"),
            xaxis_title="Date",
            yaxis_title="Price",
        )
        st.plotly_chart(fig_ml, width="stretch")

        m1, m2, m3 = st.columns(3)
        m1.metric("Forecast End Price", f"{forecast_df['Predicted_Close'].iloc[-1]:.4f}")
        m2.metric("Model MSE", f"{ml_engine.mse:.6f}")
        m3.metric("Prediction Range", f"{forecast_df['Lower_Bound'].iloc[-1]:.4f} - {forecast_df['Upper_Bound'].iloc[-1]:.4f}")

        st.markdown("---")
        st.subheader("💡 Forecast Interpretation")
        forecast_start = current_price
        forecast_end = forecast_df["Predicted_Close"].iloc[-1]
        forecast_change = ((forecast_end - forecast_start) / forecast_start) * 100 if forecast_start != 0 else 0

        if abs(forecast_change) < 1:
            st.info(f"📊 **Neutral Outlook:** Model predicts minimal movement ({forecast_change:+.2f}%) over the next 30 days. Consider maintaining current hedges.")
        elif forecast_change > 0:
            if position_type == "Long":
                st.success(f"📈 **Bullish Forecast:** Model predicts a {forecast_change:+.2f}% upside. Your long position aligns with the forecast.")
            else:
                st.warning(f"📈 **Bullish Forecast:** Model predicts a {forecast_change:+.2f}% upside, which works against your short position. Consider hedging.")
        else:
            if position_type == "Short":
                st.success(f"📉 **Bearish Forecast:** Model predicts a {forecast_change:+.2f}% downside. Your short position aligns with the forecast.")
            else:
                st.warning(f"📉 **Bearish Forecast:** Model predicts a {forecast_change:+.2f}% downside, which works against your long position. Consider hedging.")
    else:
        st.info("Not enough data in the selected range for the ML forecast. Use a wider range if you want the model to run.")

    st.markdown("---")
    st.subheader("Risk Score History")
    history_df = get_risk_scores(pair=currency_pair, months=None)
    if not history_df.empty:
        history_start = pd.Timestamp(start_date)
        history_end = pd.Timestamp(end_date) + timedelta(days=1)
        history_df = history_df[(history_df["timestamp"] >= history_start) & (history_df["timestamp"] < history_end)]

    if not history_df.empty and len(history_df) > 1:
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Scatter(
                x=history_df["timestamp"],
                y=history_df["risk_score"],
                mode="markers+lines",
                marker=dict(size=8, color="#58a6ff", symbol="circle"),
                name="Risk Score",
            )
        )
        fig_hist.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"),
            xaxis=dict(title="Date", range=[pd.Timestamp(start_date), pd.Timestamp(end_date) + timedelta(days=1)]),
            yaxis=dict(title="Risk Score", range=[0, 100]),
            shapes=[
                dict(type="rect", xref="paper", yref="y", x0=0, y0=0, x1=1, y1=40, fillcolor="green", opacity=0.1, layer="below", line_width=0),
                dict(type="rect", xref="paper", yref="y", x0=0, y0=40, x1=1, y1=60, fillcolor="yellow", opacity=0.1, layer="below", line_width=0),
                dict(type="rect", xref="paper", yref="y", x0=0, y0=60, x1=1, y1=80, fillcolor="orange", opacity=0.1, layer="below", line_width=0),
                dict(type="rect", xref="paper", yref="y", x0=0, y0=80, x1=1, y1=100, fillcolor="red", opacity=0.1, layer="below", line_width=0),
            ],
        )
        st.plotly_chart(fig_hist, width="stretch")
        st.metric("Avg Risk Score", f"{history_df['risk_score'].mean():.1f}")
    else:
        st.info("Not enough saved history in this date range. Save a few snapshots or run your historical generator.")
