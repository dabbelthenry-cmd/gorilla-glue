import os
from datetime import datetime, timedelta

import numpy as np
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
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    div[data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.9rem; }
    div[data-testid="stMetricValue"] { color: #f0f6fc; font-weight: 600; }

    .risk-badge-critical {
        background-color: #7f1d1d; color: #fecaca; padding: 4px 12px;
        border-radius: 12px; font-weight: bold; border: 1px solid #f87171;
    }
    .risk-badge-high {
        background-color: #78350f; color: #fde68a; padding: 4px 12px;
        border-radius: 12px; font-weight: bold; border: 1px solid #fbbf24;
    }
    .risk-badge-med {
        background-color: #14532d; color: #bbf7d0; padding: 4px 12px;
        border-radius: 12px; font-weight: bold; border: 1px solid #4ade80;
    }
    .risk-badge-low {
        background-color: #1f2937; color: #9ca3af; padding: 4px 12px;
        border-radius: 12px; font-weight: bold; border: 1px solid #4b5563;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: transparent;
        border-radius: 4px 4px 0px 0px; color: #8b949e; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent; color: #58a6ff; border-bottom: 2px solid #58a6ff;
    }

    h1, h2, h3 { color: #f0f6fc; font-family: 'Segoe UI', sans-serif; }
    section[data-testid="stSidebar"] {
        background-color: #010409; border-right: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
init_db()


# ---------------------------
# Helpers
# ---------------------------
def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # SMA / EMA
    data["SMA"] = data["Close"].rolling(window=20).mean()
    data["EMA"] = data["Close"].ewm(span=20, adjust=False).mean()

    # Bollinger Bands
    data["BB_Middle"] = data["Close"].rolling(window=20).mean()
    data["BB_Std"] = data["Close"].rolling(window=20).std()
    data["BB_Upper"] = data["BB_Middle"] + (2 * data["BB_Std"])
    data["BB_Lower"] = data["BB_Middle"] - (2 * data["BB_Std"])

    # RSI
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))
    data["RSI"] = data["RSI"].fillna(50)

    # MACD
    exp1 = data["Close"].ewm(span=12, adjust=False).mean()
    exp2 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = exp1 - exp2
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

    return data


def score_trend(last_row: pd.Series, weight: int = 30) -> tuple[int, bool]:
    price = max(abs(safe_float(last_row["Close"], 1.0)), 1e-9)
    ema = safe_float(last_row["EMA"])
    sma = safe_float(last_row["SMA"])
    diff_pct = (ema - sma) / price

    # 0.5% separation ~= full points for daily FX
    scaled = clamp((diff_pct / 0.005 + 1) / 2)
    points = int(round(weight * scaled))
    bullish = ema > sma
    return points, bullish


def score_momentum(last_row: pd.Series, weight: int = 30) -> tuple[int, bool]:
    price = max(abs(safe_float(last_row["Close"], 1.0)), 1e-9)
    macd = safe_float(last_row["MACD"])
    signal = safe_float(last_row["Signal_Line"])
    spread_pct = (macd - signal) / price
    level_pct = macd / price

    spread_component = clamp((spread_pct / 0.003 + 1) / 2)
    level_component = clamp((level_pct / 0.003 + 1) / 2)
    scaled = 0.6 * spread_component + 0.4 * level_component

    points = int(round(weight * scaled))
    bullish = (macd > signal) and (macd > 0)
    return points, bullish


def score_volatility(last_row: pd.Series, prev_row: pd.Series, weight: int = 25) -> tuple[int, bool]:
    close = safe_float(last_row["Close"])
    bb_mid = safe_float(last_row["BB_Middle"])
    bb_upper = safe_float(last_row["BB_Upper"])
    bb_lower = safe_float(last_row["BB_Lower"])
    prev_mid = safe_float(prev_row["BB_Middle"])

    band_width = max(bb_upper - bb_lower, 1e-9)
    position_in_band = (close - bb_lower) / band_width   # 0 = lower band, 1 = upper band
    position_component = clamp(position_in_band)

    slope = (bb_mid - prev_mid) / max(abs(bb_mid), 1e-9)
    slope_component = clamp((slope / 0.002 + 1) / 2)

    scaled = 0.7 * position_component + 0.3 * slope_component
    points = int(round(weight * scaled))
    bullish = (close > bb_upper) or (bb_mid > prev_mid)
    return points, bullish


def score_rsi(last_row: pd.Series, prev_row: pd.Series, weight: int = 15) -> tuple[int, bool]:
    rsi = safe_float(last_row["RSI"], 50)
    prev_rsi = safe_float(prev_row["RSI"], 50)

    level_component = clamp(rsi / 100.0)
    slope_component = clamp(((rsi - prev_rsi) / 10 + 1) / 2)

    overbought_penalty = 0.0
    if rsi > 70:
        overbought_penalty = clamp((rsi - 70) / 30)

    scaled = max(0.0, 0.75 * level_component + 0.25 * slope_component - 0.35 * overbought_penalty)
    points = int(round(weight * scaled))
    bullish = (rsi > 50) and (rsi > prev_rsi) and (rsi < 70)
    return points, bullish


def calculate_score(last_row: pd.Series, prev_row: pd.Series, selected_indicators: list[str], position_type: str):
    bullish_points = 0
    max_points = 0
    details = {}

    if "SMA" in selected_indicators and "EMA" in selected_indicators:
        pts, bullish = score_trend(last_row, 30)
        bullish_points += pts
        max_points += 30
        details["Trend"] = {"points": pts, "max": 30, "bullish": bullish}

    if "MACD" in selected_indicators:
        pts, bullish = score_momentum(last_row, 30)
        bullish_points += pts
        max_points += 30
        details["Momentum"] = {"points": pts, "max": 30, "bullish": bullish}

    if "Bollinger Bands" in selected_indicators:
        pts, bullish = score_volatility(last_row, prev_row, 25)
        bullish_points += pts
        max_points += 25
        details["Volatility"] = {"points": pts, "max": 25, "bullish": bullish}

    if "RSI" in selected_indicators:
        pts, bullish = score_rsi(last_row, prev_row, 15)
        bullish_points += pts
        max_points += 15
        details["RSI"] = {"points": pts, "max": 15, "bullish": bullish}

    bullish_score = int(round((bullish_points / max_points) * 100)) if max_points > 0 else 0
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
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "risk_score": risk_score,
        "risk_label": risk_label,
        "action": action,
        "badge_class": badge_class,
        "details": details,
    }


@st.cache_data(show_spinner=False)
def fetch_market_data(ticker: str, start_str: str, end_str: str, refresh_nonce: int):
    start_dt = pd.to_datetime(start_str) - timedelta(days=120)
    end_dt = pd.to_datetime(end_str) + timedelta(days=1)

    df = yf.download(
        ticker,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def build_rag_answer(prompt: str):
    embeddings = OpenAIEmbeddings()
    if not os.path.exists("knowledge_base.txt"):
        with open("knowledge_base.txt", "w", encoding="utf-8") as f:
            f.write("Gorilla Glue is an FX hedging tool using technical analysis and ML.")

    loader = TextLoader("knowledge_base.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    persist_directory = "./chroma_db"
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )
    return qa_chain.run(prompt)


if "refresh_nonce" not in st.session_state:
    st.session_state.refresh_nonce = 0
if "last_refresh_label" not in st.session_state:
    st.session_state.last_refresh_label = "Not refreshed this session"

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🦍 Gorilla Glue")
    st.caption("FX Hedging Intelligence")
    st.markdown("---")

    st.subheader("⚙️ Configuration")
    currency_pair = st.selectbox("Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD"])
    position_type = st.radio("Position Type", ["Long", "Short"], horizontal=True, help="Long: You own the currency. Short: You owe the currency.")

    st.subheader("📈 Indicators")
    selected_indicators = st.multiselect(
        "Active Signals",
        ["SMA", "EMA", "Bollinger Bands", "RSI", "MACD"],
        default=["SMA", "EMA", "Bollinger Bands", "RSI", "MACD"],
    )

    st.subheader("📅 Analysis Range")
    default_end = datetime.now().date()
    default_start = (datetime.now() - timedelta(days=365)).date()
    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=default_end)

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    st.markdown("---")
    st.subheader("🔄 Data Feed")
    if st.button("Refresh Market Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.refresh_nonce += 1
        st.session_state.last_refresh_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.rerun()
    st.caption(f"Last refresh: {st.session_state.last_refresh_label}")

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
                    response = build_rag_answer(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as exc:
                    st.error(f"Error: {str(exc)}")


# --- DATA FETCHING & LOGIC ---
ticker_map = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CHF": "CHF=X",
    "AUD/USD": "AUDUSD=X",
}
ticker = ticker_map.get(currency_pair)

with st.spinner("Loading market data..."):
    raw_df = fetch_market_data(
        ticker=ticker,
        start_str=str(start_date),
        end_str=str(end_date),
        refresh_nonce=st.session_state.refresh_nonce,
    )

if raw_df.empty:
    st.error("Failed to fetch data. Please check your connection or try refreshing.")
    st.stop()

df_full = calculate_indicators(raw_df)
analysis_start = pd.to_datetime(start_date)
analysis_end = pd.to_datetime(end_date) + timedelta(days=1)
df = df_full[(df_full.index >= analysis_start) & (df_full.index < analysis_end)].copy()

if df.empty:
    st.error("No market data exists inside the selected analysis range.")
    st.stop()

required_cols = ["SMA", "EMA", "BB_Middle", "BB_Upper", "BB_Lower", "RSI", "MACD", "Signal_Line"]
valid_df = df.dropna(subset=required_cols).copy()

if len(valid_df) < 2:
    st.warning("The selected analysis range is too short to calculate the indicators reliably. Please choose a wider date range.")
    st.stop()

last_row = valid_df.iloc[-1]
prev_row = valid_df.iloc[-2]

score_result = calculate_score(last_row, prev_row, selected_indicators, position_type)
risk_score = score_result["risk_score"]
risk_label = score_result["risk_label"]
action = score_result["action"]
badge_class = score_result["badge_class"]
current_price = safe_float(last_row["Close"])
prev_price = safe_float(prev_row["Close"], current_price)
price_change = current_price - prev_price
price_pct = (price_change / prev_price * 100) if prev_price else 0.0

# --- DASHBOARD LAYOUT ---
st.markdown(f"## {currency_pair} Executive Dashboard")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Selected End-Date Price", f"{current_price:.4f}", f"{price_pct:.2f}%")
with kpi2:
    st.metric("Position", position_type, delta=None)
with kpi3:
    st.markdown(f"""
    <div style="text-align: center; padding: 10px; background-color: #161b22; border: 1px solid #30363d; border-radius: 6px;">
        <div style="color: #8b949e; font-size: 0.8rem; margin-bottom: 4px;">RISK SCORE</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: #f0f6fc;">{risk_score}/100</div>
        <div style="color: #8b949e; font-size: 0.8rem; margin-top: 4px;">{risk_label}</div>
    </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown(f"""
    <div style="text-align: center; padding: 10px; background-color: #161b22; border: 1px solid #30363d; border-radius: 6px;">
        <div style="color: #8b949e; font-size: 0.8rem; margin-bottom: 4px;">RECOMMENDATION</div>
        <div class="{badge_class}">{action}</div>
    </div>
    """, unsafe_allow_html=True)

col_save, col_meta = st.columns([1, 2])
with col_save:
    if st.button("Save Score Snapshot", use_container_width=True):
        add_risk_score(
            currency_pair,
            int(risk_score),
            action,
            position_type,
            timestamp=last_row.name.strftime("%Y-%m-%d %H:%M:%S"),
        )
        st.success("Snapshot saved.")
with col_meta:
    st.caption(f"Scored as of {last_row.name.strftime('%Y-%m-%d')} using the selected analysis range.")

st.markdown("---")

st.subheader("📊 Market Overview")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
if "SMA" in selected_indicators:
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA"], line=dict(color="orange", width=1), name="SMA 20"))
if "EMA" in selected_indicators:
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA"], line=dict(color="cyan", width=1), name="EMA 20"))
if "Bollinger Bands" in selected_indicators:
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], line=dict(color="gray", width=0), showlegend=False, name="BB Upper"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], line=dict(color="gray", width=0), fill="tonexty", fillcolor="rgba(128,128,128,0.1)", name="Bollinger Bands"))
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

# Analysis Tabs
tab_tech, tab_macro, tab_quant = st.tabs(["📉 Technicals", "🌍 Macro & News", "🧠 Quant & Forecast"])

with tab_tech:
    st.subheader("Technical Signal Breakdown")
    t1, t2, t3, t4 = st.columns(4)

    def signal_card(title, active, bullish, points, max_points):
        status_color = "#238636" if bullish else "#da3633"
        status_text = "BULLISH" if bullish else "BEARISH"
        if not active:
            status_color = "#484f58"
            status_text = "INACTIVE"
        st.markdown(f"""
        <div style="background-color: #161b22; padding: 15px; border-radius: 6px; border-left: 4px solid {status_color};">
            <div style="font-weight: bold; color: #f0f6fc;">{title}</div>
            <div style="color: {status_color}; font-size: 0.9rem; margin-top: 5px;">{status_text}</div>
            <div style="color: #8b949e; font-size: 0.8rem; margin-top: 5px;">Bullish Points: {points} / {max_points}</div>
        </div>
        """, unsafe_allow_html=True)

    trend_info = score_result["details"].get("Trend", {"points": 0, "max": 30, "bullish": False})
    momentum_info = score_result["details"].get("Momentum", {"points": 0, "max": 30, "bullish": False})
    vol_info = score_result["details"].get("Volatility", {"points": 0, "max": 25, "bullish": False})
    rsi_info = score_result["details"].get("RSI", {"points": 0, "max": 15, "bullish": False})

    with t1:
        signal_card("Trend (EMA vs SMA)", "SMA" in selected_indicators and "EMA" in selected_indicators, trend_info["bullish"], trend_info["points"], trend_info["max"])
    with t2:
        signal_card("Momentum (MACD)", "MACD" in selected_indicators, momentum_info["bullish"], momentum_info["points"], momentum_info["max"])
    with t3:
        signal_card("Volatility (BB)", "Bollinger Bands" in selected_indicators, vol_info["bullish"], vol_info["points"], vol_info["max"])
    with t4:
        signal_card("Relative Strength (RSI)", "RSI" in selected_indicators, rsi_info["bullish"], rsi_info["points"], rsi_info["max"])

    st.markdown("---")
    st.subheader("💡 Technical Analysis Summary")

    bullish_signals = []
    bearish_signals = []
    if "SMA" in selected_indicators and "EMA" in selected_indicators:
        (bullish_signals if trend_info["bullish"] else bearish_signals).append("Trend signal")
    if "MACD" in selected_indicators:
        (bullish_signals if momentum_info["bullish"] else bearish_signals).append("Momentum signal")
    if "Bollinger Bands" in selected_indicators:
        (bullish_signals if vol_info["bullish"] else bearish_signals).append("Volatility signal")
    if "RSI" in selected_indicators:
        (bullish_signals if rsi_info["bullish"] else bearish_signals).append("RSI signal")

    if position_type == "Long":
        if bearish_signals:
            st.warning(f"**⚠️ Signals against your long position:** {', '.join(bearish_signals)}.")
        else:
            st.success(f"**✅ Signals support your long position:** {', '.join(bullish_signals)}.")
    else:
        if bullish_signals:
            st.warning(f"**⚠️ Signals against your short position:** {', '.join(bullish_signals)}.")
        else:
            st.success(f"**✅ Signals support your short position:** {', '.join(bearish_signals)}.")

    if "RSI" in selected_indicators:
        st.markdown("---")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#a371f7", width=2), name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(color="#c9d1d9"), title="RSI Momentum")
        st.plotly_chart(fig_rsi, width="stretch")

with tab_macro:
    m_col1, m_col2 = st.columns([1, 1])
    base_ccy = currency_pair.split("/")[0]
    quote_ccy = currency_pair.split("/")[1]

    with m_col1:
        st.subheader("Economic Indicators")
        fund_service = FundamentalsService()
        fund_df = fund_service.get_comparison_df(base_ccy, quote_ccy)
        if fund_df is not None:
            st.dataframe(fund_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Data unavailable.")

    with m_col2:
        st.subheader("Major Headlines")
        news_service = NewsService()
        news_items = news_service.get_combined_news(base_ccy, quote_ccy)
        if news_items:
            for item in news_items:
                badge_color = "#1f6feb" if item["currency"] == base_ccy else "#238636"
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 5px; background-color: #161b22; border: 1px solid #30363d; margin-bottom: 10px;">
                    <span style="background-color: {badge_color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7em; font-weight: bold;">{item['currency']}</span>
                    <span style="color: #8b949e; font-size: 0.8em; margin-left: 8px;">{item['time']}</span>
                    <div style="margin-top: 5px; font-weight: 500; font-size: 0.9em; color: #c9d1d9;">{item['title']}</div>
                    <div style="color: #8b949e; font-size: 0.75em; margin-top: 2px;">Source: {item['source']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No headlines found.")

with tab_quant:
    st.subheader("ML Price Prediction (30 Days)")
    ml_input = df_full[df_full.index <= last_row.name].copy()

    if len(ml_input.dropna()) < 100:
        st.info("Select a wider range or refresh data to give the ML model enough history.")
    else:
        with st.spinner("Running Random Forest Model..."):
            ml_engine = MLEngine()
            forecast_df = ml_engine.train_and_predict(ml_input, forecast_days=30)

        if forecast_df is not None and not forecast_df.empty:
            fig_ml = go.Figure()
            fig_ml.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted_Close"], line=dict(color="#58a6ff", width=2), name="Forecast"))
            fig_ml.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Upper_Bound"], line=dict(width=0), showlegend=False))
            fig_ml.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Lower_Bound"], line=dict(width=0), fill="tonexty", fillcolor="rgba(88, 166, 255, 0.2)", name="95% Conf. Interval"))
            fig_ml.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(color="#c9d1d9"), xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_ml, width="stretch")

    st.markdown("---")
    st.subheader("Risk Score History")
    history_df = get_risk_scores(pair=currency_pair, months=None)
    if not history_df.empty:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) + timedelta(days=1)
        history_df = history_df[(history_df["timestamp"] >= start_dt) & (history_df["timestamp"] < end_dt)]

    if not history_df.empty and len(history_df) > 1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=history_df["timestamp"], y=history_df["risk_score"], mode="markers+lines", marker=dict(size=8, color="#58a6ff", symbol="circle"), line=dict(color="#58a6ff", width=1), name="Risk Score"))
        fig_hist.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(color="#c9d1d9"), xaxis=dict(title="Date", range=[pd.to_datetime(start_date), pd.to_datetime(end_date) + timedelta(days=1)]), yaxis=dict(title="Risk Score", range=[0, 100]))
        st.plotly_chart(fig_hist, width="stretch")
        st.metric("Avg Risk Score", f"{history_df['risk_score'].mean():.1f}")
    else:
        st.info("Not enough saved history inside the selected date range.")
