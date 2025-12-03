import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from ML_engine import MLEngine
from database import init_db, add_risk_score, get_risk_scores

# RAG Imports
try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Page Config
st.set_page_config(
    page_title="Gorilla Glue",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database
init_db()

# Custom CSS
st.markdown("""
    <style>
    /* Main Background and Text Color */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Card Styling */
    .css-1r6slb0 {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Metric Container Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    
    /* Custom Header Styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #58a6ff;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #8b949e;
        margin-bottom: 2rem;
    }
    
    /* Analysis Section Styling */
    .analysis-container {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363d;
        margin-top: 20px;
    }
    .score-high { color: #ff4b4b; }
    .score-med { color: #ffa700; }
    .score-low { color: #21c354; }
    </style>
    """, unsafe_allow_html=True)

# --- RAG LOGIC ---
@st.cache_resource
def load_and_vectorize_knowledge(api_key):
    if not RAG_AVAILABLE:
        return None
    
    try:
        loader = TextLoader("knowledge_base.txt")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Failed to initialize RAG: {e}")
        return None

def query_rag_chain(question, vector_store, api_key):
    if not vector_store:
        return "RAG system not initialized."
        
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
    
    template = """You are an expert currency hedging assistant. Answer the user's question ONLY based on the provided context. 
    If the answer is not in the context, politely state: 'I can only answer questions related to the FX hedging strategy.'
    
    Context:
    {context}
    
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(question)

# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.caption("Configure simulation parameters")
    
    st.markdown("---")
    
    st.subheader("🛡 HEDGE PARAMETERS")
    monthly_sales = st.number_input("Monthly Sales Estimate", value=100000, step=1000)
    current_forward_rate = st.number_input("Current Forward Rate", value=1.0950, format="%.4f")
    
    st.markdown("---")
    st.subheader("💼 CURRENT POSITION")
    position_type = st.radio("Position Type", ["Long", "Short"], index=0, help="Your current market position")

    st.markdown("---")
    st.subheader("💬 AI Assistant")
    
    # API Key Input
    api_key = st.text_input("OpenAI API Key", type="password", help="Required for the Chatbot")
    try:
        if not api_key and "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
    except:
        pass  # Secrets file not found or invalid, user will need to enter key manually
        
    if RAG_AVAILABLE and api_key:
        vector_store = load_and_vectorize_knowledge(api_key)
        
        # Chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about the strategy..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = query_rag_chain(prompt, vector_store, api_key)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    elif not RAG_AVAILABLE:
        st.warning("RAG dependencies not installed.")
    else:
        st.info("Enter API Key to enable Chatbot.")

# Main Content
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="main-header">FX Hedging Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced currency risk analysis and forecasting</div>', unsafe_allow_html=True)

# Controls Section (Above Chart)
c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    currency_pair = st.selectbox(
        "Currency Pair",
        ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD"],
        index=0
    )

with c2:
    time_frame = st.radio(
        "Time Frame",
        ["1m", "5m", "15m", "1h", "4h", "1d", "1wk"],
        index=3, # Default to 1h
        horizontal=True,
        label_visibility="collapsed"
    )

with c3:
    selected_indicators = st.multiselect(
        "Technical Indicators",
        ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"],
        default=["SMA", "EMA"],
        key="indicators_multiselect"
    )

# Data Fetching (YFinance)
@st.cache_data(ttl=3600)
def get_data(pair, period="5y", interval="1d"):
    # Map pair to Yahoo Finance ticker
    ticker_map = {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "USD/CHF": "CHF=X",
        "AUD/USD": "AUDUSD=X"
    }
    ticker = ticker_map.get(pair, "EURUSD=X")
    
    # Map time frame to interval
    # Note: YFinance has limits (e.g., 1h data only for last 730 days)
    # We will use the 'period' argument to control history length
    
    # For the chart, we might want a shorter window, but for ML we need 5y.
    # We'll fetch 5y daily data for ML, and potentially separate data for the chart if needed.
    # But to keep it simple, we'll use the same dataset if possible, or resample.
    
    # If user selects "1 Hour", we can't get 5y of 1h data.
    # So we will fetch 5y Daily for ML, and use the selected interval for the Chart if possible.
    
    # Strategy:
    # 1. Fetch 5y Daily data for ML (always).
    # 2. Fetch Chart Data based on user selection (e.g. 1h for last 1mo, or Daily for 5y).
    
    ml_data = yf.download(ticker, period="5y", interval="1d", progress=False)
    
    # Flatten MultiIndex columns if present (common in recent yfinance)
    if isinstance(ml_data.columns, pd.MultiIndex):
        ml_data.columns = ml_data.columns.get_level_values(0)

    # Chart Data
    # Map UI timeframes to yfinance intervals
    chart_interval_map = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "1h", # Resample later if needed, or just show 1h
        "1d": "1d",
        "1wk": "1wk"
    }
    chart_interval = chart_interval_map.get(time_frame, "1d")
    
    # Determine period based on interval to optimize fetch and ensure data availability
    # 1m data is limited to 7 days
    if time_frame == "1m":
        chart_period = "5d"
    elif time_frame in ["5m", "15m"]:
        chart_period = "1mo"
    elif time_frame in ["1h", "4h"]:
        chart_period = "3mo" # or 6mo
    else:
        chart_period = "2y" # Longer history for daily/weekly
    
    chart_data = yf.download(ticker, period=chart_period, interval=chart_interval, progress=False)
    
    # Flatten Chart Data too
    if isinstance(chart_data.columns, pd.MultiIndex):
        chart_data.columns = chart_data.columns.get_level_values(0)
    
    return ml_data, chart_data

ml_df, chart_df = get_data(currency_pair, interval=time_frame)

# Process Chart Data
if chart_df.empty:
    st.error("No data found for the selected pair/timeframe.")
    st.stop()

# Calculate Indicators for Chart
df = chart_df.copy()
# SMA
df['SMA'] = df['Close'].rolling(window=20).mean()
# EMA
df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
# Bollinger Bands
df['BB_Middle'] = df['Close'].rolling(window=20).mean()
df['BB_Std'] = df['Close'].rolling(window=20).std()
df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
# RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))
# MACD
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Live Rate Update
try:
    current_price = float(df['Close'].iloc[-1])
    prev_price = float(df['Close'].iloc[-2])
except (TypeError, ValueError):
    # Fallback if it's a Series/DataFrame that needs explicit conversion
    current_price = df['Close'].iloc[-1].item()
    prev_price = df['Close'].iloc[-2].item()

price_delta = current_price - prev_price

with col2:
    st.metric(label="Live Rate", value=f"{current_price:.4f}", delta=f"{price_delta:.4f}")

# Chart Section
st.markdown(f"### 📈 {currency_pair} Price Chart")

# Determine Subplots
# Row 1: Price (Candlestick) + Overlays
# Row 2: Volume (Optional, but good for TV style) - Let's overlay or put in small row
# Row 3+: Oscillators

rows = 2 # Price + Volume
row_heights = [0.7, 0.15] # Price, Volume
specs = [[{"secondary_y": False}], [{"secondary_y": False}]]

if "RSI" in selected_indicators:
    rows += 1
    row_heights.append(0.15)
    specs.append([{"secondary_y": False}])

if "MACD" in selected_indicators:
    rows += 1
    row_heights.append(0.15)
    specs.append([{"secondary_y": False}])

# Normalize row heights
total_height = sum(row_heights)
row_heights = [h/total_height for h in row_heights]

fig = make_subplots(
    rows=rows, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.02, 
    row_heights=row_heights
)

# Main Chart (Candlestick) - TradingView Colors
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='Price',
    increasing_line_color='#26a69a', # TV Green
    decreasing_line_color='#ef5350'  # TV Red
), row=1, col=1)

# Volume
colors = ['#ef5350' if row['Open'] - row['Close'] >= 0 else '#26a69a' for index, row in df.iterrows()]
fig.add_trace(go.Bar(
    x=df.index, 
    y=df['Volume'],
    name='Volume',
    marker_color=colors,
    opacity=0.5
), row=2, col=1)

# Overlays (SMA, EMA, BB)
if "SMA" in selected_indicators:
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA'], line=dict(color='#2962FF', width=1.5), name='SMA'), row=1, col=1)
if "EMA" in selected_indicators:
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], line=dict(color='#B71C1C', width=1.5), name='EMA'), row=1, col=1)
if "Bollinger Bands" in selected_indicators:
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dash'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dash'), name='BB Lower', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

# Subplots (RSI, MACD)
current_row = 3
if "RSI" in selected_indicators:
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#7E57C2', width=1.5), name='RSI'), row=current_row, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="gray", row=current_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="gray", row=current_row, col=1)
    fig.update_yaxes(title_text="RSI", row=current_row, col=1)
    current_row += 1

if "MACD" in selected_indicators:
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#2962FF', width=1.5), name='MACD'), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], line=dict(color='#FF6D00', width=1.5), name='Signal'), row=current_row, col=1)
    fig.add_bar(x=df.index, y=df['MACD']-df['Signal_Line'], name='Histogram', marker_color='#26a69a', row=current_row, col=1)
    fig.update_yaxes(title_text="MACD", row=current_row, col=1)

fig.update_layout(
    height=800 if rows > 2 else 600,
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor='#161b22',
    plot_bgcolor='#161b22',
    font=dict(color='#fafafa'),
    xaxis_rangeslider_visible=False,
    xaxis=dict(showgrid=True, gridcolor='#30363d', gridwidth=0.5),
    yaxis=dict(showgrid=True, gridcolor='#30363d', gridwidth=0.5),
    showlegend=True,
    dragmode='pan',
    hovermode='x unified'
)

st.plotly_chart(fig, width="stretch")

# --- HEDGING STRATEGY LOGIC ---

# Get latest values
last_row = df.iloc[-1]
prev_row = df.iloc[-2]

# Initialize dynamic scoring
bullish_points = 0
max_points = 0

# 1. Trend (30 pts) - Requires both SMA and EMA
# Bullish: EMA > SMA
trend_bullish = False
trend_active = False
if "SMA" in selected_indicators and "EMA" in selected_indicators:
    trend_active = True
    max_points += 30
    if last_row['EMA'] > last_row['SMA']:
        trend_bullish = True
        bullish_points += 30

# 2. Momentum (30 pts) - Requires MACD
# Bullish: MACD > Signal AND MACD > 0
momentum_bullish = False
momentum_active = False
if "MACD" in selected_indicators:
    momentum_active = True
    max_points += 30
    if (last_row['MACD'] > last_row['Signal_Line']) and (last_row['MACD'] > 0):
        momentum_bullish = True
        bullish_points += 30

# 3. Volatility (25 pts) - Requires Bollinger Bands
# Bullish: Price > BB Upper OR Middle Band Rising
volatility_bullish = False
volatility_active = False
if "Bollinger Bands" in selected_indicators:
    volatility_active = True
    max_points += 25
    bb_middle_rising = last_row['BB_Middle'] > prev_row['BB_Middle']
    if (last_row['Close'] > last_row['BB_Upper']) or bb_middle_rising:
        volatility_bullish = True
        bullish_points += 25

# 4. RSI (15 pts) - Requires RSI
# Bullish: RSI > 50 AND Rising (but < 70)
rsi_bullish = False
rsi_active = False
if "RSI" in selected_indicators:
    rsi_active = True
    max_points += 15
    rsi_rising = last_row['RSI'] > prev_row['RSI']
    if (last_row['RSI'] > 50) and rsi_rising and (last_row['RSI'] < 70):
        rsi_bullish = True
        bullish_points += 15

# Total Scores (Normalized)
if max_points > 0:
    bullish_score = int((bullish_points / max_points) * 100)
else:
    bullish_score = 0 # Default if no indicators selected

bearish_score = 100 - bullish_score

# Determine Risk Score based on Position
if position_type == "Long":
    risk_score = bearish_score
    risk_label = "Bearish Reversal Risk"
else:
    risk_score = bullish_score
    risk_label = "Bullish Reversal Risk"

# Determine Action
if max_points == 0:
    action = "N/A"
    action_desc = "Select indicators to calculate risk."
    color_class = "score-low"
elif risk_score >= 80:
    action = "Strong Hedge (75-100%)"
    action_desc = f"Sell {currency_pair.split('/')[0]} / Buy {currency_pair.split('/')[1]}" if position_type == "Long" else f"Buy {currency_pair.split('/')[0]} / Sell {currency_pair.split('/')[1]}"
    color_class = "score-high"
elif risk_score >= 60:
    action = "Moderate Hedge (50-75%)"
    action_desc = f"Sell {currency_pair.split('/')[0]} / Buy {currency_pair.split('/')[1]}" if position_type == "Long" else f"Buy {currency_pair.split('/')[0]} / Sell {currency_pair.split('/')[1]}"
    color_class = "score-med"
elif risk_score >= 40:
    action = "Partial Hedge (25-50%)"
    action_desc = "Maintain a small protective hedge."
    color_class = "score-med"
else:
    action = "No Hedge / Unwind"
    action_desc = "No action; allow the position to run."
    color_class = "score-low"

# Save current score to database for persistent 6-month history
add_risk_score(currency_pair, risk_score, action, position_type)

# --- DISPLAY ANALYSIS ---
st.markdown("---")
st.markdown('<div class="analysis-container">', unsafe_allow_html=True)

st.header("🛡 Hedging Strategy Analysis")
st.caption("Confluence-based technical model to quantify reversal risk.")

col_a1, col_a2 = st.columns([1, 2])

with col_a1:
    st.markdown(f"### {risk_label}")
    st.markdown(f"<h1 class='{color_class}'>{risk_score}/100</h1>", unsafe_allow_html=True)
    st.markdown(f"**Action:** {action}")
    st.info(action_desc)

with col_a2:
    st.subheader("Signal Breakdown")
    
    # Helper for contribution display
    def get_contribution(active, is_bullish, points):
        if not active:
            return "Inactive"
        
        # Calculate normalized points contribution
        # This is tricky because we normalized the total score.
        # We'll just show the raw points status for simplicity or "Risk" vs "Support"
        
        if position_type == "Long":
            # Risk if Bearish (not Bullish)
            if not is_bullish:
                return "⚠️ Risk"
            return "✅ Support"
        else:
            # Risk if Bullish
            if is_bullish:
                return "⚠️ Risk"
            return "✅ Support"

    st.markdown(f"""
    | Indicator | Weight | Status | Contribution |
    | :--- | :--- | :--- | :--- |
    | **Trend (EMA vs SMA)** | 30 | {"Active" if trend_active else "Inactive"} | {get_contribution(trend_active, trend_bullish, 30)} |
    | **Momentum (MACD)** | 30 | {"Active" if momentum_active else "Inactive"} | {get_contribution(momentum_active, momentum_bullish, 30)} |
    | **Volatility (BB)** | 25 | {"Active" if volatility_active else "Inactive"} | {get_contribution(volatility_active, volatility_bullish, 25)} |
    | **RSI** | 15 | {"Active" if rsi_active else "Inactive"} | {get_contribution(rsi_active, rsi_bullish, 15)} |
    """)

# Historical Score Chart
st.markdown("---")
st.subheader("📊 Risk Score History")
st.caption("Track how the risk score has evolved over time")

# Date range inputs
col_date1, col_date2 = st.columns(2)
with col_date1:
    # Default start date: 12 months ago
    default_start = datetime.now() - timedelta(days=365)
    start_date = st.date_input(
        "Start Date",
        value=default_start,
        max_value=datetime.now(),
        help="Select the start date for historical data"
    )
with col_date2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now(),
        help="Select the end date for historical data"
    )

# Retrieve all historical data from database for the current pair
# We'll filter by date range below
history_df = get_risk_scores(pair=currency_pair, months=None)

# Filter by date range if data exists
if not history_df.empty:
    # Convert date inputs to datetime for comparison
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + timedelta(days=1)  # Include the end date
    
    # Filter the dataframe
    history_df = history_df[
        (history_df['timestamp'] >= start_datetime) & 
        (history_df['timestamp'] < end_datetime)
    ]

if not history_df.empty and len(history_df) > 1:
    # Create the chart
    fig_history = go.Figure()
    
    # Add the score dots (markers only, no lines)
    fig_history.add_trace(go.Scatter(
        x=history_df['timestamp'],
        y=history_df['risk_score'],
        mode='markers',  # Changed from 'lines+markers' to 'markers' only
        name='Risk Score',
        marker=dict(size=10, color='#58a6ff', symbol='circle'),  # Larger dots
        hovertemplate='<b>Time:</b> %{x}<br><b>Risk Score:</b> %{y}/100<br><extra></extra>'
    ))
    
    # Add horizontal zones for risk levels
    fig_history.add_hrect(y0=0, y1=40, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Low Risk", annotation_position="right")
    fig_history.add_hrect(y0=40, y1=60, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="Moderate Risk", annotation_position="right")
    fig_history.add_hrect(y0=60, y1=80, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="High Risk", annotation_position="right")
    fig_history.add_hrect(y0=80, y1=100, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Critical Risk", annotation_position="right")
    
    fig_history.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='#161b22',
        plot_bgcolor='#161b22',
        font=dict(color='#fafafa'),
        xaxis=dict(
            showgrid=True, 
            gridcolor='#30363d',
            title="Time",
            range=[start_datetime, end_datetime]  # Dynamically set x-axis range
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#30363d',
            title="Risk Score",
            range=[0, 100]
        ),
        showlegend=False,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_history, use_container_width=True)
    
    # Show statistics
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        avg_score = history_df['risk_score'].mean()
        st.metric("Average Risk Score", f"{avg_score:.1f}/100")
    with col_stat2:
        max_score = history_df['risk_score'].max()
        st.metric("Peak Risk Score", f"{max_score}/100")
    with col_stat3:
        min_score = history_df['risk_score'].min()
        st.metric("Lowest Risk Score", f"{min_score}/100")
else:
    st.info("Score history will appear here as data accumulates over the next 6 months. Refresh the page or change parameters to see the chart update.")


st.markdown('</div>', unsafe_allow_html=True)

# --- ML PREDICTION SECTION ---
st.markdown("---")
st.header("🧠 ML Price Prediction (1 Month Forecast)")
st.caption("Random Forest model trained on 5 years of historical data.")

with st.spinner("Training ML model and generating forecast..."):
    ml_engine = MLEngine()
    forecast_df = ml_engine.train_and_predict(ml_df, forecast_days=30)

if forecast_df is not None:
    # Plot forecast
    fig_ml = go.Figure()
    
    # Prediction Line
    fig_ml.add_trace(go.Scatter(
        x=forecast_df['Date'], 
        y=forecast_df['Predicted_Close'], 
        line=dict(color='#58a6ff', width=2), 
        name='Predicted Price'
    ))
    
    # Confidence Interval (Upper)
    fig_ml.add_trace(go.Scatter(
        x=forecast_df['Date'], 
        y=forecast_df['Upper_Bound'], 
        line=dict(width=0), 
        mode='lines',
        showlegend=False,
        name='Upper Bound'
    ))
    
    # Confidence Interval (Lower) - Filled
    fig_ml.add_trace(go.Scatter(
        x=forecast_df['Date'], 
        y=forecast_df['Lower_Bound'], 
        line=dict(width=0), 
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(88, 166, 255, 0.2)',
        showlegend=True,
        name='Confidence Interval (95%)'
    ))
    
    fig_ml.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='#161b22',
        plot_bgcolor='#161b22',
        font=dict(color='#fafafa'),
        xaxis=dict(showgrid=True, gridcolor='#30363d', title="Date"),
        yaxis=dict(showgrid=True, gridcolor='#30363d', title="Price"),
        showlegend=True
    )
    
    st.plotly_chart(fig_ml, width="stretch")
    
    # ML Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Forecast End Price", f"{forecast_df['Predicted_Close'].iloc[-1]:.4f}")
    with m2:
        st.metric("Model MSE", f"{ml_engine.mse:.6f}")
    with m3:
        st.metric("Prediction Range", f"{forecast_df['Lower_Bound'].iloc[-1]:.4f} - {forecast_df['Upper_Bound'].iloc[-1]:.4f}")

else:
    st.warning("Not enough data to generate prediction.")