import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "trades.db"

def init_db():
    """
    Initializes the database with a trades table and risk_scores table.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            pair TEXT,
            action TEXT,
            price REAL,
            amount REAL,
            notes TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS risk_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            pair TEXT,
            risk_score INTEGER,
            action TEXT,
            position_type TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_trade(pair, action, price, amount, notes=""):
    """
    Adds a trade to the database.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO trades (date, pair, action, price, amount, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (date, pair, action, price, amount, notes))
    conn.commit()
    conn.close()

def get_trades():
    """
    Retrieves all trades from the database.
    """
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM trades", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def add_risk_score(pair, risk_score, action, position_type, timestamp=None):
    """
    Adds a risk score entry to the database.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute('''
        INSERT INTO risk_scores (timestamp, pair, risk_score, action, position_type)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, pair, risk_score, action, position_type))
    conn.commit()
    conn.close()

def get_risk_scores(pair=None, months=None):
    """
    Retrieves risk scores from the database.
    If pair is specified, filters by currency pair.
    If months is specified, retrieves data from the past N months.
    If months is None, retrieves all historical data.
    """
    conn = sqlite3.connect(DB_NAME)
    try:
        from datetime import timedelta
        
        if months is not None:
            # Calculate the cutoff date
            cutoff_date = (datetime.now() - timedelta(days=months*30)).strftime("%Y-%m-%d %H:%M:%S")
            
            if pair:
                query = """
                    SELECT timestamp, pair, risk_score, action, position_type 
                    FROM risk_scores 
                    WHERE timestamp >= ? AND pair = ?
                    ORDER BY timestamp ASC
                """
                df = pd.read_sql_query(query, conn, params=(cutoff_date, pair))
            else:
                query = """
                    SELECT timestamp, pair, risk_score, action, position_type 
                    FROM risk_scores 
                    WHERE timestamp >= ?
                    ORDER BY timestamp ASC
                """
                df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        else:
            # Retrieve all data
            if pair:
                query = """
                    SELECT timestamp, pair, risk_score, action, position_type 
                    FROM risk_scores 
                    WHERE pair = ?
                    ORDER BY timestamp ASC
                """
                df = pd.read_sql_query(query, conn, params=(pair,))
            else:
                query = """
                    SELECT timestamp, pair, risk_score, action, position_type 
                    FROM risk_scores 
                    ORDER BY timestamp ASC
                """
                df = pd.read_sql_query(query, conn)
        
        # Convert timestamp to datetime
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        print(f"Error retrieving risk scores: {e}")
        df = pd.DataFrame()
    conn.close()
    return df
