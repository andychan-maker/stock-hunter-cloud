import streamlit as st
from finvizfinance.screener.overview import Overview
import pandas as pd
import os
import yfinance as yf
import gspread
import json
from datetime import datetime, timedelta

# --- CONFIGURATION ---
SHEET_NAME = "Stock Hunter History"
MAX_GAP_PERCENT = 0.10
MAX_RSI = 80
WICK_TOLERANCE = 0.6

st.set_page_config(page_title="Strong Stock Hunter: Cloud Edition", layout="wide")
st.title("🚀 Strong Stock Hunter: Cloud Edition")

# --- 1. FILTERS ---
filters_dict = {
    'Market Cap.': '+Small (over $300mln)',
    'Price': 'Over $10',
    'Average Volume': 'Over 500K',
    'Relative Volume': 'Over 1.5',
    'Industry': 'Stocks only (ex-Funds)',
    '50-Day Simple Moving Average': 'Price above SMA50',
    '200-Day Simple Moving Average': 'SMA200 below SMA50' 
}

# --- 2. GOOGLE SHEETS CONNECTION ---
def get_google_sheet():
    """Connects to Google Sheets using local file OR cloud secrets."""
    try:
        if os.path.exists("service_account.json"):
            gc = gspread.service_account(filename="service_account.json")
        else:
            if "textkey" in st.secrets:
                # We handle the potential "invisible newline" issue here too just in case
                secret_content = st.secrets["textkey"]
                # If users used triple-quotes, we ensure it's valid JSON
                try:
                    key_dict = json.loads(secret_content, strict=False)
                except json.JSONDecodeError:
                    # Fallback cleanup for common copy-paste errors
                    cleaned = secret_content.strip().replace('\n', '')
                    key_dict = json.loads(cleaned)
                
                gc = gspread.service_account_from_dict(key_dict)
            else:
                st.error("❌ No 'service_account.json' found and no Secrets configured.")
                st.stop()
            
        sh = gc.open(SHEET_NAME)
        return sh.get_worksheet(0)
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        st.stop()

# --- 3. HELPER FUNCTIONS ---
def clean_metric(value):
    if isinstance(value, (int, float)): return value
    if pd.isna(value) or value == '-': return 0
    value = str(value).strip().replace(',', '')
    if '%' in value: return float(value.replace('%', ''))
    suffixes = {'B': 1e9, 'M': 1e6, 'K': 1e3}
    if len(value) > 1 and value[-1].upper() in suffixes:
        try: return float(value[:-1]) * suffixes[value[-1].upper()]
        except: return 0
    try: return float(value)
    except: return 0

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def validate_stock_rules(df):
    tickers = df['Ticker'].tolist()
    valid_tickers = []
    if not tickers: return df
    status_text = st.empty()
    status_text.text(f"Validating Technical Rules for {len(tickers)} stocks...")
    try:
        data = yf.download(tickers, period="3mo", group_by='ticker', progress=False, threads=True)
        for ticker in tickers:
            try:
                if len(tickers) == 1: hist = data
                else: hist = data[ticker]
                clean_hist = hist[['Open', 'Close', 'High', 'Low']].dropna()
                if len(clean_hist) < 15: continue 
                
                # Rules
                high_val = clean_hist['Close'].max()
                curr_val = clean_hist['Close'].iloc[-1]
                if (high_val - curr_val) / high_val > 0.15: continue

                today_open = clean_hist['Open'].iloc[-1]
                yest_close = clean_hist['Close'].iloc[-2]
                if yest_close > 0 and (today_open - yest_close) / yest_close > MAX_GAP_PERCENT: continue

                today_high = clean_hist['High'].iloc[-1]
                today_close = clean_hist['Close'].iloc[-1]
                today_low = clean_hist['Low'].iloc[-1]
                if (today_high - today_low) > 0:
                    if (today_high - today_close) / (today_high - today_low) > WICK_TOLERANCE: continue

                rsi_series = calculate_rsi(clean_hist['Close'])
                if rsi_series.iloc[-1] > MAX_RSI: continue
                
                valid_tickers.append(ticker)
            except: continue
        status_text.empty()
        return df[df['Ticker'].isin(valid_tickers)]
    except: return df

def calculate_strength_score(df):
    max_vol = df['Volume'].max()
    df['Vol_Score'] = (df['Volume'] / max_vol * 50) if max_vol > 0 else 0
    df['Price_Score'] = df['Change'].clip(0, 20) * 2.5 
    df['Total_Score'] = (df['Vol_Score'] + df['Price_Score']).round(1)
    return df.sort_values(by='Total_Score', ascending=False)

def show_mini_charts(top_stocks_list):
    st.subheader("📈 Trend Check: Top 3 Strongest")
    cols = st.columns(3) 
    for i, ticker in enumerate(top_stocks_list[:3]):
        with cols[i]:
            st.write(f"**{ticker}**")
            try:
                data = yf.download(ticker, period="1mo", interval="1d", progress=False)
                if not data.empty: st.line_chart(data['Close'], height=150)
            except: st.write("No chart")

# --- 4. MAIN APP ---
tab1, tab2 = st.tabs(["🔎 Daily Scanner", "📊 History & Stats"])

with tab1:
    st.header("Today's Scan")
    if st.button("Run Scan & Upload to Cloud"):
        with st.spinner('Scanning & Analyzing...'):
            try:
                # 1. Finviz
                foverview = Overview()
                foverview.set_filter(filters_dict=filters_dict)
                df = foverview.screener_view(order='Change', ascend=False)
                
                if not df.empty:
                    # 2. Process
                    for col in ['Price', 'Change', 'Volume']:
                        df[col] = df[col].apply(clean_metric)
                    
                    df = validate_stock_rules(df)
                    
                    if not df.empty:
                        df = calculate_strength_score(df)
                        
                        # --- 3. FIX TIMEZONE & FORMAT ---
                        # Get UTC time and add 8 hours for Hong Kong
                        utc_now = datetime.utcnow()
                        hk_time = utc_now + timedelta(hours=8)
                        
                        # Save format as "YYYY-MM-DD HH:MM"
                        today_str_full = hk_time.strftime('%Y-%m-%d %H:%M')
                        today_date_only = hk_time.strftime('%Y-%m-%d') # Used for duplicate check

                        # Assign the full timestamp to the Date column
                        df['Date'] = today_str_full
                        final_df = df[['Date', 'Ticker', 'Sector', 'Price', 'Change', 'Volume', 'Total_Score']].copy()
                        
                        # --- 4. SMART UPLOAD ---
                        sheet = get_google_sheet()
                        existing_data = sheet.get_all_records()
                        existing_df = pd.DataFrame(existing_data)
                        
                        existing_keys = set()
                        if not existing_df.empty:
                            existing_df['Date'] = existing_df['Date'].astype(str)
                            existing_df['Ticker'] = existing_df['Ticker'].astype(str)
                            # We only check the DATE part (first 10 chars) to prevent duplicates on the same day
                            # e.g. "2026-02-19 01:05" becomes "2026-02-19"
                            existing_keys = set(zip(existing_df['Date'].str[:10], existing_df['Ticker']))
                        
                        data_to_upload = []
                        duplicates_count = 0
                        
                        for index, row in final_df.iterrows():
                            # Check based on DATE ONLY (ignore time for duplicate check)
                            check_key = (today_date_only, str(row['Ticker']))
                            
                            if check_key not in existing_keys:
                                data_to_upload.append(row.astype(str).tolist())
                            else:
                                duplicates_count += 1
                        
                        if data_to_upload:
                            sheet.append_rows(data_to_upload)
                            st.success(f"✅ Success! Added {len(data_to_upload)} NEW stocks at {today_str_full} (HKT).")
                            if duplicates_count > 0:
                                st.info(f"(Skipped {duplicates_count} duplicates)")
                        else:
                            st.warning("⚠️ No new stocks. (Already scanned today!)")

                        show_mini_charts(final_df['Ticker'].tolist())
                        st.dataframe(final_df)
                    else:
                        st.warning("Stocks found, but failed technical rules.")
                else:
                    st.warning("No stocks found.")
            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    st.header("Historical Data (Live from Cloud)")
    if st.button("Refresh History"):
        try:
            sheet = get_google_sheet()
            data = sheet.get_all_records()
            history_df = pd.DataFrame(data)
            
            if not history_df.empty:
                # Convert string dates to datetime objects
                history_df['Date'] = pd.to_datetime(history_df['Date'])
                
                st.subheader("🔥 Hot Stocks (Frequency Count)")
                col1, col2 = st.columns(2)
                
                now = datetime.now()
                week_ago = now - timedelta(days=7)
                month_ago = now - timedelta(days=30)
                
                with col1:
                    st.write("**Top Appearances (Last 7 Days)**")
                    week_df = history_df[history_df['Date'] >= week_ago]
                    if not week_df.empty:
                        st.dataframe(week_df['Ticker'].value_counts(), use_container_width=True)
                    else: st.info("No data in last 7 days.")

                with col2:
                    st.write("**Top Appearances (Last 30 Days)**")
                    month_df = history_df[history_df['Date'] >= month_ago]
                    if not month_df.empty:
                        st.dataframe(month_df['Ticker'].value_counts(), use_container_width=True)
                    else: st.info("No data in last 30 days.")
                
                st.divider()
                st.write(f"**Full History Log ({len(history_df)} records)**")
                # Format the date column nicely for display
                display_df = history_df.copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(display_df.sort_values(by='Date', ascending=False), use_container_width=True)
                
            else:
                st.info("Sheet is empty.")
        except Exception as e:
            st.error(f"Could not load history: {e}")