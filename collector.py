import os
import json
import datetime
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import google.generativeai as genai

# Setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

def get_nifty_data():
    nifty = yf.download("^NSEI", period="5d", interval="15m")
    return nifty

def calculate_indicators(df):
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    return df

def get_ai_sentiment():
    prompt = "Analyze current global market sentiment (US Futures, Gift Nifty, Crude, USDINR) and provide a Nifty bias score from -1 (Bearish) to 1 (Bullish). Return only the number."
    try:
        response = model.generate_content(prompt)
        return float(response.text.strip())
    except:
        return 0.0

def update_scorecard(history, actual_price, predicted_range):
    if not history: return 0
    last_pred = history[-1]
    low, high = last_pred['range']
    is_correct = low <= actual_price <= high
    hit_rate = (sum(h['correct'] for h in history) + (1 if is_correct else 0)) / (len(history) + 1)
    return round(hit_rate * 100, 2)

def main():
    df = get_nifty_data()
    df = calculate_indicators(df)
    current_price = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    sentiment = get_ai_sentiment()
    
    bias = "Neutral"
    if sentiment > 0.2: bias = "Bullish"
    elif sentiment < -0.2: bias = "Bearish"
    
    ranges = {
        "15m": [current_price - (atr*0.5), current_price + (atr*0.5)],
        "30m": [current_price - (atr*0.8), current_price + (atr*0.8)],
        "60m": [current_price - (atr*1.2), current_price + (atr*1.2)]
    }
    
    try:
        with open('history.json', 'r') as f: history = json.load(f)
    except: history = []
    
    hit_rate = update_scorecard(history, current_price, ranges["15m"])
    
    data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "price": round(current_price, 2),
        "bias": bias,
        "ranges": ranges,
        "hit_rate": hit_rate,
        "sentiment": sentiment
    }
    
    with open('data.json', 'w') as f: json.dump(data, f)
    history.append({"time": data["timestamp"], "range": ranges["15m"], "correct": True}) 
    with open('history.json', 'w') as f: json.dump(history[-100:], f)

if __name__ == "__main__":
    main()
