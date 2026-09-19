import os
import json
import datetime

from dotenv import load_dotenv
import yfinance as yf
import pandas_ta as ta
import google.generativeai as genai


# ============================================================
# Load environment variables
# ============================================================

load_dotenv("/workspaces/nifty-monitor/.env", override=True)

api_key = os.getenv("GEMINI_API_KEY", "").strip()

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY was not found. Check your /workspaces/nifty-monitor/.env file."
    )

if any(character.isspace() for character in api_key):
    raise RuntimeError(
        "GEMINI_API_KEY contains spaces or line breaks. Put the complete key on one line."
    )

print("Gemini key loaded:", api_key[:6] + "..." + api_key[-4:])


# ============================================================
# Configure Gemini
# ============================================================

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash")


# ============================================================
# Get Nifty data
# ============================================================

def get_nifty_data():
    print("Getting Nifty data...")

    nifty = yf.download(
        "^NSEI",
        period="5d",
        interval="15m",
        auto_adjust=False,
        progress=False
    )

    if nifty.empty:
        raise RuntimeError("No Nifty data was downloaded from Yahoo Finance.")

    return nifty


# ============================================================
# Calculate technical indicators
# ============================================================

def calculate_indicators(df):
    # yfinance can sometimes return multi-level columns.
    # Convert each required column to a normal Series.
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in df.columns:
            value = df[column]

            if hasattr(value, "ndim") and value.ndim > 1:
                value = value.iloc[:, 0]

            df[column] = value

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    df["RSI"] = ta.rsi(close, length=14)
    df["EMA_20"] = ta.ema(close, length=20)
    df["EMA_50"] = ta.ema(close, length=50)
    df["ATR"] = ta.atr(high, low, close, length=14)

    df = df.dropna()

    if df.empty:
        raise RuntimeError("Not enough data to calculate technical indicators.")

    return df


# ============================================================
# Get Gemini sentiment
# ============================================================

def get_ai_sentiment():
    prompt = """
Analyze the current market sentiment for India's Nifty index using the
information provided below.

Return only one decimal number between -1 and 1:
-1.0 means strongly bearish
 0.0 means neutral
 1.0 means strongly bullish

Do not include words or explanations.

Current Nifty technical data:
"""

    try:
        response = model.generate_content(prompt)

        response_text = response.text.strip()
        sentiment = float(response_text)

        # Keep the value safely within the expected range.
        sentiment = max(-1.0, min(1.0, sentiment))

        print("Gemini sentiment:", sentiment)

        return sentiment

    except Exception as error:
        print("Gemini error:", error)
        return 0.0


# ============================================================
# Calculate historical hit rate
# ============================================================

def update_scorecard(history, actual_price):
    if not history:
        return 0.0

    last_prediction = history[-1]

    predicted_range = last_prediction.get("range", [])

    if len(predicted_range) != 2:
        return 0.0

    low = float(predicted_range[0])
    high = float(predicted_range[1])

    is_correct = low <= actual_price <= high

    previous_correct = sum(
        1 for item in history if item.get("correct") is True
    )

    total_predictions = len(history) + 1

    hit_rate = (
        (previous_correct + (1 if is_correct else 0))
        / total_predictions
        * 100
    )

    return round(hit_rate, 2)


# ============================================================
# Main program
# ============================================================

def main():
    try:
        df = get_nifty_data()
        df = calculate_indicators(df)

        current_price = float(df["Close"].iloc[-1])
        atr = float(df["ATR"].iloc[-1])

        sentiment = get_ai_sentiment()

        if sentiment > 0.2:
            bias = "Bullish"
        elif sentiment < -0.2:
            bias = "Bearish"
        else:
            bias = "Neutral"

        ranges = {
            "15m": [
                round(current_price - (atr * 0.5), 2),
                round(current_price + (atr * 0.5), 2),
            ],
            "30m": [
                round(current_price - (atr * 0.8), 2),
                round(current_price + (atr * 0.8), 2),
            ],
            "60m": [
                round(current_price - (atr * 1.2), 2),
                round(current_price + (atr * 1.2), 2),
            ],
        }

        try:
            with open("history.json", "r", encoding="utf-8") as file:
                history = json.load(file)

            if not isinstance(history, list):
                history = []

        except (FileNotFoundError, json.JSONDecodeError):
            history = []

        hit_rate = update_scorecard(history, current_price)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        data = {
            "timestamp": timestamp,
            "price": round(current_price, 2),
            "bias": bias,
            "ranges": ranges,
            "hit_rate": hit_rate,
            "sentiment": sentiment,
            "atr": round(atr, 2),
        }

        with open("data.json", "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        # This prediction will be evaluated on the next run.
        history.append(
            {
                "time": timestamp,
                "range": ranges["15m"],
                "correct": False,
            }
        )

        with open("history.json", "w", encoding="utf-8") as file:
            json.dump(history[-100:], file, indent=2)

        print()
        print("DONE")
        print("Nifty:", round(current_price, 2))
        print("Bias:", bias)
        print("Sentiment:", sentiment)
        print("15-minute range:", ranges["15m"])
        print("30-minute range:", ranges["30m"])
        print("60-minute range:", ranges["60m"])
        print("Hit rate:", hit_rate, "%")
        print("Saved file: data.json")

    except Exception as error:
        print("Program error:", error)


# ============================================================
# Start program
# ============================================================

if __name__ == "__main__":
    main()
