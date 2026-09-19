import os
import json
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise Exception("GEMINI_API_KEY is missing in the .env file")

client = genai.Client(api_key=api_key)


def get_nifty():
    data = yf.download(
        "^NSEI",
        period="5d",
        interval="15m",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise Exception("Nifty data was not downloaded")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data["EMA20"] = data["Close"].ewm(span=20).mean()
    data["EMA50"] = data["Close"].ewm(span=50).mean()
    data["ATR"] = (
        data["High"] - data["Low"]
    ).rolling(14).mean()

    return data.dropna()


def get_other_markets():
    symbols = {
        "crude_oil": "CL=F",
        "usd_inr": "INR=X",
        "sp500_futures": "ES=F",
        "nasdaq_futures": "NQ=F"
    }

    result = {}

    for name, symbol in symbols.items():
        try:
            data = yf.download(
                symbol,
                period="2d",
                interval="1h",
                auto_adjust=False,
                progress=False
            )

            close = data["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            result[name] = round(float(close.dropna().iloc[-1]), 2)

        except Exception:
            result[name] = None

    return result


def get_moneycontrol_news():
    try:
        url = "https://www.moneycontrol.com/news/business/markets/"

        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
            timeout=15
        )

        soup = BeautifulSoup(response.text, "html.parser")

        news = []

        for item in soup.select("h2 a, h3 a")[:10]:
            title = item.get_text(" ", strip=True)
            link = item.get("href")

            if title and link:
                news.append({
                    "title": title,
                    "url": link
                })

        return news

    except Exception:
        return []


def get_sentiment(context):
    prompt = f"""
Analyze the Nifty market using the data below.

Return only one number between -1 and 1.

-1 = bearish
0 = neutral
1 = bullish

Data:
{json.dumps(context, indent=2, default=str)}
"""

    try:
        answer = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )

        value = float(answer.text.strip())

        return max(-1, min(1, value))

    except Exception as error:
        print("Gemini error:", error)
        return 0


def main():
    print("Getting Nifty data...")

    nifty = get_nifty()

    price = float(nifty["Close"].iloc[-1])
    atr = float(nifty["ATR"].iloc[-1])

    markets = get_other_markets()
    news = get_moneycontrol_news()

    context = {
        "nifty_price": price,
        "atr": atr,
        "markets": markets,
        "news": news
    }

    sentiment = get_sentiment(context)

    if sentiment > 0.2:
        bias = "Bullish"
    elif sentiment < -0.2:
        bias = "Bearish"
    else:
        bias = "Neutral"

    result = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nifty_price": round(price, 2),
        "bias": bias,
        "sentiment": sentiment,
        "ranges": {
            "15_minutes": [
                round(price - atr * 0.5, 2),
                round(price + atr * 0.5, 2)
            ],
            "30_minutes": [
                round(price - atr * 0.8, 2),
                round(price + atr * 0.8, 2)
            ],
            "60_minutes": [
                round(price - atr * 1.2, 2),
                round(price + atr * 1.2, 2)
            ]
        },
        "other_markets": markets,
        "moneycontrol_news": news
    }

    with open("data.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("")
    print("DONE")
    print("Nifty:", result["nifty_price"])
    print("Bias:", result["bias"])
    print("Sentiment:", result["sentiment"])
    print("Saved file: data.json")


if __name__ == "__main__":
    main()
