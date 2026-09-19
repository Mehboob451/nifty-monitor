import json
import streamlit as st


st.set_page_config(
    page_title="Nifty Monitor",
    layout="wide"
)


st.title("Nifty Market Monitor")


try:
    with open("data.json", "r", encoding="utf-8") as file:
        data = json.load(file)

except FileNotFoundError:
    st.error("data.json was not found.")
    st.write("Run this command first:")
    st.code("python main.py")
    st.stop()


column1, column2, column3 = st.columns(3)


column1.metric(
    "Nifty Price",
    data.get("price", "N/A")
)

column2.metric(
    "Market Bias",
    data.get("bias", "N/A")
)

column3.metric(
    "Hit Rate",
    f'{data.get("hit_rate", 0)}%'
)


st.subheader("Predicted Ranges")
st.json(data.get("ranges", {}))


st.subheader("AI Sentiment")
st.write(data.get("sentiment", "N/A"))


st.subheader("External Markets")
st.json(data.get("external_markets", {}))


st.subheader("Moneycontrol Headlines")

news = data.get("moneycontrol_news", [])

if not news:
    st.write("No headlines found.")

for article in news:
    st.write(f'[{article["title"]}]({article["url"]})')
