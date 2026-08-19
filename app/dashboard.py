import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src.data import load_air_quality, load_config


st.set_page_config(page_title="Dự báo PM2.5 TP.HCM", layout="wide")
st.title("Dự báo PM2.5 giờ tiếp theo")
st.caption("Bản trình diễn nghiên cứu, không dùng thay cho hệ thống cảnh báo chính thức.")
config = load_config("configs/config.yaml")
data = load_air_quality(config)
station_column = config["data"]["station_column"]
timestamp_column = config["data"]["timestamp_column"]
target = config["data"]["target_column"]
station = st.selectbox("Trạm quan trắc", sorted(data[station_column].unique()))
history = data[data[station_column] == station].sort_values(timestamp_column).tail(24)

try:
    payload_history = history.copy()
    payload_history[timestamp_column] = payload_history[timestamp_column].astype(str)
    payload_history = payload_history.astype(object).where(pd.notna(payload_history), None)
    response = requests.post("http://localhost:8000/predict", json={"observations": payload_history.to_dict("records")}, timeout=10)
    response.raise_for_status()
    result = response.json()
    first, second, third = st.columns(3)
    first.metric("PM2.5 hiện tại", f"{result['current_pm25']:.1f} µg/m³")
    second.metric("Dự báo giờ tới", f"{result['predicted_pm25']:.1f} µg/m³")
    third.metric("Mức", result["level"])
    confidence = result.get("confidence")
    st.caption(f"Độ tin cậy tương đối: {'chưa xác định' if confidence is None else f'{confidence:.0%}'} · Cập nhật: {result['updated_at']}")
except requests.RequestException:
    st.warning("Không kết nối được API. Hãy chạy: uvicorn app.api:app --reload")

figure = px.line(history, x=timestamp_column, y=target, markers=True, title="24 giờ gần nhất")
st.plotly_chart(figure, use_container_width=True)
