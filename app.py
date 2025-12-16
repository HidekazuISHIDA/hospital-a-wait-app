import json
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import jpholiday
import xgboost as xgb
import matplotlib.pyplot as plt

# ---------------------------
# 設定
# ---------------------------
st.set_page_config(page_title="A病院 待ち人数・待ち時間 予測", layout="wide")

MODEL_DIR = Path(__file__).parent / "models"
COUNT_MODEL_PATH = MODEL_DIR / "model_A_timeseries.json"
COUNT_COLUMNS_PATH = MODEL_DIR / "columns_A_timeseries.json"
WAITTIME_MODEL_PATH = MODEL_DIR / "model_A_waittime_30min.json"
QUEUE_MODEL_PATH = MODEL_DIR / "model_A_queue_30min.json"
MULTI_COLUMNS_PATH = MODEL_DIR / "columns_A_multi_30min.json"


@st.cache_resource
def load_models():
    # XGBoostモデル
    count_model = xgb.XGBRegressor()
    count_model.load_model(str(COUNT_MODEL_PATH))

    waittime_model = xgb.XGBRegressor()
    waittime_model.load_model(str(WAITTIME_MODEL_PATH))

    queue_model = xgb.XGBRegressor()
    queue_model.load_model(str(QUEUE_MODEL_PATH))

    # カラム
    with open(COUNT_COLUMNS_PATH, "r", encoding="utf-8") as f:
        count_cols = json.load(f)
    with open(MULTI_COLUMNS_PATH, "r", encoding="utf-8") as f:
        multi_cols = json.load(f)

    return count_model, waittime_model, queue_model, count_cols, multi_cols


def is_holiday_like(d: pd.Timestamp) -> bool:
    return (
        jpholiday.is_holiday(d)
        or d.weekday() >= 5
        or (d.month == 12 and d.day >= 29)
        or (d.month == 1 and d.day <= 3)
    )


def simulate_day(target_date: pd.Timestamp, total_patients: int, weather: str,
                 count_model, waittime_model, queue_model, count_cols, multi_cols):
    is_holiday_daily = is_holiday_like(target_date)
    prev_date = target_date - timedelta(days=1)
    is_prev_holiday = is_holiday_like(prev_date)

    time_slots = pd.date_range(
        start=target_date.replace(hour=8, minute=0),
        end=target_date.replace(hour=18, minute=0),
        freq="30T"
    )

    results = []
    lags = {"lag_30min": 0.0, "lag_60min": 0.0, "lag_90min": 0.0}
    queue_at_start = 0

    for ts in time_slots:
        # --- 1) 受付人数予測 features ---
        count_features = pd.DataFrame(columns=count_cols)
        count_features.loc[0] = 0

        # 必須の時間系
        if "hour" in count_cols: count_features.at[0, "hour"] = ts.hour
        if "minute" in count_cols: count_features.at[0, "minute"] = ts.minute
        if "is_first_slot" in count_cols:
            count_features.at[0, "is_first_slot"] = 1 if (ts.hour == 8 and ts.minute == 0) else 0
        if "is_second_slot" in count_cols:
            count_features.at[0, "is_second_slot"] = 1 if (ts.hour == 8 and ts.minute == 30) else 0

        # 日次
        if "月" in count_cols: count_features.at[0, "月"] = ts.month
        if "週回数" in count_cols: count_features.at[0, "週回数"] = (ts.day - 1) // 7 + 1
        if "前日祝日フラグ" in count_cols: count_features.at[0, "前日祝日フラグ"] = int(is_prev_holiday)
        if "total_outpatient_count" in count_cols: count_features.at[0, "total_outpatient_count"] = total_patients
        if "is_holiday" in count_cols: count_features.at[0, "is_holiday"] = int(is_holiday_daily)

        # 天気
        if "雨フラグ" in count_cols: count_features.at[0, "雨フラグ"] = 1 if "雨" in weather else 0
        if "雪フラグ" in count_cols: count_features.at[0, "雪フラグ"] = 1 if "雪" in weather else 0
        weather_cat_col = f"天気カテゴリ_{weather[0]}"
        if weather_cat_col in count_cols: count_features.at[0, weather_cat_col] = 1

        # 曜日（one-hot）
        dayofweek_col = f"dayofweek_{ts.dayofweek}"
        if dayofweek_col in count_cols: count_features.at[0, dayofweek_col] = 1

        # lag系
        rolling_mean = (lags["lag_30min"] + lags["lag_60min"]) / 2.0
        for lag_col, lag_val in lags.items():
            if lag_col in count_cols:
                count_features.at[0, lag_col] = lag_val
        if "rolling_mean_60min" in count_cols:
            count_features.at[0, "rolling_mean_60min"] = rolling_mean

        predicted_reception = float(count_model.predict(count_features[count_cols])[0])
        predicted_reception_count = max(0, int(round(predicted_reception)))

        # --- 2) 待ち人数・待ち時間 features ---
        multi_features = pd.DataFrame(columns=multi_cols)
        multi_features.loc[0] = 0

        if "hour" in multi_cols: multi_features.at[0, "hour"] = ts.hour
        if "minute" in multi_cols: multi_features.at[0, "minute"] = ts.minute
        if "reception_count" in multi_cols: multi_features.at[0, "reception_count"] = predicted_reception_count
        if "queue_at_start_of_slot" in multi_cols: multi_features.at[0, "queue_at_start_of_slot"] = queue_at_start

        if "月" in multi_cols: multi_features.at[0, "月"] = ts.month
        if "週回数" in multi_cols: multi_features.at[0, "週回数"] = (ts.day - 1) // 7 + 1
        if "前日祝日フラグ" in multi_cols: multi_features.at[0, "前日祝日フラグ"] = int(is_prev_holiday)
        if "total_outpatient_count" in multi_cols: multi_features.at[0, "total_outpatient_count"] = total_patients
        if "is_holiday" in multi_cols: multi_features.at[0, "is_holiday"] = int(is_holiday_daily)

        if "雨フラグ" in multi_cols: multi_features.at[0, "雨フラグ"] = 1 if "雨" in weather else 0
        if "雪フラグ" in multi_cols: multi_features.at[0, "雪フラグ"] = 1 if "雪" in weather else 0
        weather_cat_col_multi = f"天気カテゴリ_{weather[0]}"
        if weather_cat_col_multi in multi_cols: multi_features.at[0, weather_cat_col_multi] = 1

        dayofweek_col_multi = f"dayofweek_{ts.dayofweek}"
        if dayofweek_col_multi in multi_cols: multi_features.at[0, dayofweek_col_multi] = 1

        predicted_queue = float(queue_model.predict(multi_features[multi_cols])[0])
        predicted_queue_size = max(0, int(round(predicted_queue)))

        predicted_wait = float(waittime_model.predict(multi_features[multi_cols])[0])
        predicted_avg_wait_time = max(0, int(round(predicted_wait)))

        results.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": predicted_reception_count,
            "予測待ち人数(人)": predicted_queue_size,
            "予測平均待ち時間(分)": predicted_avg_wait_time,
        })

        # 状態更新
        lags = {"lag_30min": predicted_reception_count, "lag_60min": lags["lag_30min"], "lag_90min": lags["lag_60min"]}
        queue_at_start = predicted_queue_size

    return pd.DataFrame(results)


# ---------------------------
# UI
# ---------------------------
st.title("🏥 A病院 待ち人数・待ち時間 統合予測アプリ（Streamlit版）")

with st.sidebar:
    st.header("入力")
    target_date = st.date_input("予測対象日", value=date.today() + timedelta(days=1))
    total_patients = st.number_input("延べ外来患者数", min_value=0, max_value=100000, value=1200, step=10)
    weather = st.selectbox("天気予報", ["晴", "曇", "雨", "雪", "快晴", "薄曇"], index=0)
    run = st.button("シミュレーション実行")

st.caption("※ 予測は 08:00〜18:00 を30分刻みで表示します。")

count_model, waittime_model, queue_model, count_cols, multi_cols = load_models()

if run:
    td = pd.to_datetime(target_date)
    df = simulate_day(td, int(total_patients), weather,
                      count_model, waittime_model, queue_model, count_cols, multi_cols)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"{td.strftime('%Y-%m-%d')} の予測（表）")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("CSVをダウンロード", data=csv, file_name=f"A_pred_{td.strftime('%Y%m%d')}.csv", mime="text/csv")

    with col2:
        st.subheader("グラフ")
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.bar(df["時間帯"], df["予測待ち人数(人)"])
        ax1.set_xlabel("時間帯")
        ax1.set_ylabel("予測待ち人数(人)")
        ax1.tick_params(axis="x", rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(df["時間帯"], df["予測平均待ち時間(分)"], marker="o")
        ax2.set_ylabel("予測平均待ち時間(分)")
        fig.tight_layout()
        st.pyplot(fig)
