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
# ページ設定
# ---------------------------
st.set_page_config(page_title="A病院 待ち人数・待ち時間 予測", layout="wide")

# ---------------------------
# パス設定（repo直下に models/ を置く）
# ---------------------------
MODEL_DIR = Path(__file__).parent / "models"

COUNT_MODEL_PATH = MODEL_DIR / "model_A_timeseries.json"
COUNT_COLUMNS_PATH = MODEL_DIR / "columns_A_timeseries.json"

WAITTIME_MODEL_PATH = MODEL_DIR / "model_A_waittime_30min.json"
QUEUE_MODEL_PATH = MODEL_DIR / "model_A_queue_30min.json"
MULTI_COLUMNS_PATH = MODEL_DIR / "columns_A_multi_30min.json"


# ---------------------------
# ユーティリティ
# ---------------------------
def is_holiday_like(d: pd.Timestamp) -> bool:
    """病院の運用ルールに合わせた休日判定"""
    return (
        jpholiday.is_holiday(d)
        or d.weekday() >= 5
        or (d.month == 12 and d.day >= 29)
        or (d.month == 1 and d.day <= 3)
    )


def safe_set(df: pd.DataFrame, col: str, val):
    """その列が存在するときだけ代入（columnsズレ対策）"""
    if col in df.columns:
        df.at[0, col] = val


def onehot_if_exists(df: pd.DataFrame, colname: str):
    if colname in df.columns:
        df.at[0, colname] = 1


# ---------------------------
# モデル読み込み（Boosterで安定運用）
# ---------------------------
@st.cache_resource
def load_assets():
    # Booster で読む（XGBRegressor.load_modelの型チェック回避）
    count_booster = xgb.Booster()
    count_booster.load_model(str(COUNT_MODEL_PATH))

    wait_booster = xgb.Booster()
    wait_booster.load_model(str(WAITTIME_MODEL_PATH))

    queue_booster = xgb.Booster()
    queue_booster.load_model(str(QUEUE_MODEL_PATH))

    with open(COUNT_COLUMNS_PATH, "r", encoding="utf-8") as f:
        count_cols = json.load(f)

    with open(MULTI_COLUMNS_PATH, "r", encoding="utf-8") as f:
        multi_cols = json.load(f)

    return count_booster, wait_booster, queue_booster, count_cols, multi_cols


def predict_with_booster(booster: xgb.Booster, X: pd.DataFrame) -> float:
    dmat = xgb.DMatrix(X)
    return float(booster.predict(dmat)[0])


def simulate_day(
    target_date: pd.Timestamp,
    total_patients: int,
    weather_label: str,
    count_booster: xgb.Booster,
    wait_booster: xgb.Booster,
    queue_booster: xgb.Booster,
    count_cols: list,
    multi_cols: list,
) -> pd.DataFrame:
    """08:00〜18:00を30分刻みで逐次予測"""

    is_holiday_daily = is_holiday_like(target_date)
    prev_date = target_date - timedelta(days=1)
    is_prev_holiday = is_holiday_like(prev_date)

    # 08:00〜18:00（30分刻み）
    time_slots = pd.date_range(
        start=target_date.replace(hour=8, minute=0),
        end=target_date.replace(hour=18, minute=0),
        freq=pd.Timedelta(minutes=30),
    )

    results = []
    lags = {"lag_30min": 0.0, "lag_60min": 0.0, "lag_90min": 0.0}
    queue_at_start = 0

    # 天気カテゴリ（先頭文字を使う設計のまま踏襲）
    weather_first = weather_label[0] if isinstance(weather_label, str) and len(weather_label) > 0 else ""
    is_rain = 1 if "雨" in weather_label else 0
    is_snow = 1 if "雪" in weather_label else 0

    for ts in time_slots:
        # -----------------------
        # 1) 受付数（count_model）
        # -----------------------
        count_X = pd.DataFrame([np.zeros(len(count_cols))], columns=count_cols)

        safe_set(count_X, "hour", ts.hour)
        safe_set(count_X, "minute", ts.minute)
        safe_set(count_X, "is_first_slot", 1 if (ts.hour == 8 and ts.minute == 0) else 0)
        safe_set(count_X, "is_second_slot", 1 if (ts.hour == 8 and ts.minute == 30) else 0)

        safe_set(count_X, "is_holiday", int(is_holiday_daily))
        safe_set(count_X, "月", ts.month)
        safe_set(count_X, "週回数", (ts.day - 1) // 7 + 1)
        safe_set(count_X, "前日祝日フラグ", int(is_prev_holiday))
        safe_set(count_X, "total_outpatient_count", int(total_patients))

        # 天候（学習に気象実数列がある場合は、ここでは入れない＝0のまま）
        safe_set(count_X, "雨フラグ", is_rain)
        safe_set(count_X, "雪フラグ", is_snow)

        # 曜日one-hot
        onehot_if_exists(count_X, f"dayofweek_{ts.dayofweek}")

        # 天気カテゴリ one-hot
        if weather_first:
            onehot_if_exists(count_X, f"天気カテゴリ_{weather_first}")

        # lag / rolling
        safe_set(count_X, "lag_30min", lags["lag_30min"])
        safe_set(count_X, "lag_60min", lags["lag_60min"])
        safe_set(count_X, "lag_90min", lags["lag_90min"])
        safe_set(count_X, "rolling_mean_60min", (lags["lag_30min"] + lags["lag_60min"]) / 2.0)

        pred_reception = predict_with_booster(count_booster, count_X[count_cols])
        predicted_reception_count = max(0, int(round(pred_reception)))

        # -----------------------
        # 2) 待ち人数＆待ち時間（multi）
        # -----------------------
        multi_X = pd.DataFrame([np.zeros(len(multi_cols))], columns=multi_cols)

        safe_set(multi_X, "hour", ts.hour)
        safe_set(multi_X, "minute", ts.minute)
        safe_set(multi_X, "reception_count", predicted_reception_count)
        safe_set(multi_X, "queue_at_start_of_slot", queue_at_start)

        safe_set(multi_X, "is_holiday", int(is_holiday_daily))
        safe_set(multi_X, "月", ts.month)
        safe_set(multi_X, "週回数", (ts.day - 1) // 7 + 1)
        safe_set(multi_X, "前日祝日フラグ", int(is_prev_holiday))
        safe_set(multi_X, "total_outpatient_count", int(total_patients))

        safe_set(multi_X, "雨フラグ", is_rain)
        safe_set(multi_X, "雪フラグ", is_snow)

        onehot_if_exists(multi_X, f"dayofweek_{ts.dayofweek}")
        if weather_first:
            onehot_if_exists(multi_X, f"天気カテゴリ_{weather_first}")

        pred_queue = predict_with_booster(queue_booster, multi_X[multi_cols])
        pred_wait = predict_with_booster(wait_booster, multi_X[multi_cols])

        predicted_queue_size = max(0, int(round(pred_queue)))
        predicted_avg_wait = max(0, int(round(pred_wait)))

        # 保存
        results.append(
            {
                "時間帯": ts.strftime("%H:%M"),
                "予測受付数": predicted_reception_count,
                "予測待ち人数(人)": predicted_queue_size,
                "予測平均待ち時間(分)": predicted_avg_wait,
            }
        )

        # 状態更新
        lags = {
            "lag_30min": float(predicted_reception_count),
            "lag_60min": lags["lag_30min"],
            "lag_90min": lags["lag_60min"],
        }
        queue_at_start = predicted_queue_size

    return pd.DataFrame(results)


# ---------------------------
# UI
# ---------------------------
st.title("🏥 A病院 待ち人数・待ち時間 統合予測アプリ（Streamlit版）")

# 起動時チェック
missing = [p for p in [COUNT_MODEL_PATH, COUNT_COLUMNS_PATH, WAITTIME_MODEL_PATH, QUEUE_MODEL_PATH, MULTI_COLUMNS_PATH] if not p.exists()]
if missing:
    st.error("必要ファイルが見つかりません。`models/` 配下に置けているか確認してください。")
    st.write("見つからないファイル:")
    for p in missing:
        st.write(f"- {p}")
    st.stop()

count_booster, wait_booster, queue_booster, count_cols, multi_cols = load_assets()

with st.sidebar:
    st.header("入力")
    target_date = st.date_input("予測対象日", value=date.today() + timedelta(days=1))
    total_patients = st.number_input("延べ外来患者数", min_value=0, max_value=200000, value=1200, step=10)

    # 学習時の天気カテゴリの先頭文字（晴/曇/雨/雪/大 など）が出るようにしておく
    weather_label = st.selectbox("天気予報", ["晴", "曇", "雨", "雪", "快晴", "薄曇", "大雨", "大雪"], index=0)

    run = st.button("シミュレーション実行", type="primary")

st.caption("※ 08:00〜18:00 を30分刻みで逐次予測します。")

if run:
    td = pd.to_datetime(target_date)

    with st.spinner("計算中..."):
        df = simulate_day(
            td,
            int(total_patients),
            weather_label,
            count_booster,
            wait_booster,
            queue_booster,
            count_cols,
            multi_cols,
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"{td.strftime('%Y-%m-%d')} の予測（表）")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "CSVをダウンロード",
            data=csv,
            file_name=f"A_pred_{td.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

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

else:
    st.info("左の入力を設定して「シミュレーション実行」を押してください。")
