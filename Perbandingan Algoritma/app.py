import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse
import pickle

# ======================== Custom CSS Styling ========================
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #eef2f3, #ffffff);
        padding: 1rem;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stSelectbox, .stFileUploader, .stTextInput {
        background-color: #f8f9fa;
    }
    .css-1d391kg h1 {
        color: #003366;
    }
    </style>
""", unsafe_allow_html=True)

# ======================== Header ========================
st.markdown("""
    <h1 style='text-align: center;'>Perbandingan Algoritma Harga Bitcoin menggunakan data History</h1>
    <h4 style='text-align: center;'>Menggunakan Model LSTM & XGBoost</h4>
""", unsafe_allow_html=True)

# ======================== Upload Dataset / Load Default ========================
st.write("Upload dataset Bitcoin (format seperti BITCOIN2025_FORMAT.csv)")
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil diunggah!")
else:
    st.info("Tidak ada file diunggah, menggunakan dataset default dari GitHub.")
    url = "https://raw.githubusercontent.com/Nobody-slay/MyMachineLearning/refs/heads/main/Perbandingan%20Algoritma/BITCOIN2025_FORMAT.csv"
    try:
        df = pd.read_csv(url)
        st.success("Dataset berhasil dimuat dari GitHub.")
    except Exception as e:
        st.error(f"Gagal memuat dataset dari GitHub: {e}")
        st.stop()

# ======================== Preprocessing ========================
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

st.subheader("📈 Grafik Harga BTC (Close)")
st.line_chart(df['Close'])

data = df[['Close']].copy()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, window=10):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

WINDOW = 10
X_seq, y_seq = create_sequences(data_scaled, window=WINDOW)
split_index = int(len(X_seq) * 0.8)
X_train, y_train = X_seq[:split_index], y_seq[:split_index]
X_test, y_test = X_seq[split_index:], y_seq[split_index:]

# ======================== Model Selection ========================
st.sidebar.title("🔍 Pengaturan")
model_choice = st.sidebar.multiselect(
    "Pilih Model yang Ingin Digunakan",
    ["LSTM", "XGBoost"],
    default=["LSTM", "XGBoost"]
)

if st.sidebar.button("🔮 Jalankan Prediksi"):
    predictions = {}

    if "LSTM" in model_choice:
        try:
            model = load_model('Perbandingan Algoritma/model/lstm_model.h5', custom_objects={'mse': mse})
            lstm_pred_scaled = model.predict(X_test)
            predictions['LSTM'] = scaler.inverse_transform(lstm_pred_scaled).flatten()
            st.success("Model LSTM berhasil dijalankan.")
        except Exception as e:
            st.error(f"Gagal memuat atau menjalankan model LSTM: {e}")

    if "XGBoost" in model_choice:
        try:
            with open('Perbandingan Algoritma/model/xgboost_model.pkl', 'rb') as file:
                model = pickle.load(file)
            X_xgb = X_seq.reshape((X_seq.shape[0], -1))
            xgb_pred_scaled = model.predict(X_xgb[split_index:])
            predictions['XGBoost'] = scaler.inverse_transform(xgb_pred_scaled.reshape(-1, 1)).flatten()
            st.success("Model XGBoost berhasil dijalankan.")
        except Exception as e:
            st.error(f"Gagal memuat atau menjalankan model XGBoost: {e}")

    # ======================== Visualisasi ========================
    if predictions:
        st.subheader("📉 Prediksi vs Aktual - 30 Hari Terakhir")
        actual = data['Close'].iloc[-30:].values
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=actual, mode='lines+markers', name='Actual', line=dict(color='black')))

        for name, pred in predictions.items():
            fig.add_trace(go.Scatter(y=pred[-30:], mode='lines+markers', name=name))

        fig.update_layout(
            title="Perbandingan Prediksi Harga BTC (30 Hari Terakhir)",
            plot_bgcolor='rgba(245, 245, 255, 1)',
            paper_bgcolor='rgba(245, 245, 255, 1)',
            xaxis_title='Hari',
            yaxis_title='Harga BTC (USD)',
            legend_title="Model"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ======================== Evaluasi ========================
        st.subheader("📊 Evaluasi Model")
        model_metrics = {}

        for name, pred in predictions.items():
            rmse = np.sqrt(mean_squared_error(actual, pred[-30:]))
            mae = mean_absolute_error(actual, pred[-30:])
            mape = np.mean(np.abs((actual - pred[-30:]) / actual)) * 100
            r2 = r2_score(actual, pred[-30:])

            model_metrics[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2
            }

            st.markdown(f"**{name}**")
            st.write(f"- RMSE: `${rmse:,.2f}`")
            st.write(f"- MAE: `${mae:,.2f}`")
            st.write(f"- MAPE: `{mape:.2f}%`")
            st.write(f"- R² Score: `{r2:.4f}`")

        # ======================== Kesimpulan Otomatis ========================
        st.subheader("📝 Kesimpulan")
        if len(model_metrics) > 1:
            sorted_models = sorted(model_metrics.items(), key=lambda x: (x[1]['MAE'], -x[1]['R2']))
            best_model = sorted_models[0][0]
            best_stats = sorted_models[0][1]

            st.markdown(f"""
            Berdasarkan hasil evaluasi terhadap 30 hari terakhir, model **{best_model}** memberikan performa terbaik:

            - MAE terendah: **${best_stats['MAE']:,.2f}**
            - R² Score tertinggi: **{best_stats['R2']:.4f}**

            Hal ini menunjukkan bahwa model {best_model} mampu memprediksi harga Bitcoin dengan kesalahan kecil dan ketepatan yang baik terhadap variasi data aktual.
            """)
        else:
            st.markdown("Hanya satu model yang dipilih, sehingga tidak ada perbandingan performa.")
