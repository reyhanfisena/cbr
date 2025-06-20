import streamlit as st
import pandas as pd
from cbr_engine import CBREngine
import datetime
import os

# --- PATH KONSTAN SESUAI STRUKTUR ---
DATA_FILE_PATH = 'knowledge_base/data_historis.csv'
LOG_DIR = 'logs'
LOG_FILE_PATH = os.path.join(LOG_DIR, 'prediksi.log')
os.makedirs(LOG_DIR, exist_ok=True)

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Riset CBR", layout="wide")
st.title("🔬 Laboratorium Riset CBR Adaptif & Probabilistik")
st.markdown("Sebuah platform interaktif untuk prediksi harga dengan **Pembelajaran dari Pengalaman**.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Konfigurasi Model")
    if 'k_value' not in st.session_state: st.session_state.k_value = 7
    if 'window_size' not in st.session_state: st.session_state.window_size = 5
    if 'lambda_val' not in st.session_state: st.session_state.lambda_val = 0.5

    if st.button("Cari K Optimal"):
        with st.spinner("Menjalankan validasi..."):
            try:
                engine = CBREngine(DATA_FILE_PATH, LOG_FILE_PATH)
                k_opts = [k for k in range(3, 16, 2)]
                optimal_k = engine.find_optimal_k(st.session_state.window_size, k_opts, 30, st.session_state.lambda_val)
                if optimal_k: st.session_state.k_value = optimal_k
                st.success(f"K Optimal ditemukan: {st.session_state.k_value}")
            except Exception as e:
                st.error(f"Error: {e}", icon="🔥")

    st.session_state.k_value = st.slider("Jumlah Tetangga (K)", 1, 20, st.session_state.k_value)
    st.session_state.window_size = st.slider("Ukuran Window (Pola Hari)", 2, 30, st.session_state.window_size)
    st.session_state.lambda_val = st.slider("Sensitivitas Error (λ)", 0.0, 2.0, st.session_state.lambda_val, 0.1)

# --- MAIN INTERFACE ---
st.header("🎯 Target Prediksi & Simulasi")
target_date = st.date_input("Pilih tanggal", datetime.date.today() + datetime.timedelta(days=1))

if st.button("Jalankan Prediksi / Simulasi", type="primary", use_container_width=True):
    train_end_date = target_date - datetime.timedelta(days=1)
    try:
        engine = CBREngine(DATA_FILE_PATH, LOG_FILE_PATH, end_date=train_end_date.strftime('%Y-%m-%d'))
        result = engine.predict(k=st.session_state.k_value, window_size=st.session_state.window_size, lambda_val=st.session_state.lambda_val)

        pred, std_dev, interval = result['prediction'], result['uncertainty_std_dev'], result['prediction_interval_95']
        st.subheader(f"Hasil Analisis untuk {target_date.strftime('%d %B %Y')}")

        if std_dev > (pred * 0.02):
            st.warning(f"**Peringatan Ketidakpastian Tinggi:** Sebaran data kasus terdekat beragam (Std Dev: {std_dev:,.2f}).", icon="⚠️")

        col1, col2, col3 = st.columns(3)
        col1.metric("🔮 Prediksi Harga (IDR)", f"{pred:,.2f}")
        col2.metric("📏 Rentang Prediksi 95%", f"{interval[0]:,.0f} - {interval[1]:,.0f}")
        col3.metric("🌀 Entropi (Ketidakpastian)", f"{result['entropy']:.4f}")

        if target_date <= datetime.date.today():
            full_df = pd.read_csv(DATA_FILE_PATH, encoding='utf-8-sig')
            full_df['Tanggal'] = pd.to_datetime(full_df['Tanggal'], format='%d/%m/%Y').dt.date
            actual_row = full_df[full_df['Tanggal'] == target_date]

            if not actual_row.empty:
                actual_price = float(str(actual_row.iloc[0]['Terakhir']).replace('.', '', regex=False).replace(',', '.', regex=False))
                error = pred - actual_price
                st.metric("✔️ Harga Aktual", f"{actual_price:,.2f}", delta=f"Error: {error:,.2f}", delta_color="inverse")

                log_data = {
                    'Tanggal_Prediksi': target_date.strftime('%Y-%m-%d'),
                    'Harga_Aktual': actual_price,
                    'Harga_Prediksi': pred,
                    'Error_Absolut': abs(error),
                    'K': st.session_state.k_value,
                    'Window_Size': st.session_state.window_size,
                    'Lambda': st.session_state.lambda_val,
                    'Uncertainty_StdDev': std_dev
                }

                log_df = pd.DataFrame([log_data])
                if not os.path.exists(LOG_FILE_PATH) or os.path.getsize(LOG_FILE_PATH) == 0:
                    log_df.to_csv(LOG_FILE_PATH, index=False, encoding='utf-8-sig')
                else:
                    log_df.to_csv(LOG_FILE_PATH, mode='a', header=False, index=False, encoding='utf-8-sig')

                st.success("✔️ Pengalaman berhasil disimpan ke log!")
                st.dataframe(log_df)
            else:
                st.warning("❗ Tidak ditemukan harga aktual untuk tanggal tersebut.")

        with st.expander("Lihat Detail Kasus Terdekat"):
            st.dataframe(result['k_neighbors_info'])

    except Exception as e:
        st.error(f"Error: {e}", icon="🚨")

# --- RETAIN FAKTUAL ---
with st.expander("🧠 Retain Faktual: Tambah Data Baru ke Basis Pengetahuan"):
    with st.form("retain_form", clear_on_submit=True):
        new_date = st.date_input("Tanggal Data Baru")
        new_close = st.number_input("Harga Terakhir", format="%.2f")
        new_open = st.number_input("Harga Pembukaan", format="%.2f")
        new_high = st.number_input("Harga Tertinggi", format="%.2f")
        new_low = st.number_input("Harga Terendah", format="%.2f")

        if st.form_submit_button("Simpan Data Faktual Baru"):
            new_data = {
                'Tanggal': new_date.strftime('%d/%m/%Y'),
                'Terakhir': str(new_close).replace('.', ','),
                'Pembukaan': str(new_open).replace('.', ','),
                'Tertinggi': str(new_high).replace('.', ','),
                'Terendah': str(new_low).replace('.', ','),
                'Vol.': '-', 'Perubahan%': '-'
            }
            try:
                df = pd.read_csv(DATA_FILE_PATH, encoding='utf-8-sig')
                df_new = pd.DataFrame([new_data])
                df_concat = pd.concat([df, df_new], ignore_index=True)
                df_concat['Tanggal'] = pd.to_datetime(df_concat['Tanggal'], format='%d/%m/%Y')
                df_concat = df_concat.drop_duplicates(subset=['Tanggal'], keep='last').sort_values('Tanggal')
                df_concat['Tanggal'] = df_concat['Tanggal'].dt.strftime('%d/%m/%Y')
                df_concat.to_csv(DATA_FILE_PATH, index=False, encoding='utf-8-sig')
                st.success(f"Data untuk {new_date.strftime('%d-%m-%Y')} berhasil disimpan!")
            except Exception as e:
                st.error(f"Gagal menyimpan: {e}")

# --- TAMPILKAN LOG PENGALAMAN ---
st.markdown("---")
st.header("📜 Log Pengalaman Sistem")

if os.path.exists(LOG_FILE_PATH) and os.path.getsize(LOG_FILE_PATH) > 0:
    try:
        df_log = pd.read_csv(LOG_FILE_PATH, encoding='utf-8-sig')
        if not df_log.empty:
            st.dataframe(df_log)
        else:
            st.warning("📭 File log ditemukan tapi kosong.")
    except Exception as e:
        st.error(f"❌ Gagal membaca log: {e}")
else:
    st.info("📄 Log masih kosong. Jalankan prediksi dan simpan hasilnya untuk melihat data di sini.")
