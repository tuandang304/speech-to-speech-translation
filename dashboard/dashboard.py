import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(layout="wide", page_title="S2ST Dashboard")

# --- Helper Functions ---
def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None

# --- Paths ---
RESULTS_DIR = '../results/'
DATA_DIR = '../data/'
METRICS_PATH = os.path.join(RESULTS_DIR, 'metrics.json')
HISTORY_PATH = os.path.join(RESULTS_DIR, 'history.json')

# --- Main Dashboard ---
st.title("📊 Bảng Điều Khiển Dự Án S2ST")
st.markdown("Trực quan hóa kết quả huấn luyện và nghe các mẫu dịch.")

# --- Tab Layout ---
tab1, tab2 = st.tabs(["📈 Kết Quả Huấn Luyện", "🎧 Nghe Mẫu Dịch"])

with tab1:
    st.header("Chỉ Số Đánh Giá Mô Hình")
    metrics = load_json(METRICS_PATH)
    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("BLEU Score", f"{metrics.get('BLEU', 'N/A')}")
        col2.metric("Mean Opinion Score (MOS)", f"{metrics.get('MOS', 'N/A')}")
        col3.metric("F0 Correlation", f"{metrics.get('F0_corr', 'N/A')}")

        st.subheader("Lịch sử BLEU Score (dữ liệu giả)")
        chart_data = pd.DataFrame({
            'epoch': range(1, 11),
            'BLEU': [10, 12, 15, 18, 20, 22, 24, 25, 25.2, 25.5]
        })
        st.line_chart(chart_data.rename(columns={'epoch':'index'}).set_index('index'))
    else:
        st.warning(f"Không tìm thấy file `results/metrics.json`. Hãy chạy kịch bản huấn luyện để tạo file này.")

with tab2:
    st.header("Lịch Sử Dịch và Nghe Lại")
    history = load_json(HISTORY_PATH)
    
    if history:
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp', ascending=False)
        st.dataframe(df)

        st.subheader("Chọn một mẫu để nghe")
        
        # Create a display label for the selectbox
        df['display_label'] = df['input_filename'] + " → " + df['output_filename']
        selected_label = st.selectbox("Chọn bản dịch:", df['display_label'])
        
        selected_row = df[df['display_label'] == selected_label].iloc[0]
        
        output_audio_path = os.path.join(RESULTS_DIR, selected_row['output_filename'])
        input_audio_path = os.path.join(DATA_DIR, 'en', selected_row['input_filename']) # Assumes input is in data/en

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Âm thanh Gốc (English)")
            if os.path.exists(input_audio_path):
                st.audio(input_audio_path)
            else:
                st.error(f"Không tìm thấy file âm thanh gốc: `{input_audio_path}`")
        
        with col2:
            st.markdown("#### Âm thanh Đã Dịch (Vietnamese)")
            if os.path.exists(output_audio_path):
                st.audio(output_audio_path)
            else:
                st.error(f"Không tìm thấy file âm thanh đã dịch: `{output_audio_path}`")
    else:
        st.info("Không có lịch sử dịch nào. Hãy sử dụng WebApp để dịch một file âm thanh.")