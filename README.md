# Discrete Unit-Based Speech-to-Speech Translation (EN-VI)

Dự án này triển khai một pipeline Speech-to-Speech Translation (S2ST) hoàn chỉnh từ tiếng Anh sang tiếng Việt, dựa trên phương pháp mã hóa âm thanh thành các đơn vị rời rạc (discrete units).

## Pipeline Tổng thể
English Audio -> Feature Extraction (HuBERT) -> Quantization (VQ-VAE) -> Translation (Seq2Seq Transformer) -> Vocoder (HiFi-GAN) -> Vietnamese Audio

## Cách chạy dự án

### Bước 0: Cài đặt
```bash
pip install -r requirements.txt
Bước 1: Tạo dữ liệu giả (Dummy Data)
Dự án cần các file audio song song trong data/en và data/vie. Script sau sẽ tạo ra các file âm thanh đơn giản để có thể chạy thử.
code
Bash
python training/generate_dummy_data.py
Bước 2: Tiền xử lý dữ liệu
Script này sẽ trích xuất các đơn vị rời rạc (discrete units) từ dữ liệu âm thanh, chuẩn bị cho việc huấn luyện Translator và Vocoder.
code
Bash
python training/preprocess_data.py
Lưu ý: Script này yêu cầu các checkpoint của Feature Extractor và Quantizer. Nó sẽ tự động gọi các script train tương ứng nếu checkpoint không tồn tại.
Bước 3: Huấn luyện từng module
Chạy các script sau theo thứ tự. Các script sẽ tạo ra các file checkpoint trong thư mục checkpoints/.
code
Bash
# 1. "Train" Feature Extractor (tải model pre-trained)
python training/train_feature_extractor.py

# 2. Train Quantizer (sử dụng dữ liệu thật nếu có)
python training/train_quantizer.py

# 3. Train Translator (dựa trên units đã tiền xử lý)
python training/train_translator.py

# 4. "Train" Vocoder (tải model pre-trained)
python training/train_vocoder.py
Bước 4: Chạy Inference trên Command Line
Dịch một file audio từ tiếng Anh sang tiếng Việt.
code
Bash
python main.py --input_audio data/en/sample_0.wav --output_audio translated.wav
Bước 5: Chạy Web Application
Khởi động server Flask để sử dụng giao diện web.
code
Bash
python webapp/app.py
Truy cập http://127.0.0.1:5000 trên trình duyệt của bạn, upload file .wav tiếng Anh và nhận kết quả.