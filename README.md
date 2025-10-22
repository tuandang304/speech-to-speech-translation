# 🎙️ S2ST Project: Dịch Nói Trực Tiếp Anh ↔ Việt

Dự án này xây dựng một hệ thống **Speech-to-Speech Translation (S2ST)** từ **tiếng Anh sang tiếng Việt**.  
Hệ thống được thiết kế theo **kiến trúc module hóa**, cho phép huấn luyện và tinh chỉnh từng phần riêng biệt, với mục tiêu cuối cùng là **một ứng dụng web dễ sử dụng**.

---

## ✨ Tính Năng Chính

- **Pipeline End-to-End:** Dịch trực tiếp từ file âm thanh tiếng Anh sang âm thanh tiếng Việt.  
- **Kiến trúc Module hóa:** Các thành phần Encoder, Quantizer, Translator, Vocoder có thể được huấn luyện và thay thế độc lập.  
- **WebApp Tự Động:** Giao diện web đơn giản, chỉ cần tải file lên, hệ thống sẽ tự động dịch và phát kết quả.  
- **Dashboard Trực Quan:** Theo dõi các chỉ số huấn luyện và nghe lại các mẫu dịch để đánh giá chất lượng.  
- **Đóng gói Dễ Dàng:** Hỗ trợ tạo file thực thi (.exe) để triển khai dễ dàng.  

---

## 🏗️ Kiến trúc Pipeline

```
[Audio EN .wav] -> [HuBERT Encoder] -> [Features] -> [Quantizer] -> [Units EN] -> [Translator] -> [Units VN] -> [Vocoder] -> [Audio VN .wav]
```

---

## 🚀 Hướng Dẫn Bắt Đầu Nhanh

### 1. Yêu Cầu Hệ Thống
- Python 3.8+  
- Git  
- (Khuyến nghị) GPU NVIDIA với CUDA để huấn luyện và tăng tốc inference.

### 2. Cài Đặt

**Bước 1:** Clone repository
```bash
git clone <URL_CUA_BAN_DEN_PROJECT_NAY>
cd Project
```

**Bước 2:** Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

**Bước 3:** Cài đặt thư viện cần thiết
```bash
pip install -r requirements.txt
```
> Lưu ý: Cài đặt PyTorch có thể mất vài phút.

---

## 💻 Hướng Dẫn Chạy WebApp (Dành cho Người Dùng Cuối)

Bạn có thể chạy ứng dụng web **mà không cần huấn luyện lại mô hình** (giả sử đã có checkpoint).

**Bước 1:** Khởi động server
```bash
python webapp/backend/app.py
```
> Server sẽ chạy trên cổng **5001** theo mặc định.

**Bước 2:** Mở trình duyệt
```
http://127.0.0.1:5001
```

**Bước 3:** Sử dụng ứng dụng
- Nhấn **"Chọn File Âm Thanh"**
- Tải file `.wav` tiếng Anh từ máy tính
- Hệ thống sẽ **tự động dịch và phát âm thanh tiếng Việt**

---

## 🛠️ Hướng Dẫn Huấn Luyện Mô Hình (Dành cho Lập Trình Viên)

### Bước 1: Chuẩn Bị Dữ Liệu
Cấu trúc dữ liệu:
```
data/
├── en/    # chứa file .wav tiếng Anh
└── vn/    # chứa file .wav tiếng Việt tương ứng
```

Ví dụ:
```
data/en/cau_chuyen_01.wav
data/vn/cau_chuyen_01.wav
```

### Bước 2: Trích Xuất Đặc Trưng HuBERT
```bash
python models/encoder_hubert.py
```
Kết quả lưu tại:
```
data/processed/en_features/
data/processed/vn_features/
```

### Bước 3: Huấn Luyện Pipeline End-to-End
```bash
python training/train_end2end.py --quantizer vqvae
```
- `--quantizer vqvae`: sử dụng VQ-VAE Quantizer (có thể thay bằng `gumbel`).  
- Checkpoints lưu tại `checkpoints/`  
- Metrics lưu tại `results/metrics.json`  

---

## 📊 Dashboard Phân Tích

Chạy lệnh sau để mở dashboard trực quan:
```bash
streamlit run dashboard/dashboard.py
```
Truy cập vào URL mà **Streamlit** cung cấp để xem chỉ số và nghe lại mẫu dịch.

---

## 📦 Đóng Gói Thành File Thực Thi (.exe)

Dùng **PyInstaller** để tạo file chạy độc lập:
```bash
pyinstaller --name S2ST_Project --onefile --add-data "webapp/frontend;webapp/frontend" run_webapp.py
```
- `--onefile`: tạo 1 file `.exe` duy nhất  
- `--add-data`: đóng gói toàn bộ thư mục giao diện web  

> Sau khi hoàn tất, file `S2ST_Project.exe` nằm trong thư mục `dist/`  
> Chạy file đó để khởi động ứng dụng.

---

## 📚 Thông Tin Bổ Sung

- **Tác giả:** Nhóm phát triển S2ST Việt–Anh  
- **Phiên bản:** 1.0  
- **Giấy phép:** MIT License
