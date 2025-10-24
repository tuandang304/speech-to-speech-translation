# S2ST Demo Progress (v2)

## Tổng Quan
- Mục tiêu: kiểm tra pipeline huấn luyện với dữ liệu đã segment, xác nhận có thể tiếp tục train trên cặp EN–VI hiện có và ghi nhận kết quả trung gian để báo cáo tiến trình.
- Tình trạng: tiền xử lý đã chạy, Quantizer đã có checkpoint, Units đã sinh cho EN/VIE, Translator đã chỉnh để train đúng 1968 cặp.

## Chuẩn Bị Dữ Liệu
- Cắt câu từ YAML PhoST: `Datasets/PhoST/mini_data/split.py`
  - Đầu vào: `Datasets/PhoST/text_data/train/<id>/<id>.yaml` + audio ở `Datasets/PhoST/audio_data/train[/wav]/<id>.wav`.
  - Đầu ra EN: `Datasets/PhoST/mini_data/en/<id>/<id>_<####>.wav` và `.txt`.
  - Quy tắc đặt tên: `<id>_<4 chữ số>.wav` (ví dụ: `1_0001.wav` … `1_0144.wav`).
- Đã di chuyển EN mini data flatten về: `v2/data/en/*.wav`.

## Sửa Lỗi/Điều Chỉnh Kỹ Thuật
- Vấn đề TorchCodec (torchaudio backend) khi đọc WAV:
  - `v2/training/model/utils_audio.py`: thêm fallback `librosa` → tránh phụ thuộc TorchCodec/FFmpeg.
- Trích xuất HuBERT:
  - `v2/training/model/feature_extractor.py`: dùng AMP GPU, 1-pass (không chunk), thêm padding tối thiểu cho clip quá ngắn để tránh lỗi conv.
- Tiền xử lý linh hoạt ngôn ngữ:
  - `v2/training/preprocess_data.py`: chỉ duyệt các ngôn ngữ có thư mục tồn tại (EN/VIE).
- Ghép cặp Units EN–VI theo basename giao nhau:
  - `v2/training/train_translator.py`: dùng `common_ids` thay vì ghép theo thứ tự danh sách.

## Kết Quả Tiền Xử Lý (Preprocess)
- Lệnh chạy (từ `v2/`): `python training/preprocess_data.py`
- Kết quả đếm thực tế:
  - Features `.npy`: 2870 (v2/cache/features)
  - Units EN `.npy`: 2843 (v2/cache/units/en)
  - Units VI `.npy`: 1968 (v2/cache/units/vie)
  - Cặp EN–VI trùng tên: 1968
  - Quantizer checkpoint: có (`v2/checkpoints/quantizer.pt`)

## Train Quantizer (đã xong)
- Pipeline đã tự train khi chưa có checkpoint.
- Checkpoint: `v2/checkpoints/quantizer.pt`

## Train Translator (1968 cặp)
- Chỉnh sửa: `v2/training/train_translator.py` ghép cặp theo giao basename.
- Cách chạy: (từ `v2/`)
  - `python training/train_translator.py`
- Kết quả: đã train xong với 1968 cặp, checkpoint tồn tại tại `v2/checkpoints/translator.pt`.
- Log mẫu: `Số cặp EN–VI dùng để huấn luyện: 1968` và in loss theo epoch trong quá trình huấn luyện.

## Train Vocoder (tùy chọn)
- Yêu cầu: audio VIE khớp tên với units (ví dụ `v2/data/vie/1_0001.wav` ↔ `v2/cache/units/vie/1_0001.npy`).
- Lệnh: `python training/train_vocoder.py` (từ `v2/`).
- Checkpoint kỳ vọng: `v2/checkpoints/vocoder.pt`.
- Lưu ý đã sửa trainer: thêm `import torch.nn.functional as F` trong `train_vocoder.py` để tránh lỗi khi gọi `F.embedding`.

## Inference Test (1 file demo)
- Script: `v2/test_inference.py`
- Cách chạy (từ thư mục gốc dự án):
  - Mặc định lấy file đầu tiên trong `v2/data/en`:
    - `python v2/test_inference.py`
  - Chỉ định file đầu vào và đầu ra cụ thể:
    - `python v2/test_inference.py --input v2/data/en/1_0001.wav --output v2/results/1_0001_vi.wav --device cuda`
- Yêu cầu checkpoints: `checkpoints/feature_extractor` (thư mục), `checkpoints/quantizer.pt`, `checkpoints/translator.pt`, `checkpoints/vocoder.pt`.
- Lưu ý: lần đầu HiFi-GAN (torch.hub) có thể tải về model (cần mạng), các lần sau sẽ cache.

## Ghi Chú Vấn Đề & Cách Khắc Phục
- TorchCodec/FFmpeg mismatch: đã tránh bằng `librosa` fallback (không cần cài TorchCodec nặng).
- OOM khi HuBERT với clip dài: đã dùng AMP và (trước đó) chunking; hiện dùng 1-pass + padding tối thiểu vì dữ liệu đã segment.
- Thiếu `v2/data/vie`: script preprocess hiện tự bỏ qua ngôn ngữ không tồn tại và cảnh báo.

## Bước Tiếp Theo (Gợi Ý)
- Tăng số cặp: cắt VIE theo cùng YAML để sinh thêm units VIE, tăng cặp trùng tên > 1968.
- Huấn luyện Translator/Vocoder hoàn chỉnh; thêm eval BLEU/MOS nội bộ nếu cần báo cáo.
- (Tùy chọn) Scaffold pipeline streaming (mic → VAD → per‑segment inference) để demo độ trễ ~10–15s theo câu.

## Tái Hiện Nhanh (Commands)
```bash
cd v2
# Tiền xử lý (đã cấu hình tự động bỏ qua lang thiếu)
python training/preprocess_data.py

# Train translator trên 1968 cặp
python training/train_translator.py

# (Tùy chọn) Train vocoder nếu đã có v2/data/vie/*.wav khớp tên
python training/train_vocoder.py
```
