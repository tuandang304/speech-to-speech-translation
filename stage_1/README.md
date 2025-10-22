# S2ST Pipeline cho Tiếng Việt (Giai đoạn 1 - Huấn luyện Module hóa)

Dự án này triển khai một pipeline Speech-to-Speech Translation (S2ST) dựa trên các unit rời rạc, với kiến trúc được module hóa để có thể huấn luyện riêng biệt từng thành phần.

## Cấu trúc thư mục

- `configs/`: Chứa các file cấu hình YAML cho từng bước huấn luyện.
- `scripts/`: Chứa các script để chạy từng bước của quy trình (huấn luyện quantizer, trích xuất unit, huấn luyện translator).
- `src/`: Chứa mã nguồn định nghĩa các model và dataloader.
- `inference.py`: Script để chạy pipeline S2ST hoàn chỉnh với các model đã huấn luyện.
- `requirements.txt`: Các thư viện Python cần thiết.

## Thiết lập môi trường

1.  Clone repository này:
    ```bash
    git clone <your-repo-url>
    cd s2st-vietnamese
    ```

2.  Tạo và kích hoạt môi trường ảo (khuyến khích):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows: venv\Scripts\activate
    ```

3.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Chuẩn bị dữ liệu**: Bạn cần chuẩn bị các file manifest (ví dụ: `.tsv`) trỏ đến vị trí các file audio.
    -   Đối với `train_quantizer`, chỉ cần danh sách các file audio đơn ngữ.
    -   Đối với `extract_units` và `train_translator`, cần danh sách các cặp file audio song ngữ (nguồn và đích).

## Quy trình Huấn luyện

Quy trình được chia thành các bước độc lập. Bạn phải chạy theo đúng thứ tự.

### Bước 1: Huấn luyện Quantizer

Quantizer (ví dụ VQ-VAE) có nhiệm vụ chuyển đổi các đặc trưng âm thanh liên tục từ HuBERT thành một chuỗi các unit rời rạc (indices từ một codebook).

-   **Tùy chọn A: Huấn luyện VQ-VAE (Learnable Quantization)**
    ```bash
    python scripts/01_train_quantizer.py --config configs/quantizer_config.yaml
    ```
    Mô hình quantizer sẽ được lưu vào thư mục `checkpoints/quantizer/`.

-   **Tùy chọn B: Tạo K-means Model (Baseline)**
    *(Script này cần được bạn hiện thực)*
    ```bash
    python scripts/02_generate_kmeans.py --config configs/data_config.yaml --output_path checkpoints/kmeans/kmeans_model.pt
    ```
    Script này sẽ trích xuất đặc trưng HuBERT từ một tập dữ liệu lớn và thực hiện phân cụm K-means để tạo codebook.

### Bước 2: Trích xuất Units cho dữ liệu song ngữ

Sau khi có quantizer, chúng ta dùng nó để "tokenize" toàn bộ tập dữ liệu audio song ngữ thành các chuỗi unit. Việc này giúp tăng tốc độ huấn luyện translator một cách đáng kể vì không cần chạy HuBERT nữa.

```bash
python scripts/03_extract_units.py \
    --parallel-data-manifest /path/to/parallel_audio.tsv \
    --quantizer-path checkpoints/quantizer/vqvae_quantizer.pt \
    --output-manifest data/train_units.txt