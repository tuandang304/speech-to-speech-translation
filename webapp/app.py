from flask import Flask, render_template, request, send_file
import os
import torch
import time
from werkzeug.utils import secure_filename

# Thêm thư mục gốc vào path để import model
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.pipeline_inference import S2ST_Pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'webapp/uploads'
app.config['OUTPUT_FOLDER'] = 'webapp/outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Tải pipeline một lần duy nhất khi server khởi động
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    pipeline = S2ST_Pipeline(device=device)
except FileNotFoundError as e:
    print("="*50)
    print(f"LỖI: {e}")
    print("Vui lòng chạy các script training trong thư mục 'training/' để tạo checkpoints.")
    print("="*50)
    pipeline = None # Đánh dấu pipeline chưa sẵn sàng

@app.route('/')
def index():
    if pipeline is None:
        return "Lỗi: Pipeline chưa được khởi tạo do thiếu checkpoints. Vui lòng kiểm tra console và chạy các script huấn luyện.", 500
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    if pipeline is None:
        return "Pipeline chưa sẵn sàng.", 503

    if 'audio_file' not in request.files:
        return "Không có file nào được tải lên.", 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return "Chưa chọn file.", 400

    if file and file.filename.endswith('.wav'):
        # Tạo tên file an toàn và độc nhất
        timestamp = int(time.time())
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Tạo đường dẫn output
        output_filename = f"translated_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Dịch
        try:
            pipeline.translate(input_path, output_path)
        except Exception as e:
            print(f"Lỗi trong quá trình dịch: {e}")
            return "Đã có lỗi xảy ra trong quá trình dịch.", 500
        
        return render_template('result.html', original_file=filename, translated_file=output_filename)

    return "File không hợp lệ, vui lòng tải lên file .wav", 400

@app.route('/play/<folder>/<filename>')
def play_audio(folder, filename):
    if folder not in ['uploads', 'outputs']:
        return "Thư mục không hợp lệ", 404
    path = os.path.join(app.config[folder.upper() + '_FOLDER'], filename)
    return send_file(path)

if __name__ == '__main__':
    app.run(debug=True)