import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from api.auto_translate import run_translation_pipeline
from api.utils import save_log

# -- Flask App Setup --
# We set static_folder to point to the 'frontend' directory
app = Flask(__name__, static_folder='../frontend', static_url_path='')

# -- Configuration --
# Define paths relative to this file's location
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'temp_uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, '..', '..', 'results') # ../../results
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = os.path.abspath(RESULT_FOLDER)

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# -- Routes --

@app.route('/')
def index():
    """Serve the main HTML file."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads, trigger translation, and return the result."""
    if 'audio_file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Run the S2ST pipeline
        try:
            output_filename, duration = run_translation_pipeline(input_path, app.config['RESULT_FOLDER'])
            output_url = f"/results/{output_filename}"
            
            # Log the translation event
            save_log({
                "input_filename": filename,
                "output_filename": output_filename,
                "duration": duration,
                "timestamp": __import__('datetime').datetime.now().isoformat()
            })
            
            return jsonify({"success": True, "output_audio_url": output_url})
        except Exception as e:
            # Log the error for debugging
            print(f"Error during translation pipeline: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": "An internal error occurred during translation."}), 500

@app.route('/results/<path:filename>')
def get_result_audio(filename):
    """Serve the translated audio files from the results directory."""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    print("Flask server starting...")
    print(f"Serving frontend from: {os.path.abspath(app.static_folder)}")
    print(f"Saving results to: {app.config['RESULT_FOLDER']}")
    app.run(debug=True, port=5001)