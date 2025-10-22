import os
import sys
import time

# Add project root to path to import inference module.
# This makes the webapp runnable from anywhere.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from inference.s2st_pipeline import S2STPipeline
from inference.utils_audio import get_audio_duration

# Initialize the pipeline once when the module is loaded.
# This is efficient as it avoids reloading the models on every request.
print("Loading S2ST pipeline for the webapp...")
pipeline = S2STPipeline(checkpoints_dir='checkpoints')
print("S2ST pipeline is ready for web requests.")

def run_translation_pipeline(input_path, result_dir):
    """
    Takes an input audio path, runs the S2ST pipeline, and returns details of the output.
    
    Args:
        input_path (str): Path to the uploaded English audio file.
        result_dir (str): Directory where the translated audio should be saved.

    Returns:
        tuple: (output_filename, duration)
    """
    timestamp = int(time.time())
    original_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_filename = f"{original_filename}_vn_{timestamp}.wav"
    output_path = os.path.join(result_dir, output_filename)
    
    # Run the core translation function
    pipeline.translate_audio(input_path, output_path)
    
    # Get the duration of the generated audio
    duration = get_audio_duration(output_path)
    
    return output_filename, duration