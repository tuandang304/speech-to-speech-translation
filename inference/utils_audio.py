import torchaudio

def get_audio_duration(file_path):
    """
    Returns the duration of an audio file in seconds.
    """
    try:
        metadata = torchaudio.info(file_path)
        duration = metadata.num_frames / metadata.sample_rate
        return round(duration, 2)
    except Exception as e:
        print(f"Could not get duration for {file_path}: {e}")
        return 0.0