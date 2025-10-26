import os
from pydub import AudioSegment

input_folder_vie = "data/vie_m4a"
input_folder_en = "data/en_m4a"
output_folder_vie = "data/vie_wav"
output_folder_en = "data/en_wav"

os.makedirs(output_folder_vie, exist_ok=True)
os.makedirs(output_folder_en, exist_ok=True)

def convert_m4a_to_wav(input_path, output_path):
    for file_name in os.listdir(input_path):
        if file_name.endswith(".m4a"):
            m4a_path = os.path.join(input_path, file_name)
            wav_path = os.path.join(output_path, file_name.replace(".m4a", ".wav"))
            
            audio = AudioSegment.from_file(m4a_path, format="m4a")
            audio.export(wav_path, format="wav")
            print(f"[data] Converted {m4a_path} to {wav_path}")

convert_m4a_to_wav(input_folder_vie, output_folder_vie)
convert_m4a_to_wav(input_folder_en, output_folder_en)


