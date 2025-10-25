import re
import os
import json
import subprocess
import pandas as pd

audio_link = 'process_data/audio_link.csv'
df = pd.read_csv(audio_link)
df['had_collected'].fillna(0, inplace=True)

dir_vie_m4a = 'data/vie_m4a/'
dir_en_m4a = 'data/en_m4a/'

dir_vie_wav = 'data/vie_wav/'
dir_en_wav = 'data/en_wav/'

def download_audio_from_youtube(id, youtube_url, output_path):
    comnand = [
        'yt-dlp',
        '-f', id,
        '-o', output_path,
        youtube_url
    ]
    subprocess.run(comnand, check=True)

df_links = df[df['had_collected'] == 0]
for idx, row in df_links.iterrows():
    youtube_url = row['link']
    
    proc = subprocess.run(['yt-dlp', '-F', youtube_url], capture_output=True, text=True)
    # matches = re.findall(r'^(\S+)\s+\S+\s+audio only.*mp4a.*\[vi\].*$', proc.stdout, re.MULTILINE)

    for line in proc.stdout.splitlines():
        vi_id = None
        en_id = None
        if "[vi]" in line and "audio only" in line:
            if "medium" in line:
                vi_id = re.match(r'^(\S+)', line).group(1)
        elif "[en]" in line and "audio only" in line:
            if "medium" in line:
                en_id = re.match(r'^(\S+)', line).group(1)
        if vi_id is not None and en_id is not None:
            break

    # if not best and matches:
    #     best = matches[0]
    if vi_id is not None and en_id is not None:
        download_audio_from_youtube(vi_id, youtube_url, os.path.join(dir_vie_m4a, f'{idx}.m4a'))
        download_audio_from_youtube(en_id, youtube_url, os.path.join(dir_en_m4a, f'{idx}.m4a'))

    df.at[idx, 'had_collected'] = 1

df.to_csv(audio_link, index=False)