import re
import os
import json
import subprocess
import pandas as pd

audio_link = 'process_data/audio_link.csv'
dir_vie_m4a = 'data/vie_m4a/'
dir_en_m4a = 'data/en_m4a/'
num_videos = 4 # number of videos to download audio from

df = pd.read_csv(audio_link)
df['had_collected'] = df['had_collected'].fillna(0)
count = df['had_collected'].value_counts().get(1, 0)

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
    if count >= num_videos:
        print(f"[GET] Reached the limit of videos to process ({count}/{num_videos})")
        break

    youtube_url = row['link']
    print(f"[data] Downloading video: {youtube_url}")
    proc = subprocess.run(['yt-dlp', '-F', youtube_url], capture_output=True, text=True)

    vi_id = None
    en_id = None
    for line in proc.stdout.splitlines():
        if "[vi]" in line and "m4a" in line:
            if "medium" in line:
                vi_id = re.match(r'^(\S+)', line).group(1)
        elif "[en]" in line and "m4a" in line:
            if "medium" in line:
                en_id = re.match(r'^(\S+)', line).group(1)
        if (vi_id is not None and en_id is not None):
            break

    print(f"Vietnamese audio ID: {vi_id}, English audio ID: {en_id}")

    if vi_id is not None and en_id is not None:
        download_audio_from_youtube(vi_id, youtube_url, os.path.join(dir_vie_m4a, f'{idx}.m4a'))
        download_audio_from_youtube(en_id, youtube_url, os.path.join(dir_en_m4a, f'{idx}.m4a'))

    df.at[idx, 'had_collected'] = 1
    count += 1

df.to_csv(audio_link, index=False)
print("[data] Updated audio link statuses saved to", audio_link)