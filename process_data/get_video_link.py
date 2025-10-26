import yt_dlp
import pandas as pd

channel_url = "https://www.youtube.com/channel/UCX6OQ3DkcsbYNE6H8uQQuVA"
audio_link_path = "process_data/audio_link.csv"

ydl_opts = {
    'extract_flat': True,
    'quiet': True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(channel_url, download=False)
    if info.get('_type') == 'playlist' and len(info['entries']) > 0 and isinstance(info['entries'][0], dict) and 'entries' in info['entries'][0]:
        video_entries = info['entries'][0]['entries']
    else:
        video_entries = info.get('entries', [])
    videos = []
    for entry in video_entries:
        if 'url' in entry:
            videos.append(entry['url'])
        elif 'id' in entry:
            videos.append(f"https://www.youtube.com/watch?v={entry['id']}")

df = pd.read_csv(audio_link_path)
df_extended = pd.DataFrame({"link": videos, "had_collected": 0})
df = pd.concat([df, df_extended], ignore_index=True)
df = df.drop_duplicates(subset=['link'], keep='first').reset_index(drop=True)
df.to_csv(audio_link_path, index=False)
print("[data] Updated audio links saved to", audio_link_path)