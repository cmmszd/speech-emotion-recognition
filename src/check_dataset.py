import os

data_path = "../data/raw_audio"
if not os.path.exists(data_path):
    print(f"路径不存在: {data_path}")
else:
    for emotion in os.listdir(data_path):
        emotion_path = os.path.join(data_path, emotion)
        if os.path.isdir(emotion_path):
            wavs = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
            print(f"{emotion}: {len(wavs)} 个 wav 文件")