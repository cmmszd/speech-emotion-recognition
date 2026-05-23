import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ========== 配置参数 ==========
DATA_ROOT = "../data/raw_audio"   # 数据集根目录
TARGET_SR = 22050                  # 采样率
N_MELS = 128                       # 梅尔滤波器个数
FMAX = 8000                        # 最高频率
OUTPUT_IMG = "mel_spectrogram_6emotions.png"

# 情感顺序（与文件夹名一致）
emotions = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
display_names = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# 自动选取每个情感文件夹的第一个 wav 文件
audio_files = []
for emo in emotions:
    folder = os.path.join(DATA_ROOT, emo)
    wavs = [f for f in os.listdir(folder) if f.endswith('.wav')]
    audio_files.append(os.path.join(folder, wavs[0]))
    print(f"{display_names[emotions.index(emo)]} -> {audio_files[-1]}")

# ========== 绘制 2行3列子图 ==========
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
fig.suptitle('Mel-spectrograms of Six Emotions (CASIA Dataset)', fontsize=14)
axes = axes.flatten()

for idx, (file_path, title) in enumerate(zip(audio_files, display_names)):
    y, sr = librosa.load(file_path, sr=TARGET_SR)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
                             ax=axes[idx], cmap='viridis')
    axes[idx].set_title(title, fontsize=10)
    axes[idx].set_xlabel('Time (s)')
    axes[idx].set_ylabel('Mel Frequency')

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
plt.show()