import pickle
import numpy as np
import tensorflow as tf
import librosa

# 1. 加载预处理参数
with open("../data/processed/features.pkl", "rb") as f:
    data = pickle.load(f)
scaler = data['scaler']
max_len = data['max_len']
label_map = data['label_map']
idx_to_label = {v: k for k, v in label_map.items()}
print("标签映射:", idx_to_label)

# 2. 加载模型
model = tf.keras.models.load_model("../models/best_model.h5")

# 3. 指定 CASIA 中一个 angry 音频文件的路径（请修改为实际路径）
audio_path = "../data/raw_audio/angry/204.wav"  # 替换为你的文件

# 4. 特征提取（完全复制 app.py 中的 preprocess_audio 逻辑）
y, sr = librosa.load(audio_path, sr=22050)
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)
feat = mel_db.T  # (T, 128)

# 填充/截断
if feat.shape[0] < max_len:
    pad = max_len - feat.shape[0]
    feat = np.pad(feat, ((0, pad), (0, 0)), mode='constant')
else:
    feat = feat[:max_len, :]

# 标准化
orig_shape = feat.shape
feat_2d = feat.reshape(-1, feat.shape[-1])
feat_scaled = scaler.transform(feat_2d)
feat = feat_scaled.reshape(orig_shape)

# 预测
feat_input = np.expand_dims(feat, axis=0)
pred_probs = model.predict(feat_input, verbose=0)
pred_idx = np.argmax(pred_probs[0])
pred_emotion = idx_to_label[pred_idx]
print(f"预测情感: {pred_emotion}")
print(f"概率分布: {pred_probs[0]}")