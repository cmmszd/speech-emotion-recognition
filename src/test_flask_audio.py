import numpy as np
import pickle
import tensorflow as tf
import librosa

# 加载训练时的预处理参数
with open("../data/processed/features.pkl", "rb") as f:
    data = pickle.load(f)
    scaler = data['scaler']
    max_len = data['max_len']
    label_map = data['label_map']
    idx_to_label = {v: k for k, v in label_map.items()}

# 加载模型
model = tf.keras.models.load_model("../models/best_model.h5")

# 加载 Flask 保存的音频文件
audio_path = "debug_uploaded.wav"   # 这是 Flask 接口保存的副本
print(f"Testing file: {audio_path}")

y, sr = librosa.load(audio_path, sr=22050)

# 特征提取（与 app.py 中 preprocess_audio 完全一致）
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)
feat = mel_db.T

if feat.shape[0] < max_len:
    pad = max_len - feat.shape[0]
    feat = np.pad(feat, ((0, pad), (0, 0)), mode='constant')
else:
    feat = feat[:max_len, :]

orig_shape = feat.shape
feat_2d = feat.reshape(-1, feat.shape[-1])
feat_scaled = scaler.transform(feat_2d)
feat = feat_scaled.reshape(orig_shape)

feat_input = np.expand_dims(feat, axis=0)
pred_probs = model.predict(feat_input, verbose=0)
pred_idx = np.argmax(pred_probs[0])
pred_emotion = idx_to_label[pred_idx]
print(f"独立脚本预测结果: {pred_emotion}")
print(f"概率分布: {pred_probs[0]}")