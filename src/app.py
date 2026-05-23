import os
import numpy as np
import pickle
import tensorflow as tf
import librosa
from flask import Flask, request, jsonify, render_template, send_file
import pyttsx3
import random
import tempfile
import uuid
import traceback
import subprocess
import shutil

app = Flask(__name__)
app.template_folder = '../templates'

# ========== 加载预处理参数和模型 ==========

print("正在加载模型和预处理参数...")
try:
    with open("../data/processed/features.pkl", "rb") as f:
        data = pickle.load(f)
        scaler = data['scaler']
        max_len = data['max_len']
        label_map = data['label_map']
        idx_to_label = {v: k for k, v in label_map.items()}
    model = tf.keras.models.load_model("../models/best_model.h5")
    print("模型加载成功，情感类别：", list(label_map.keys()))
    print(f"max_len = {max_len}")
except Exception as e:
    print("模型加载失败:", e)
    traceback.print_exc()
    exit(1)

# ========== 回复语句库 ==========

reply_library = {
    "angry": ["别生气啦，放松一下。", "我知道你现在很恼火，深呼吸～", "冷静一点，我们一起想办法。", "生气解决不了问题，先喝口水吧。"],
    "fear": ["别怕，有我在呢。", "放轻松，一切都会好的。", "别担心，我会一直陪着你。", "害怕是正常的，我们慢慢来。"],
    "happy": ["真为你开心！我也被感染了。", "快乐是会传染的，继续分享吧。", "看到你这么高兴，我也觉得幸福。", "哈哈，太好了！还有什么开心的事吗？"],
    "neutral": ["好的，我明白了。", "嗯，继续说吧。", "原来如此。", "我听着呢。"],
    "sad": ["别难过，我会一直陪着你。", "有什么想说的，我都在听。", "难过的话就哭出来吧，我在这里。", "时间会治愈一切的，要加油哦。"],
    "surprise": ["哇，真没想到！太有意思了。", "这真令人惊讶，然后呢？", "真的吗？太神奇了！", "出乎意料！快多讲一些。"]
}

# ========== TTS ==========

def text_to_speech(text):
    temp_filename = tempfile.gettempdir() + f"/tts_{uuid.uuid4().hex}.wav"
    subprocess.run(["python", "tts_worker.py", text, temp_filename], check=True)
    return temp_filename

def preprocess_audio(file_path):
    """
    预处理音频文件，返回模型输入特征 (1, max_len, 128)
    假设 file_path 指向一个 WAV 文件（22050Hz，单声道）
    """
    y, sr = librosa.load(file_path, sr=22050)

    # 提取梅尔频谱图
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feat = mel_db.T  # (T, 128)
    
    if feat.shape[0] < max_len:
        pad = max_len - feat.shape[0]
        feat = np.pad(feat, ((0, pad), (0, 0)), mode='constant')
    else:
        feat = feat[:max_len, :]
    
    orig_shape = feat.shape
    feat_2d = feat.reshape(-1, feat.shape[-1])
    feat_scaled = scaler.transform(feat_2d)
    feat = feat_scaled.reshape(orig_shape)
    
    return np.expand_dims(feat, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emotion', methods=['POST'])
def emotion_api():
    temp_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file uploaded'}), 400
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # 保存上传的音频到临时文件
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        file.save(temp_path)
        print(f"Saved temp file: {temp_path}")
        
        # 【调试】保留一份永久副本用于独立测试
        debug_copy = "debug_uploaded.wav"
        shutil.copy(temp_path, debug_copy)
        print(f"Debug copy saved to {debug_copy}")
        
        # 预处理和预测
        features = preprocess_audio(temp_path)
        pred_probs = model.predict(features, verbose=0)
        pred_idx = np.argmax(pred_probs[0])
        emotion = idx_to_label[pred_idx]
        print(f"Predicted emotion: {emotion}, probs: {pred_probs[0]}")
        
        return jsonify({'emotion': emotion})
    except Exception as e:
        print("=== Emotion API Error ===")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Removed temp file: {temp_path}")

@app.route('/reply', methods=['POST'])
def reply_api():
    try:
        data = request.get_json()
        if not data or 'emotion' not in data:
            return jsonify({'error': 'No emotion provided'}), 400
        emotion = data['emotion']
        replies = reply_library.get(emotion, ["我不太确定该说什么。"])
        text = random.choice(replies)
        audio_path = text_to_speech(text)
        return send_file(audio_path, mimetype='audio/wav', as_attachment=False)
    except Exception as e:
        print("=== Reply API Error ===")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('static/temp_audio', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)