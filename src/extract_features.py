import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def extract_features(data_path, save_path, feature_type='melspectrogram'):
    """
    提取音频特征并保存（包含标准化和标签编码）

    Parameters:
        data_path (str): 音频数据目录路径，内部子文件夹名为情绪标签
        save_path (str): 特征保存文件路径（.pkl）
        feature_type (str): 特征类型，可选 'melspectrogram', 'mfcc', 'spectrogram'
    """
    features = []
    labels = []

    # 遍历每个情绪类别文件夹
    for emotion_label in os.listdir(data_path):
        emotion_path = os.path.join(data_path, emotion_label)

        if not os.path.isdir(emotion_path):
            continue

        print(f"Processing emotion: {emotion_label}")

        # 遍历每个音频文件
        for audio_file in os.listdir(emotion_path):
            if not audio_file.endswith('.wav'):
                continue

            audio_path = os.path.join(emotion_path, audio_file)

            try:
                # 加载音频文件
                y, sr = librosa.load(audio_path, sr=22050)

                # 提取特征
                if feature_type == 'melspectrogram':
                    feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    feat = librosa.power_to_db(feat, ref=np.max)  # 局部最大值参考
                elif feature_type == 'mfcc':
                    feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                elif feature_type == 'spectrogram':
                    feat = librosa.stft(y)
                    feat = np.abs(feat)
                else:
                    raise ValueError(f"Unsupported feature type: {feature_type}")

                # 转置，使得时间轴在第一维
                feat = feat.T

                features.append(feat)
                labels.append(emotion_label)

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue

    if len(features) == 0:
        raise ValueError("No audio files found or all failed to load.")

    # 动态计算最大帧数
    max_len = max(f.shape[0] for f in features)

    # 对特征进行填充/截断，使其长度一致
    padded_features = []
    for feat in features:
        n_frames = feat.shape[0]
        if n_frames < max_len:
            pad_width = max_len - n_frames
            padded = np.pad(feat, ((0, pad_width), (0, 0)), mode='constant')
        else:
            padded = feat[:max_len, :]
        padded_features.append(padded)

    # 转换为numpy数组
    features_array = np.array(padded_features)  # shape: (n_samples, time, n_features)
    labels_array = np.array(labels)

    # 特征标准化：将时间轴和样本轴合并，计算均值和方差，然后还原形状
    scaler = StandardScaler()
    n_samples, n_time, n_feat = features_array.shape
    features_2d = features_array.reshape(-1, n_feat)
    features_2d_scaled = scaler.fit_transform(features_2d)
    features_scaled = features_2d_scaled.reshape(n_samples, n_time, n_feat)

    # 标签编码
    unique_labels = sorted(set(labels_array))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    labels_encoded = np.array([label_to_int[l] for l in labels_array])

    # 保存特征、标签、映射关系以及标准化器
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump({
            'features': features_scaled,
            'labels': labels_encoded,
            'label_map': label_to_int,
            'scaler': scaler,
            'max_len': max_len,
            'feature_type': feature_type
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Features extracted successfully! Shape: {features_scaled.shape}")
    print(f"Labels shape: {labels_encoded.shape}")
    print(f"Label mapping: {label_to_int}")
    return features_scaled, labels_encoded


if __name__ == "__main__":
    # 路径设置（请根据实际位置调整）
    data_path = "../data/raw_audio"           # 原始音频目录
    save_path = "../data/processed/features.pkl"  # 特征保存文件

    # 提取 mel-spectrogram 特征（可根据需要修改 feature_type）
    extract_features(data_path, save_path, feature_type='melspectrogram')