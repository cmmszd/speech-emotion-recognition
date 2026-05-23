"""
train.py
加载预处理好的特征和标签，划分数据集，训练模型，并保存最佳模型。
"""

import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from model import create_cnn_bilstm_attn

# ========== 1. 加载预处理好的特征和标签 ==========
data_path = "../data/processed/features.pkl"
with open(data_path, 'rb') as f:
    data = pickle.load(f)

X = data['features']          # shape: (n_samples, timesteps, n_features)
y = data['labels']            # shape: (n_samples,)

print(f"特征形状: {X.shape}")
print(f"标签形状: {y.shape}")
print(f"类别数: {len(np.unique(y))}")

# ========== 2. 划分训练集、验证集和测试集 (8:1:1) ==========
# 首先分出 20% 作为临时集（包含验证和测试）
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 再将临时集等分为验证集和测试集（各占原始数据的 10%）
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# ========== 3. 将标签转换为 one-hot 编码 ==========
num_classes = len(np.unique(y))
y_train_categorical = to_categorical(y_train, num_classes=num_classes)
y_val_categorical = to_categorical(y_val, num_classes=num_classes)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")
print(f"测试集大小: {X_test.shape}")

# ========== 4. 准备模型输入形状 ==========
# 原始特征形状已经是 (timesteps, n_features)，无需增加通道维度
input_shape = (X_train.shape[1], X_train.shape[2])   # (timesteps, n_features)
print(f"输入形状: {input_shape}")

# ========== 5. 创建模型 ==========
model = create_cnn_bilstm_attn(input_shape, num_classes)

# ========== 6. 编译模型 ==========
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ========== 7. 定义回调函数 ==========
os.makedirs('../models', exist_ok=True)

# 模型检查点：保存验证准确率最高的模型
checkpoint = ModelCheckpoint(
    filepath='../models/best_model.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

# 早停：若验证损失连续 20 个 epoch 不下降则停止训练
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# TensorBoard：记录训练曲线和直方图
tensorboard = TensorBoard(
    log_dir='../logs',
    histogram_freq=1
)

# ========== 8. 开始训练 ==========
history = model.fit(
    x=X_train,
    y=y_train_categorical,
    batch_size=32,
    epochs=100,
    validation_data=(X_val, y_val_categorical),
    callbacks=[checkpoint, early_stopping, tensorboard],
    verbose=1
)

# ========== 9. 在测试集上评估最终模型 ==========
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
print(f"\n=== 最终测试集准确率: {test_accuracy:.4f} ===")

# 保存最终模型（可能不是最佳，但也可用）
model.save('../models/final_model.h5')
print("模型已保存至 models/ 目录")