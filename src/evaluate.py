import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. 加载预处理好的特征和标签
with open("../data/processed/features.pkl", "rb") as f:
    data = pickle.load(f)

X = data['features']
y = data['labels']
label_map = data['label_map']

# 2. 划分数据集（与训练脚本完全一致）
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 3. 加载训练好的最佳模型
model = tf.keras.models.load_model("../models/best_model.h5")

# 4. 在测试集上预测
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# 5. 生成分类报告
class_names = [name for name, idx in sorted(label_map.items(), key=lambda x: x[1])]
print("\n=== 分类报告 ===")
print(classification_report(y_test, y_pred, target_names=class_names))