import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. 加载特征
with open("../data/processed/features.pkl", "rb") as f:
    data = pickle.load(f)
X = data['features']
y = data['labels']
label_map = data['label_map']
label_names = [name for name, idx in sorted(label_map.items(), key=lambda x: x[1])]

# 2. 划分数据集（与 train.py 一致）
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 3. 加载模型
model = tf.keras.models.load_model("../models/best_model.h5")

# 4. 预测测试集
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# 5. 打印分类报告
print("\n=== 分类报告（测试集）===")
print(classification_report(y_test, y_pred, target_names=label_names))

# 6. 打印每个类别的召回率
print("\n=== 各类别召回率 ===")
for i, name in enumerate(label_names):
    true_i = (y_test == i)
    pred_i = (y_pred == i)
    correct = np.sum(true_i & pred_i)
    total = np.sum(true_i)
    recall = correct / total if total > 0 else 0
    print(f"{name}: {recall:.3f} ({correct}/{total})")