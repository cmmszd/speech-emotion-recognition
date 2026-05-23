import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 加载特征和标签
with open("../data/processed/features.pkl", "rb") as f:
    data = pickle.load(f)
X = data['features']
y = data['labels']
label_map = data['label_map']

# 2. 划分数据集（必须与训练时完全一致）
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 3. 加载训练好的模型
model = tf.keras.models.load_model("../models/best_model.h5")

# 4. 预测
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# 5. 获取类别名称（按整数编码排序）
class_names = [name for name, idx in sorted(label_map.items(), key=lambda x: x[1])]

# 6. 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix for Speech Emotion Recognition', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()