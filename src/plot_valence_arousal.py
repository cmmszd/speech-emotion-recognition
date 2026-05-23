import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

fig, ax = plt.subplots(figsize=(6, 6))

# 坐标轴
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel('Valence (Negative ← → Positive)', fontsize=12)
ax.set_ylabel('Arousal (Low ← → High)', fontsize=12)
ax.set_title('Valence-Arousal Emotion Model', fontsize=14)

# 添加情感区域（椭圆）
regions = [
    (0.7, 0.7, 0.5, 0.4, 'yellow', 'Happy'),
    (-0.6, 0.7, 0.5, 0.4, 'red', 'Angry'),
    (-0.6, -0.7, 0.5, 0.4, 'blue', 'Sad'),
    (0.0, 0.0, 0.4, 0.3, 'lightgreen', 'Calm')
]
for x, y, w, h, color, label in regions:
    ellipse = Ellipse((x, y), w, h, color=color, alpha=0.4)
    ax.add_patch(ellipse)
    ax.text(x, y, label, ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.savefig('valence_arousal_model.png', dpi=300)
plt.show()