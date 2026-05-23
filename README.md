# 语音情感识别系统 (Speech Emotion Recognition)

基于深度学习的语音情感识别系统，使用 CASIA 等数据集训练，支持 6 种基本情感（生气、恐惧、开心、中性、悲伤、惊讶）。提供 Flask Web 服务接口和完整的训练、评估流程。

## 功能特点
- **情感识别**：支持实时上传音频文件（.wav），预测情感类别
- **离线训练**：完整的数据预处理、特征提取、模型训练脚本
- **可视化**：混淆矩阵、Mel 频谱图、效价-唤醒度分析
- **部署简单**：提供 Flask API 和简洁的前端页面

## 技术栈
- Python 3.10
- TensorFlow / Keras
- Librosa（音频特征提取）
- Flask（Web 服务）
- Scikit-learn（评估指标）

## 项目结构

```
SpeechEmotionRecognition/
├── data/                   # 数据集（按情感分类的音频文件）
├── models/                 # 训练好的模型（best_model.h5, final_model.h5）
├── logs/                   # TensorBoard 训练日志
├── src/
│   ├── model.py            # 模型定义
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本
│   ├── extract_features.py # 特征提取
│   ├── app.py              # Flask Web 应用
│   └── ...                 # 其他辅助脚本
├── templates/
│   └── index.html          # Web 前端界面
├── requirements.txt        # Python 依赖
└── README.md
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/cmmszd/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- tensorflow
- flask
- librosa
- numpy, scikit-learn, matplotlib

### 3. 运行 Web 服务
```bash
python src/app.py
```
然后打开浏览器访问 `http://127.0.0.1:5000`，上传 `.wav` 文件进行测试。

### 4. 训练新模型
```bash
python src/train.py
```
训练日志会自动保存到 `logs/`，可使用 TensorBoard 查看。

### 5. 评估模型
```bash
python src/evaluate.py
```
会输出准确率、混淆矩阵等指标，并生成 `confusion_matrix.png`。

## 数据集说明
- 使用 CASIA 数据集（6 种情感 × 1000+ 样本）
- 音频文件统一转换为 16kHz 单声道
- 提取 MFCC、Mel 频谱等特征作为输入

## 模型架构
- 输入：固定长度的特征向量（如 40 维 MFCC × 帧数）
- 网络：LSTM + Dense 层，或 CNN + LSTM
- 输出：6 类情感的概率分布

## 可视化结果
运行以下脚本生成可视化图表：
- `python src/plot_mel_spectrograms.py` → Mel 频谱图
- `python src/plot_confusion_matrix.py` → 混淆矩阵
- `python src/plot_valence_arousal.py` → 效价-唤醒度分布

## 常见问题

**Q: 运行时提示缺少某些 Python 模块？**  
A: 检查是否安装了所有依赖：`pip install -r requirements.txt`。如果仍有缺失，手动安装对应模块。

**Q: 上传音频后返回错误？**  
A: 确保音频格式为 `.wav`，时长 1-5 秒，采样率 16kHz。过短或过长的音频可能提取特征失败。

**Q: 训练时显存不足？**  
A: 减小 batch_size 或输入特征维度。

## 后续改进方向
- 添加实时麦克风录音识别
- 支持更多情感类别
- 优化模型大小，部署到移动端

## 作者

丁梦琳

## 许可证

MIT
```
## 如何添加到项目
1. 在项目根目录（`D:\SpeechEmotionRecognition\`）新建一个文本文件，命名为 `README.md`
2. 将上面的内容复制进去并保存
3. 提交到 GitHub：

```bash
git add README.md
git commit -m "添加项目 README"
git push
```

