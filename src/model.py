import tensorflow as tf
from tensorflow.keras import layers, Model

def create_cnn_bilstm_attn(input_shape, num_classes):
    """
    创建 CNN + BiLSTM + Attention 模型

    Parameters:
        input_shape (tuple): 输入形状，例如 (timesteps, n_features, 1)
        num_classes (int): 类别数

    Returns:
        model: Keras 模型
    """
    inputs = tf.keras.Input(shape=input_shape)

    # 1. 卷积层提取局部特征
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # 2. BiLSTM 捕捉时序依赖
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # 3. 注意力机制
    # 使用自定义 Attention 层（也可用其他实现）
    attention = layers.Attention()([x, x])   # 自注意力
    # 或者用以下简单加权：
    # attention_weights = layers.Dense(1, activation='tanh')(x)
    # attention_weights = layers.Flatten()(attention_weights)
    # attention_weights = layers.Activation('softmax')(attention_weights)
    # attended = layers.Dot(axes=1)([attention_weights, x])

    # 展平后接全连接
    flat = layers.Flatten()(attention)
    flat = layers.Dropout(0.5)(flat)
    flat = layers.Dense(256, activation='relu')(flat)
    outputs = layers.Dense(num_classes, activation='softmax')(flat)

    model = Model(inputs=inputs, outputs=outputs)
    return model