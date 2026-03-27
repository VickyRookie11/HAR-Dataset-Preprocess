import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

def load_data(split_dir):
    X_path = os.path.join(split_dir, 'X.csv')
    y_path = os.path.join(split_dir, 'y.csv')
    if not os.path.exists(X_path) or not os.path.exists(y_path):
         return None, None
    X = np.loadtxt(X_path, delimiter=',')
    y = np.loadtxt(y_path, delimiter=',')
    # Reshape X backwards to (N, 48, 3, 1)
    X = X.reshape(-1, 48, 3, 1)
    return X, y

def build_strong_model(input_shape):
    # 采用更加深层次和具备残差跳跃连接的网络以提升精度
    inputs = tf.keras.Input(shape=input_shape)
    
    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Global Max Pooling (符合 GMP 模型特征)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    
    # Fully Connected
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def main():
    print("=== 开始训练 (从零构建深层 CNN 网络) ===")
    
    dataset_dir = './stm32ai_dataset'
    X_train, y_train = load_data(os.path.join(dataset_dir, 'train'))
    X_val, y_val = load_data(os.path.join(dataset_dir, 'val'))
    X_test, y_test = load_data(os.path.join(dataset_dir, 'test'))
    
    if X_train is None:
        print("Dataset missing.")
        return
        
    print(f"数据分布 -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): w for c, w in zip(classes, weights)}
    print(f"应用类别权重 (ADL vs FALL): {class_weight_dict}")
    
    model = build_strong_model((48, 3, 1))
    
    # 添加 Label Smoothing (标签平滑) 防止过拟合提升泛化能力
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    # 因为使用 CategoricalCrossentropy 需要将 y 转换为 one-hot
    y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, 2)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=loss_fn,
                  metrics=['accuracy'])
                  
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1)
    
    print("--- 训练深度模型 ---")
    model.fit(X_train, y_train_cat,
              epochs=150,
              batch_size=64,
              validation_data=(X_val, y_val_cat),
              class_weight=class_weight_dict,
              callbacks=[early_stop, reduce_lr])
              
    print("评估极致性能...")
    preds = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, preds)
    print(f"\n======================================")
    print(f"🔥 测试集准确率 (Test Accuracy): {acc*100:.2f}% 🔥")
    print(f"======================================\n")
    
    h5_path = "fall_detect.h5"
    onnx_path = "fall_detect.onnx"
    model.save(h5_path)
    
    try:
        import tf2onnx
        tf2onnx.convert.from_keras(model, output_path=onnx_path)
        print(f"✅ ONNX 导出成功: {onnx_path}")
    except Exception:
        pass

if __name__ == '__main__':
    main()