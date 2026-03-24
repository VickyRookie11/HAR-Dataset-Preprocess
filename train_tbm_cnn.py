import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from scipy.signal import resample
import math
import random

# TBM 模拟阈值
TH1 = 0.72  # SMV_min 阈值 (g)
TH2 = 1.71  # SMV_max 阈值 (g)

# 参数
WINDOW_LEN = 150      # 3秒 (50Hz采样率)
SMV_SEG_LEN = 48      # SMV 编码长度
SAMPLE_RATE = 50      # 目标采样率
ORIGINAL_RATE = 200   # 原始采样率

def read_sisfall_file(file_path):
    # SisFall 数据格式是 "ADXL345(x,y,z) ITG3200(x,y,z) MMA8451Q(x,y,z)"
    # 我们使用 MMA8451Q，最后一列，1LSB = 1/4096g，或者 ADXL345 (前三列，16g范围，1/256g)
    # 取决于数据格式，尝试解析 ADXL345 并换算成 g。根据提供的 head，为逗号和分号分隔
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                     # 使用前三列，1LSB = 1/256g (可以根据实际数据集调整)
                     ax = float(parts[0]) / 256.0
                     ay = float(parts[1]) / 256.0
                     az = float(parts[2].replace(';', '')) / 256.0
                     data.append([ax, ay, az])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return np.array(data)

def preprocess_and_detect_tbm(data_dir, label):
    X_samples = []
    y_samples = []
    
    files = glob.glob(os.path.join(data_dir, '*.txt'))
    print(f"Found {len(files)} files in {data_dir}")
    
    for file in files:
        data = read_sisfall_file(file)
        if len(data) == 0:
            continue
            
        # 降采样：200Hz -> 50Hz
        num_samples = int(len(data) * (SAMPLE_RATE / ORIGINAL_RATE))
        if num_samples == 0:
            continue
        data_50hz = resample(data, num_samples, axis=0)
        
        # 计算 SMV序列 (合加速度)
        smv = np.sqrt(np.sum(data_50hz**2, axis=1))
        
        # 滑动窗口模拟 TBM
        step = SAMPLE_RATE # 1秒步长
        for i in range(0, len(smv) - WINDOW_LEN + 1, step):
            window = smv[i:i + WINDOW_LEN]
            
            # TBM 判断条件
            if np.min(window) < TH1 and np.max(window) > TH2:
                # 寻找窗口内的 SMV 最大值索引
                max_idx_in_window = np.argmax(window)
                max_idx_global = i + max_idx_in_window
                
                # 截取前后各 24点 (总长48点)
                start_idx = max_idx_global - (SMV_SEG_LEN // 2)
                end_idx = start_idx + SMV_SEG_LEN
                
                # 边界处理 (补0或边缘填充)
                seg = np.zeros(SMV_SEG_LEN)
                if start_idx < 0:
                    valid_data = smv[0:end_idx]
                    seg[-len(valid_data):] = valid_data
                    # 或者第一采样点前补相同的以避免不连续
                elif end_idx > len(smv):
                    valid_data = smv[start_idx:]
                    seg[:len(valid_data)] = valid_data
                else:
                    seg = smv[start_idx:end_idx]
                
                X_samples.append(seg)
                y_samples.append(label)
    
    return X_samples, y_samples

def encode_smv_to_gramian(smv_seq):
    # 归一化 s = 2*(smv - min)/(max - min) - 1
    s_min = np.min(smv_seq)
    s_max = np.max(smv_seq)
    
    if s_max - s_min == 0:
        s = np.zeros_like(smv_seq)
    else:
        s = 2 * (smv_seq - s_min) / (s_max - s_min) - 1
        
    # 防止因精度导致略微超出 [-1, 1] 范围而 arccos 报错
    s = np.clip(s, -1.0, 1.0)
    
    # 极坐标角度 phi
    phi = np.arccos(s)
    
    # 构建 Gramian 矩阵
    gramian = np.zeros((SMV_SEG_LEN, SMV_SEG_LEN))
    for i in range(SMV_SEG_LEN):
        for j in range(SMV_SEG_LEN):
            # 将角合并后计算 Cos
            val = np.cos(phi[i] + phi[j])
            # Linear map [-1, 1] to [0, 255]
            gramian[i, j] = (val + 1) * 127.5
            
    # 转为 uint8
    return gramian.astype(np.uint8)

print("--- 开始 TBM 筛选 ---")
# 正常活动 ADL 作为负类 (0)
X_adl_smv, y_adl = preprocess_and_detect_tbm('ADL', 0)
print(f"ADL (正常) 提取到疑似摔倒窗口: {len(X_adl_smv)} 个")

# 跌倒活动 FALL 作为正类 (1)
X_fall_smv, y_fall = preprocess_and_detect_tbm('FALL', 1)
print(f"FALL (跌倒) 提取到疑似摔倒窗口: {len(X_fall_smv)} 个")

X_all_smv = X_adl_smv + X_fall_smv
y_all = y_adl + y_fall

if len(X_all_smv) == 0:
    print("没有筛选到任何样本，请检查阈值和数据读取是否正确！")
    exit()

print("--- 开始 Gramian 图像编码 ---")
X_images = []
for seq in X_all_smv:
    img = encode_smv_to_gramian(seq)
    X_images.append(img)
    
X_images = np.array(X_images)
# 扩充最后一个维度作为通道数 (单通道灰度图)
X_images = np.expand_dims(X_images, axis=-1)
y_all = np.array(y_all)

# One-hot 编码
y_all_cat = tf.keras.utils.to_categorical(y_all, num_classes=2)

# 打乱数据
indices = np.arange(len(X_images))
np.random.shuffle(indices)
X_images = X_images[indices]
y_all_cat = y_all_cat[indices]

print(f"总数据量: {len(X_images)}，输入形状: {X_images.shape}")

print("--- 构建并训练轻量级 CNN 模型 ---")

model = Sequential([
    # C1 层
    Conv2D(6, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(SMV_SEG_LEN, SMV_SEG_LEN, 1)),
    # S2 层
    MaxPooling2D(pool_size=(2, 2), strides=2),
    # C3 层
    Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'),
    # S4 层
    MaxPooling2D(pool_size=(2, 2), strides=2),
    
    Flatten(),
    
    # F5 层
    Dense(256, activation='relu'),
    Dropout(0.5),
    
    # F6 层
    Dense(256, activation='relu'),
    Dropout(0.5),
    
    # Output 层
    Dense(2, activation='softmax')
])

# 优化器
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 划分 80% 训练，20% 验证
# 在 50 个 epochs 内如果验证准确率不再提升，可使用 EarlyStopping 防止过拟合
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

history = model.fit(
    X_images, y_all_cat,
    validation_split=0.2,
    batch_size=32,
    epochs=50,
    callbacks=[early_stop]
)

print("--- 保存模型 ---")
model.save('best_fall_detector_lenet.h5')
print("模型已保存为 best_fall_detector_lenet.h5 (Keras HDF5 格式)，可用于 STM32Cube.AI 转换。")

print("--- 保存为 ONNX 格式 ---")
try:
    import tf2onnx
    # input_signature 可选，通常 tf2onnx.convert.from_keras 可以自动推断
    onnx_model, _ = tf2onnx.convert.from_keras(model, output_path="best_fall_detector_lenet.onnx")
    print("模型已同时保存为 best_fall_detector_lenet.onnx (ONNX 格式)。")
except Exception as e:
    print(f"转换为 ONNX 格式时出错：{e}")