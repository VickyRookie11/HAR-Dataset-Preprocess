import os
import pandas as pd
import numpy as np
from scipy import signal
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict

# --- 1. 配置参数 ---
DATA_DIR = './fall_detection_data/Dataset_no_heart/'
OUTPUT_DIR = './stm32ai_dataset'
ORIGINAL_SR = 20
TARGET_SR = 50
WINDOW_SIZE = 48
STEP_SIZE = 24

def process_file(filepath, label):
    try:
        df = pd.read_csv(filepath, header=0, names=['t', 'x', 'y', 'z', 'a', 'sensor'])
        df['sensor'] = df['sensor'].astype(str).str.lower()
        df = df[df['sensor'] == 'acc']
        
        if df.empty:
            return None
            
        # 移除可能存在的NaN
        df = df.dropna(subset=['x', 'y', 'z'])
        data = df[['x', 'y', 'z']].values.astype(np.float32)
        
        if len(data) < 2:
            return None
        
        num_samples = int(len(data) * TARGET_SR / ORIGINAL_SR)
        if num_samples < WINDOW_SIZE:
             return None 
             
        resampled_data = signal.resample(data, num_samples, axis=0)
        
        # === 核心优化：跌倒数据只取最大冲击(SMV Peak)窗口，消除非跌倒(静止)时间的错误标签 ===
        if label == 1:
            # 计算合加速度(SMV)以找到冲击点
            smv = np.sqrt(np.sum(resampled_data**2, axis=1))
            peak_idx = np.argmax(smv)
            
            # 以峰值为中心切割一个窗口
            start_idx = peak_idx - (WINDOW_SIZE // 2)
            
            # 防止越界
            if start_idx < 0:
                start_idx = 0
            if start_idx + WINDOW_SIZE > len(resampled_data):
                start_idx = len(resampled_data) - WINDOW_SIZE
            
            if start_idx < 0: # 如果总长度还是不够
                return None
                
            return np.array([resampled_data[start_idx:start_idx + WINDOW_SIZE]])
        else:
            # 日常活动 (ADL) 正常滑动窗口采样
            windows = []
            for start in range(0, len(resampled_data) - WINDOW_SIZE + 1, STEP_SIZE):
                window = resampled_data[start:start + WINDOW_SIZE]
                windows.append(window)
                
            return np.array(windows)
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    print("=== 开始数据提取与预处理 ===")
    user_windows = defaultdict(list)
    user_labels = defaultdict(list)
    
    for label_name, label in [('adl', 0), ('fall', 1)]:
        dir_path = os.path.join(DATA_DIR, label_name)
        if not os.path.exists(dir_path):
            continue
            
        for user_id in os.listdir(dir_path):
            user_path = os.path.join(dir_path, user_id)
            if not os.path.isdir(user_path):
                continue
                
            for filename in os.listdir(user_path):
                if not filename.endswith('.csv'):
                    continue
                
                filepath = os.path.join(user_path, filename)
                windows = process_file(filepath, label)
                
                if windows is not None and len(windows) > 0:
                    for w in windows:
                        user_windows[user_id].append(w)
                        user_labels[user_id].append(label)
                        
    all_users = list(user_windows.keys())
    if not all_users:
        print("未找到任何有效数据！")
        return
        
    np.random.seed(42)
    np.random.shuffle(all_users)
    
    train_users, temp_users = train_test_split(all_users, test_size=0.3, random_state=42)
    val_users, test_users = train_test_split(temp_users, test_size=0.5, random_state=42)
    
    def gather_data(user_list):
        X, y = [], []
        for u in user_list:
            X.extend(user_windows[u])
            y.extend(user_labels[u])
        return np.array(X), np.array(y)
        
    X_train, y_train = gather_data(train_users)
    X_val, y_val = gather_data(val_users)
    X_test, y_test = gather_data(test_users)
    
    means = np.mean(X_train, axis=(0, 1))
    stds = np.std(X_train, axis=(0, 1))
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'gmp_norm_params.json'), 'w') as f:
        json.dump({'mean': means.tolist(), 'std': stds.tolist()}, f)
        
    def save_split(X, y, split_name):
        if len(X) == 0: return
        X_norm = (X - means) / (stds + 1e-8)
        # Flatten for CSV matching shape required: N x 144
        X_flat = X_norm.reshape(X_norm.shape[0], -1)
        
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        np.savetxt(os.path.join(split_dir, 'X.csv'), X_flat, delimiter=',')
        np.savetxt(os.path.join(split_dir, 'y.csv'), y, delimiter=',', fmt='%d')

    print("保存数据集 CSV ...")
    save_split(X_train, y_train, 'train')
    save_split(X_val, y_val, 'val')
    save_split(X_test, y_test, 'test')
    
    print(f"数据处理完成，储存在 {OUTPUT_DIR}")

if __name__ == '__main__':
    main()