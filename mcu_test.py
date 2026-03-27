import serial
import struct
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

# ==========================================
# 串口配置 (运行前请根据设备管理器修改 COM 口)
# ==========================================
COM_PORT = 'COM13'  # <-- 请修改为 STM32 的实际串口号
BAUD_RATE = 115200

# ==========================================
# 加载验证/测试数据
# ==========================================
print("Loading test dataset...")
X_test = pd.read_csv('stm32ai_dataset/test/X.csv', header=None).values.astype(np.float32)
y_test = pd.read_csv('stm32ai_dataset/test/y.csv', header=None).values.astype(int).flatten()

num_samples = len(y_test)
print(f"Total test samples: {num_samples}")

correct = 0
fp = 0 # 误报 (把日常当跌倒)
fn = 0 # 漏报 (真跌倒没测出来)

try:
    with serial.Serial(COM_PORT, BAUD_RATE, timeout=2.0) as ser:
        print(f"Connected to {COM_PORT} at {BAUD_RATE} bps.")
        time.sleep(1) # 等待单片机复位
        ser.reset_input_buffer()

        for i in tqdm(range(num_samples), desc="MCU Testing"):
            # 1. 取一个样本 (144个float32), 转化为 576 字节流
            sample_data = X_test[i]
            true_label = y_test[i]
            
            byte_data = struct.pack('<144f', *sample_data)
            
            # 2. 发送给 STM32
            ser.write(byte_data)
            
            # 3. 接收 STM32 推理返回的 2 个 float32 概率 (8 字节)
            rx_data = ser.read(8)
            if len(rx_data) != 8:
                print(f"\n[Error] Sample {i}: Failed to receive 8 bytes. Got {len(rx_data)} bytes.")
                continue
                
            pred_probs = struct.unpack('<2f', rx_data)
            pred_label = 1 if pred_probs[1] > 0.5 else 0 # 根据阈值判断
            
            # 4. 统计结果
            if pred_label == true_label:
                correct += 1
            else:
                if true_label == 0 and pred_label == 1:
                    fp += 1
                elif true_label == 1 and pred_label == 0:
                    fn += 1

            # 【修复卡死/死锁的关键】：
            # PC 端拿到前一帧结果后，稍微等一下，给 STM32 单片机重新开启下一轮 UART 测试接收的时间。
            # 如果不加延时，PC 发送比单片机开启 DMA 还要快，会导致单片机 UART 丢字节、发生 OverRun Error (ORE)，进而彻底死机。
            time.sleep(0.02)

except serial.SerialException as e:
    print(f"Serial Error: {e}")
    print("请确保单片机已插上，且串口号正确！")

# ==========================================
# 输出报告
# ==========================================
print("\n========== MCU HIL 硬件在环测试结果 ==========")
print(f"总测试样本数: {num_samples}")
print(f"MCU 侧准确率: {correct / num_samples * 100:.2f}%")
print(f"误报次数(FP): {fp} (影响体验)")
print(f"漏报次数(FN): {fn} (最致命！)")
print("==============================================")
