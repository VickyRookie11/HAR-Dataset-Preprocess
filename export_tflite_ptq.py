import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob

print("Start INT8 Model Quantization...")

# 1. Load original best model
model = tf.keras.models.load_model('fall_detect.h5')

# 2. Prepare representative dataset generator for calibration
def representative_dataset():
    train_csvs = glob.glob('stm32ai_dataset/train/X*.csv')
    sample_count = 0
    for csv_file in train_csvs[:20]:
        df = pd.read_csv(csv_file, header=None)
        # In our project, X.csv is already just the features (no label)
        X = df.values.astype(np.float32)
        X = X.reshape(-1, 48, 3, 1)
        for i in range(len(X)):
            yield [X[i:i+1]]
            sample_count += 1
            if sample_count >= 500:
                return

# 3. Configure TFLite Converter for INT8 Post-Training Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Restrict operations to INT8 standard
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Keep input/output as float32 so C code does not have to change type
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

# 4. Perform quantization
tflite_quant_model = converter.convert()

# 5. Save quantized model
out_name = 'fall_detect_quant.tflite'
with open(out_name, 'wb') as f:
    f.write(tflite_quant_model)

print(f"✅ INT8 TFLite Exported: {out_name}")
print(f"📦 Compressed Size: {os.path.getsize(out_name) / 1024:.2f} KB (Approx 1/4 of ONNX)")
