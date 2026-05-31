import os
import subprocess
import sys
import numpy as np
import tensorflow as tf

def run_onnx2tf(onnx_path, tf_model_dir):
    """Converts the ONNX model to TensorFlow SavedModel using onnx2tf command line."""
    print(f"\n=== Converting ONNX to TensorFlow SavedModel ===")
    cmd = [
        sys.executable,
        "-m",
        "onnx2tf",
        "-i", onnx_path,
        "-o", tf_model_dir,
        "-osd" # Output SignatureDefs
    ]
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("onnx2tf conversion failed!")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        raise RuntimeError("onnx2tf failed to convert ONNX model.")
    print("onnx2tf conversion complete!")


def convert_to_tflite_fp32(tf_model_dir, tflite_path):
    """Converts TensorFlow SavedModel to standard FP32 TFLite model."""
    print(f"\n=== Converting SavedModel to FP32 TFLite ===")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"FP32 TFLite model saved to {tflite_path}")


def convert_to_tflite_int8(tf_model_dir, tflite_quant_path, val_x_path):
    """Converts SavedModel to INT8-quantized TFLite model using representative dataset."""
    print(f"\n=== Converting SavedModel to INT8 Quantized TFLite ===")
    
    # 1. Load validation data to use as calibration dataset
    if not os.path.exists(val_x_path):
        raise FileNotFoundError(f"Validation data not found at {val_x_path}")
        
    val_x = np.load(val_x_path)
    print(f"Loaded calibration data from {val_x_path} of shape {val_x.shape}")
    
    # 2. Define representative dataset generator
    def representative_dataset():
        # Subsample 100 random windows for calibration
        np.random.seed(42)
        num_calibration_samples = min(100, len(val_x))
        indices = np.random.choice(len(val_x), size=num_calibration_samples, replace=False)
        
        for idx in indices:
            # ONNX export format has input shape (batch_size, channels, sequence_length) -> (1, 1, 1000)
            # The converted TensorFlow SavedModel expects channel-last format -> (1, 1000, 1)
            sample = val_x[idx : idx + 1] # shape (1, 1, 1000)
            sample = np.transpose(sample, (0, 2, 1)) # shape (1, 1000, 1)
            yield [sample.astype(np.float32)]

    # 3. Configure converter for full integer post-training quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Restrict operations to standard 8-bit integer ops
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Keep input and output as float32 for easy application integration (removes manual dequant/quant step in Android)
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_quant_model = converter.convert()
    
    with open(tflite_quant_path, "wb") as f:
        f.write(tflite_quant_model)
    print(f"INT8 Quantized TFLite model saved to {tflite_quant_path}")


def main():
    model_dir = "models"
    onnx_path = os.path.join(model_dir, "resnet10_5gamers.onnx")
    tf_model_dir = os.path.join(model_dir, "saved_model")
    tflite_fp32_path = os.path.join(model_dir, "resnet10_5gamers.tflite")
    tflite_int8_path = os.path.join(model_dir, "resnet10_5gamers_quant.tflite")
    val_x_path = "data/processed/val_x.npy"
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}. Please run src/export_onnx.py first.")
        
    # Clean up old SavedModel directory if it exists to avoid conversion conflicts
    import shutil
    if os.path.exists(tf_model_dir):
        print(f"Cleaning up old SavedModel directory at {tf_model_dir}...")
        shutil.rmtree(tf_model_dir, ignore_errors=True)

    # 1. Convert ONNX to TensorFlow SavedModel
    run_onnx2tf(onnx_path, tf_model_dir)
    
    # 2. Convert to FP32 TFLite
    convert_to_tflite_fp32(tf_model_dir, tflite_fp32_path)
    
    # 3. Convert to INT8 Quantized TFLite
    convert_to_tflite_int8(tf_model_dir, tflite_int8_path, val_x_path)
    
    # 4. Print and compare sizes
    print("\n=== Model Size Comparison ===")
    pytorch_path = os.path.join(model_dir, "best_model.pth")
    
    if os.path.exists(pytorch_path):
        print(f"PyTorch Checkpoint:      {os.path.getsize(pytorch_path) / 1024 / 1024:.2f} MB")
    print(f"ONNX Model:              {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
    print(f"TFLite FP32 Model:       {os.path.getsize(tflite_fp32_path) / 1024 / 1024:.2f} MB")
    print(f"TFLite INT8 Quant Model: {os.path.getsize(tflite_int8_path) / 1024 / 1024:.2f} MB")
    
    # Footprint reduction ratio
    reduction = (1.0 - (os.path.getsize(tflite_int8_path) / os.path.getsize(tflite_fp32_path))) * 100
    print(f"Footprint Reduction via INT8 Quantization: {reduction:.2f}%")


if __name__ == "__main__":
    main()
