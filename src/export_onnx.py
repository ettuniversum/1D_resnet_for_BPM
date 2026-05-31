import os
import torch
from model import resnet10_1d

def main():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    pytorch_path = os.path.join(model_dir, "best_model.pth")
    onnx_path = os.path.join(model_dir, "resnet10_5gamers.onnx")
    
    print("=== Exporting PyTorch Model to ONNX ===")
    
    # 1. Load the model architecture
    model = resnet10_1d(num_classes=1)
    
    # 2. Load the trained checkpoint
    if not os.path.exists(pytorch_path):
        raise FileNotFoundError(
            f"Trained PyTorch model not found at {pytorch_path}. "
            "Please run 'python src/train.py' first to train the model."
        )
    
    model.load_state_dict(torch.load(pytorch_path, map_location="cpu"))
    model.eval()
    
    # 3. Create dummy input matching the model input shape (batch_size, channels, sequence_length)
    # Batch size = 1, Channels = 1, Sequence length = 1000 (10 seconds at 100Hz)
    dummy_input = torch.randn(1, 1, 1000)
    
    # 4. Perform the export
    print(f"Exporting model to: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,        # Store the trained parameter weights inside the model file
        opset_version=11,          # Highly compatible with TensorFlow / TFLite converters
        do_constant_folding=True,  # Optimizes constant operations in the graph
        input_names=["input"],     # Name of input node in the ONNX graph
        output_names=["output"]    # Name of output node in the ONNX graph
    )
    print("ONNX model exported successfully!")


if __name__ == "__main__":
    main()
