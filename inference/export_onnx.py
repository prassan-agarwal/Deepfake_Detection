import os
import torch
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_model import DeepfakeHybridModel

def export_to_onnx(model_path="best_hybrid_model.pth", output_path="inference_results/deepfake_model.onnx"):
    print(f"Loading PyTorch model from {model_path}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DeepfakeHybridModel()
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Please train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    
    # Create a dummy input tensor matching the sequence input shape
    # Shape: [Batch_size, Sequence_length, Channels, Height, Width]
    # We use Batch Size 1 for standard inference
    dummy_input = torch.randn(1, 16, 3, 224, 224, device=device)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting model to {output_path}...")
    
    try:
        torch.onnx.export(
            model,               # model being run
            dummy_input,         # model input (or a tuple for multiple inputs)
            output_path,         # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=14,    # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=['input_sequence'],   # the model's input names
            output_names=['fake_probability_logit'], # the model's output names
            dynamic_axes={'input_sequence': {0: 'batch_size', 1: 'sequence_length'},    # variable length axes
                          'fake_probability_logit': {0: 'batch_size'}}
        )
        print("Optimization complete! ONNX model exported successfully.")
    except Exception as e:
        print(f"ONNX Export Failed: {str(e)}")
    
if __name__ == "__main__":
    export_to_onnx()
