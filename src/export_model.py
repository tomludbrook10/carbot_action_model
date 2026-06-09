import torch 
import onnx
from src.model_context import ModelContext


def export_model_to_onnx(output_model, data_dir, model_name, model_context: ModelContext):

    output_onnx_path = f'../models/{data_dir}/{model_name}.onnx'
    dummy_input = torch.randn(1, model_context.channels, model_context.height, model_context.width).to(model_context.device).to(model_context.data_type)
    torch.onnx.export(output_model, 
                        dummy_input, 
                        output_onnx_path, 
                        opset_version=18,
                        export_params=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamo=False)

    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

def export_model_to_onnx_v2(output_model, data_dir, model_name, model_context):
    output_model.eval()
    output_onnx_path = f'../models/{data_dir}/{model_name}.onnx'
    dummy_input = torch.randn(1, model_context.num_prev_observations + 1, model_context.channels, model_context.height, model_context.width).to(model_context.device).to(model_context.data_type)
    torch.onnx.export(output_model, 
                        dummy_input, 
                        output_onnx_path, 
                        opset_version=18,
                        export_params=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamo=False)

    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")
    return onnx_model