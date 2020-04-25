import torch
import torchvision
from model import CardRecognition

import onnx
from onnx import optimizer


def export():
    model = CardRecognition()
    model.load_state_dict(torch.load("./run/epoch_60.pth"))
    model.to('cuda:0')

    batch_size = 16
    dummy_input = torch.randn(batch_size, 3, 224, 224, device='cuda')

    torch.onnx.export(
        model, dummy_input, './build/cardrecog_b{}_v1.onnx'.format(batch_size),
        input_names=['input_1'],
        output_names=['output_1'],
        dynamic_axes={'input_1': {0: 'batch_size'},
                      'output_1': {0: 'batch_size'}},
        keep_initializers_as_inputs=True)


def optim():
    onnx_model = onnx.load("./build/cardrecog_b16_v1.onnx")
    # optim
    passes = ["extract_constant_to_initializer",
              "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(onnx_model, passes)

    onnx.save(optimized_model, './build/cardrecog_b16_v1(optim).onnx')


if __name__ == "__main__":
    export()
    optim()
