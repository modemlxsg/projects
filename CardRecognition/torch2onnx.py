import torch
import torchvision
from model import CardRecognition

model = CardRecognition()
model.load_state_dict(torch.load("./run/epoch_60.pth"))
model.to('cuda:0')

batch_size = 16
dummy_input = torch.randn(batch_size, 3, 224, 224, device='cuda')

torch.onnx.export(model, dummy_input, f'cardrecog_b{batch_size}_v1.onnx')
