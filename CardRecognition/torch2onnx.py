import torch
import torchvision

dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
model = torchvision.models.shufflenet_v2_x1_0(pretrained=True).cuda()

torch.onnx.export(model, dummy_input, 'shuffleNetV2.onnx')
