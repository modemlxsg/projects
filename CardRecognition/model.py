import torch
import torchvision


class CardRecognition(torch.nn.Module):
    def __init__(self):
        super(CardRecognition, self).__init__()
        self.backbone = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        self.backbone.fc = torch.nn.Linear(
            in_features=1024, out_features=53, bias=True)

    def forward(self, inputs):
        out = self.backbone(inputs)
        return out


if __name__ == "__main__":
    model = CardRecognition()
    inp = torch.randn(10, 3, 224, 224)
    print(model.forward(inp).shape)
