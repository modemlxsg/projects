import torch
import torchvision
from model import CardRecognition

transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(3),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])
ds_train = torchvision.datasets.FashionMNIST(
    '//linx-nas/datasets', transform=transform)
dl_train = torch.utils.data.DataLoader(
    ds_train, batch_size=32, shuffle=True, num_workers=0)

ds_test = torchvision.datasets.FashionMNIST(
    '//linx-nas/datasets', transform=transform, train=False)
dl_test = torch.utils.data.DataLoader(
    ds_test, batch_size=32, shuffle=True, num_workers=0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = CardRecognition()
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.1)

epochs = 10
for epoch in range(epochs):

    # train
    model.train()
    for step, (img, lbl) in enumerate(dl_train):
        img = img.to(device)
        lbl = lbl.to(device)

        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred, lbl)
        loss.backward()
        optimizer.step()

        print_str = f"step : {step+1}"
        print(print_str, end="")
        print("\b" * (len(print_str) * 2), end="", flush=True)

    # val
    model.eval()
    positive = 0
    total = 0
    for img, lbl in dl_test:
        img = img.to(device)
        lbl = lbl.to(device)

        out = model(img)
        pred = torch.argmax(out, dim=1)

        positive += (pred == lbl).sum().float()
        total += img.shape[0]

    acc = positive / total
    print(
        f"epoch : {epoch+1}/{epochs}, loss : {loss.detach().cpu().numpy()}, positive: {positive}, acc : {acc}, lr : {scheduler.get_lr()}")
    scheduler.step()
