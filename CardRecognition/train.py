import torch
import os
import time
from model import CardRecognition
from dataset import CardDataLoader, CardDataset

train_ds = CardDataset('train')
val_ds = CardDataset('val')
test_ds = CardDataset('test')

train_dl = CardDataLoader(train_ds)
val_dl = CardDataLoader(val_ds)
test_dl = CardDataLoader(test_ds)


model = CardRecognition()
model.load_state_dict(torch.load("./run/epoch_300.pth"))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 60
for epoch in range(epochs):
    # train
    model.train()
    for step, (imgs, labels) in enumerate(train_dl()):
        optimizer.zero_grad()
        imgs = imgs.to(device)
        labels = labels.to(device)

        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        print_str = f"step : {step+1}"
        print(print_str, end="")
        print("\b" * (len(print_str) * 2), end="", flush=True)

    # val
    model.eval()
    positive = 0
    total = 0
    for imgs, labels in val_dl():
        imgs = imgs.to(device)
        labels = labels.to(device)
        out = model(imgs)
        pred = torch.argmax(out, dim=1)

        positive += (pred == labels).sum().float()
        total += imgs.shape[0]

    acc = positive / total
    print(
        f"epoch : {epoch+1}/{epochs}, loss : {loss.detach().cpu().numpy()}, positive: {positive}, acc : {acc}, lr : {scheduler.get_lr()}")
    scheduler.step()

    # save
    filepath = './run'
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    filename = f"epoch_{epoch+1}.pth"

    if (epoch+1) % 20 == 0:
        torch.save(model.state_dict(), os.path.join(filepath, filename))
