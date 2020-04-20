import torch
from model import CardRecognition
from dataset import CardDataLoader, CardDataset


dataloader = CardDataLoader()
model = CardRecognition()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)


epochs = 10
for epoch in range(epochs):
    model.train()
    for imgs, labels in dataloader():
        optimizer.zero_grad()

        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        print(loss)
