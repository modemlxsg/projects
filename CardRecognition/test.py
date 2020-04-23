import torch
from model import CardRecognition
from dataset import CardDataLoader, CardDataset
import time
import numpy as np

ds = CardDataset('test')
dl = CardDataLoader(ds)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CardRecognition()
model.load_state_dict(torch.load('./run/epoch_60.pth'))
model.to(device)
model.eval()

positive = 0
total = 0
times = []
for img, lbl in dl():
    img = img.to(device)
    lbl = lbl.to(device)

    start = time.time()
    out = model(img)
    times.append(time.time() - start)
    pred = torch.argmax(out, dim=1)

    print(pred)
    print(lbl)
    print('\n')

    positive += (pred == lbl).sum().float()
    total += img.shape[0]

print(f"acc : {positive/total}, time : {np.mean(times[1:])}")
