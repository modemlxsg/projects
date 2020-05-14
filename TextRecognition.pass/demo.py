import os
from model import CRNN
import utils
import cv2
import tensorflow as tf
import glob
import numpy as np

model = CRNN(92)
model.load_weights('./save/epoch_50.h5')

imgs = glob.glob(os.path.join(os.getcwd(), 'img/*.jpg'))

data = []
for img in imgs:
    im = cv2.imread(img)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (100, 32))
    im = im[:, :, np.newaxis]
    data.append(im)

data = np.array(data)
print(data.shape)

out = model(data)
print(out.shape)

decoded = utils.Decoder().decode(out, 'greedy', 91)
print(decoded)

strs = utils.Decoder().convert2str(decoded)
print(strs)
