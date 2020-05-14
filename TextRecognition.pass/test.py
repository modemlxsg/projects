import tensorflow as tf
from dataset import Trdg_Dataset
from model import CRNN
from utils import Decoder

ds = Trdg_Dataset('train').get_ds()

model = CRNN(92)
model.load_weights("./save/epoch_50.h5")


decoder = Decoder()

for imgs, lbls in ds:
    out = model.predict(imgs)

    lbls = tf.sparse.to_dense(lbls, default_value=91)
    lbls_str = decoder.convert2str(lbls)
    print(lbls_str)

    decoded = decoder.decode(out, 'greedy', 91)
    strs = decoder.convert2str(decoded)
    print(strs)

    break
