from dataset import Trdg_Dataset
import utils
import tensorflow as tf
print(tf.version.VERSION)

ds = Trdg_Dataset('train')
ds = ds.get_ds()

val_ds = Trdg_Dataset('val').get_ds()

for index, (imgs, lbls) in enumerate(ds):

    y_true = tf.sparse.to_dense(lbls, default_value=91)
    print(y_true)
    break
