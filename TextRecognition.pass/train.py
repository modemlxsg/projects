import os
import time
import utils
import tensorflow as tf
from tensorflow import keras
import numpy as np
from dataset import Trdg_Dataset, Mj_Dataset
from model import CRNN
from losses import CTCLoss
from metric import OCR_Accuracy

# config
config = utils.load_config()

# data
train_ds = Trdg_Dataset('train').get_ds()
val_ds = Trdg_Dataset('val').get_ds()

# train_ds = Mj_Dataset('train').getDS()
# val_ds = Mj_Dataset('val').getDS()

# model
nclass = config['crnn']['nClass']
lr = config['train']['lr']
backbone = config['crnn']['backbone']
num_rnn = config['crnn']['num_rnn']
checkpoint = config['train']['checkpoint']

model = CRNN(nclass, backbone, num_rnn)

if checkpoint is not None:
    print("load weights!")
    model.load_weights(checkpoint)


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, 20000, 0.5)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
criterion = CTCLoss(logits_time_major=False)
metric = OCR_Accuracy()

epochs = config['train']['epochs']
for epoch in range(epochs):
    metric.reset_states()
    start = time.time()

    # train
    for step, (imgs, labels) in enumerate(train_ds):
        y_true = labels  # sparse_tensor

        with tf.GradientTape() as tape:
            y_pred = model(imgs)
            loss = criterion(y_true, y_pred)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        metric.update_state(y_true, y_pred)

        print_str = f"step : {step+1}"
        print(print_str, end="")
        print("\b" * (len(print_str) * 2), end="", flush=True)

    train_acc = metric.result()

    # save model
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    if (epoch+1) % 50 == 0:
        model.save_weights(f'./checkpoints/epoch_{epoch+1}.h5')

    # val
    metric.reset_states()
    for step, (imgs, labels) in enumerate(val_ds):
        out = model(imgs)
        metric.update_state(labels, out)
    val_acc = metric.result()

    # print
    print(
        f"epoch : {epoch} , loss : {loss} , train_acc : {train_acc}, val_acc : {val_acc}, time : {time.time()-start}")
