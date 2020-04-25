import numpy as np
import onnxruntime as rt
import time
# from dataset import CardDataset
# ds = CardDataset('test')

# img, lbl = ds.__getitem__(0)

# img = np.transpose(img, (2, 0, 1))
# img = np.expand_dims(img, 0)
# print(img.shape)


sess = rt.InferenceSession('./build/cardrecog_b16_v1(optim).onnx')
provider = rt.get_available_providers()
print(provider)
device = rt.get_device()
print(device)
# input_name = sess.get_inputs()[0].name

# start = time.time()
# pred = sess.run(None, {input_name: img.astype(np.float32)})[0]

# print(pred, type(pred))
# print(f'time : {time.time()-start}, FPS : {1/(time.time()-start)}')
