import tensorflow as tf
import os
import yaml
import cv2
import utils


class Trdg_Dataset:

    def __init__(self, mode):
        self.config = utils.load_config()
        self.root_dir = self.config['dataset']['trdg']['root_dir']
        self.imgs, self.labels = self.read_data()
        sp = round(len(self.imgs) * 0.9)

        if mode == 'train':
            self.imgs = self.imgs[0:sp]
            self.labels = self.labels[0:sp]
        elif mode == 'val':
            self.imgs = self.imgs[sp:]
            self.labels = self.labels[sp:]
        else:
            raise Exception('mode error!')

        lexicon = utils.load_lexicon()
        self.lexicon = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
            lexicon, tf.range(len(lexicon))
        ),  -1)

    def get_ds(self):
        ds_img = tf.data.Dataset.from_tensor_slices(self.imgs)
        ds_lbl = tf.data.Dataset.from_tensor_slices(self.labels)
        ds_img = ds_img.map(self.load_img)
        ds = tf.data.Dataset.zip((ds_img, ds_lbl))
        ds = ds.batch(self.config['train']['batch_size'])
        ds = ds.map(self.label_encode)
        ds = ds.shuffle(50)
        return ds

    def label_encode(self, imgs, labels):
        chars = tf.strings.unicode_split(labels, input_encoding="UTF-8")
        mapped_label = tf.ragged.map_flat_values(self.lexicon.lookup, chars)
        sparse_label = mapped_label.to_sparse()
        sparse_label = tf.cast(sparse_label, tf.int32)

        return imgs, sparse_label

    def load_img(self, path):
        im = tf.io.read_file(path)
        im = tf.io.decode_jpeg(im, channels=1)
        im = tf.image.convert_image_dtype(im, tf.float32)
        im = tf.image.resize(im, [32, 100])
        return im

    def read_data(self):
        imgs = []
        labels = []

        for root in self.root_dir:
            with open(os.path.join(root, "labels.txt"), "r", encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    img, label = line.split(' ')
                    img = os.path.join(root, img)
                    imgs.append(img)

                    label = label.strip('\n')
                    labels.append(label)
        return imgs, labels


class Mj_Dataset:
    def __init__(self, mode):
        self.config = utils.load_config()
        self.root_dir = self.config['dataset']['SynthText']['root_dir']
        self.item_num = self.config['dataset']['SynthText'][f'{mode}_num']
        self.mode = mode

        lexicon = utils.load_lexicon()
        self.lexicon = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                lexicon, tf.range(len(lexicon))), -1
        )

        self.paths = self.getPaths()

    def getDS(self):
        ds = tf.data.Dataset.from_tensor_slices(self.paths)
        ds = ds.map(self.readData)
        ds = ds.batch(self.config['train']['batch_size']).map(
            self.label_encode)
        return ds

    def readData(self, path):
        label = tf.strings.split(path, '/')[-1]
        label = tf.strings.split(label, '_')[1]
        im = tf.io.read_file(path)
        im = tf.io.decode_jpeg(im, channels=1)
        im = tf.image.convert_image_dtype(im, tf.float32)
        im = tf.image.resize(im, [32, 100])
        # im = im / 255.0

        return im, label

    def label_encode(self, imgs, labels):
        chars = tf.strings.unicode_split(labels, input_encoding="UTF-8")
        mapped_label = tf.ragged.map_flat_values(self.lexicon.lookup, chars)
        sparse_label = mapped_label.to_sparse()
        sparse_label = tf.cast(sparse_label, tf.int32)

        return imgs, sparse_label

    def getPaths(self):
        with open(os.path.join(self.root_dir, f"annotation_{self.mode}.txt")) as f:
            lines = f.readlines()

        i, items = 0, []
        for line in lines:
            if i >= self.item_num:
                break
            fullpath = os.path.join(self.root_dir, line.split(' ')[0])
            if os.path.exists(fullpath):
                items.append(fullpath)
                i += 1
        return items
