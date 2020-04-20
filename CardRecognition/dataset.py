import torch
import config
import os
import glob
import cv2
from lxml import etree
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np


class CardDataLoader(object):
    def __init__(self):
        self.ds = CardDataset()

    def __call__(self):
        dl = torch.utils.data.DataLoader(
            self.ds, batch_size=32, num_workers=0, collate_fn=self.collate_fn)
        return dl

    def collate_fn(self, batch):
        tmp = []
        imgs = []
        for img, label in batch:
            index = config.CARDS.index(label.lower())
            tmp.append(index)
            imgs.append(img)

        imgs = torch.FloatTensor(imgs).permute(0, 3, 1, 2)

        return imgs, torch.LongTensor(tmp)


class CardDataset(torch.utils.data.Dataset):

    def __init__(self):
        imgs = glob.glob(os.path.join(
            config.IMG_DIR, 'extractCard/*.jpg'))
        self.imgs = imgs[0:666]
        self.aug = iaa.Sequential([

        ])

        labels = glob.glob(os.path.join(
            config.IMG_DIR, 'extractCard/outputs/*.xml'))
        self.labels = labels[0:666]

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.getLabel(index)
        img = cv2.imread(img)
        img = cv2.resize(img, (224, 224))
        img = img[:, :, ::-1]

        return img, label

    def __len__(self):
        return len(self.imgs)

    def getLabel(self, index):
        label = self.labels[index]
        root = etree.parse(label).getroot()  # type:etree.Element
        label = root.find(".//*/name").text

        return label
