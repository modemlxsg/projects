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
    def __init__(self, dataset):
        self.ds = dataset
        weigths = [0.2 if label == 'back' else 1 for _, label in self.ds]
        self.sampler = torch.utils.data.WeightedRandomSampler(
            weigths, 64, replacement=True)

    def __call__(self):
        dl = torch.utils.data.DataLoader(
            self.ds, batch_size=16, num_workers=0, collate_fn=self.collate_fn, sampler=self.sampler)
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

    def __init__(self, mode):
        self.imgs = glob.glob(os.path.join(
            config.IMG_DIR, 'cards/*.jpg'))
        self.labels = glob.glob(os.path.join(
            config.IMG_DIR, 'cards/outputs/*.xml'))

        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.Affine(scale=(0.6, 1.1),
                       translate_percent=(-0.2, 0.2), rotate=(-25, 25))
        ])

        if mode == 'test':
            self.aug = iaa.Sequential([])

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.getLabel(index)
        img = cv2.imread(img)
        img = cv2.resize(img, (224, 224))
        img = img[:, :, ::-1]
        # img = self.aug(image=img)

        return img, label

    def __len__(self):
        return len(self.imgs)

    def getLabel(self, index):
        label = self.labels[index]
        root = etree.parse(label).getroot()  # type:etree.Element
        label = root.find(".//*/name").text

        return label
