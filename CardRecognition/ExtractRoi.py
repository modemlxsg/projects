import os
import cv2
import config
import glob
import numpy as np


class ExtractRoi(object):
    def __init__(self):
        self.imgs = glob.glob(os.path.join(config.IMG_DIR, "*.jpg"))

        card_rois = config.PS_ROI['SPIN&GO']
        self.card_rois = card_rois['POS0'] + \
            card_rois['POS1'] + card_rois['POS2'] + card_rois['POT']
        self.card_img_size = card_rois['IMG_SIZE']

    def extractCard(self):
        count = 1

        save_path = os.path.join(config.IMG_DIR, "extractCard")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for i in range(len(self.imgs)):
            img = cv2.imread(self.imgs[i])
            img = cv2.resize(
                img, (self.card_img_size[1], self.card_img_size[0]))

            for card_roi in self.card_rois:
                leftTop = card_roi[0]
                rightButtom = card_roi[1]
                roi = img[leftTop[1]:rightButtom[1],
                          leftTop[0]:rightButtom[0], :]

                filename = "%05d" % count
                cv2.imwrite(os.path.join(save_path, f"{filename}.jpg"), roi)
                count += 1
            print(f"{i+1}/{len(self.imgs)}")


if __name__ == "__main__":
    c = ExtractRoi()
    c.extractCard()
