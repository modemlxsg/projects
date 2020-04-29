import os
import cv2
import glob
import numpy as np
import info


class ExtractRoi(object):
    def __init__(self):
        self.imgs = glob.glob(os.path.join('F:\\img\\saved', "*.jpg"))

        text_rois = info.PS_ROI['SPIN&GO']
        self.text_rois = text_rois['POS0_TEXT'] + \
            text_rois['POS1_TEXT'] + \
            text_rois['POS2_TEXT'] + text_rois['POT_TEXT'] + \
            text_rois['TABEL_INFO']
        self.img_size = text_rois['IMG_SIZE']

        self.bet_rois = text_rois['BET0'] + \
            text_rois['BET1'] + text_rois['BET2']

    def extractText(self):
        count = 1

        save_path = os.path.join('F:\\img\\saved', "extractText")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for i in range(len(self.imgs)):
            img = cv2.imread(self.imgs[i])
            img = cv2.resize(
                img, (self.img_size[1], self.img_size[0]))

            for text_roi in self.text_rois:
                leftTop = text_roi[0]
                rightButtom = text_roi[1]
                roi = img[leftTop[1]:rightButtom[1],
                          leftTop[0]:rightButtom[0], :]

                filename = "%05d" % count
                cv2.imwrite(os.path.join(save_path, f"{filename}.jpg"), roi)
                count += 1

            for bet_roi in self.bet_rois:
                leftTop = bet_roi[0]
                rightButtom = bet_roi[1]
                roi = img[leftTop[1]:rightButtom[1],
                          leftTop[0]: rightButtom[0], :]

                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                low = np.array(info.PS_ROI['SPIN&GO']['HSV_LOW'])
                high = np.array(info.PS_ROI['SPIN&GO']['HSV_HIGH'])
                dst = cv2.inRange(hsv_roi, low, high)

                filename = "%05d" % count
                cv2.imwrite(os.path.join(save_path, f"{filename}.jpg"), roi)
                count += 1

                filename = "%05d" % count
                cv2.imwrite(os.path.join(save_path, f"{filename}.jpg"), dst)
                count += 1

            print(f"{i+1}/{len(self.imgs)}")


if __name__ == "__main__":
    c = ExtractRoi()
    c.extractText()
