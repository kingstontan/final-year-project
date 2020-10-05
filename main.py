import os
import cv2
import numpy as np
from tqdm import tqdm

REBUILD_DATA = True

labels = []


class lineLabel():
    def __init__(self, lineid, transcript):
        lineid = lineid.split('-')
        self.rootdir = lineid[0]
        self.subdir = lineid[1]
        self.id = lineid[2]
        self.transcript = transcript


class dataClass():
    training_data = []

    def read_file(self):
        with open("data/lines.txt") as f:
            raw = f.readlines()

        lines = []

        for x in raw:
            lines.append(x.split())

        for x in lines:
            labels.append(lineLabel(x[0], x[8]))

    def make_training_data(self):

        widths = []
        heights = []

        count = 0

        for x in tqdm(labels):
            count += 1
            if count == 100:
                break
            path = "data/line_images/%s/%s-%s/%s-%s-%s.png" % (
                x.rootdir, x.rootdir, x.subdir, x.rootdir, x.subdir, x.id)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            #img = cv2.resize(img, (0, 0), fx=0.095, fy=0.095)  # reduce image size to only 10%
            h, w = img.shape[0:2]
            heights.append(h)
            widths.append(w)

            self.training_data.append([np.array(img), x])

        max_height = 32
        max_width = 256

        # for x in tqdm(self.training_data):
        #
        #     if x[0].shape[0] % 2 == 0 and x[0].shape[1] % 2 == 0:
        #         x[0] = cv2.copyMakeBorder(x[0], int((max_height - x[0].shape[0]) / 2), int((max_height - x[0].shape[0]) / 2), int((max_width - x[0].shape[1]) / 2), int((max_width - x[0].shape[1]) / 2), cv2.BORDER_CONSTANT, value=255)
        #     elif x[0].shape[0] % 2 != 0 and x[0].shape[1] % 2 == 0:
        #         x[0] = cv2.copyMakeBorder(x[0], int((max_height - x[0].shape[0]) / 2) + 1, int((max_height - x[0].shape[0]) / 2), int((max_width - x[0].shape[1]) / 2), int((max_width - x[0].shape[1]) / 2), cv2.BORDER_CONSTANT, value=255)
        #     if x[0].shape[0] % 2 == 0 and x[0].shape[1] % 2 != 0:
        #         x[0] = cv2.copyMakeBorder(x[0], int((max_height - x[0].shape[0]) / 2), int((max_height - x[0].shape[0]) / 2), int((max_width - x[0].shape[1]) / 2) + 1, int((max_width - x[0].shape[1]) / 2), cv2.BORDER_CONSTANT, value=255)
        #     if x[0].shape[0] % 2 != 0 and x[0].shape[1] % 2 != 0:
        #         x[0] = cv2.copyMakeBorder(x[0], int((max_height - x[0].shape[0]) / 2) + 1, int((max_height - x[0].shape[0]) / 2), int((max_width - x[0].shape[1]) / 2) + 1, int((max_width - x[0].shape[1]) / 2), cv2.BORDER_CONSTANT, value=255)

        for x in tqdm(self.training_data):
            # ret, basic210 = cv2.threshold(x[0], 210, 255, cv2.THRESH_BINARY_INV)
            #
            # gaussian199 = cv2.adaptiveThreshold(x[0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
            # # gaussian210 = cv2.adaptiveThreshold(x[0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 210, 5)
            #
            # mean199 = cv2.adaptiveThreshold(x[0], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
            # # mean210 = cv2.adaptiveThreshold(x[0], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 210, 5)
            #
            #
            # cv2.imshow('basic210',basic210)
            #
            # cv2.imshow('gaussian199',gaussian199)
            # # cv2.imshow('gaussian210',gaussian210)
            #
            # cv2.imshow('mean199', mean199)
            # # cv2.imshow('mean210', mean210)
            # cv2.waitKey(0)
            ret, b195 = cv2.threshold(x[0], 195, 255, cv2.THRESH_BINARY_INV)
            ret, b165 = cv2.threshold(x[0], 165, 255, cv2.THRESH_BINARY_INV)
            ret, b175 = cv2.threshold(x[0], 175, 255, cv2.THRESH_BINARY_INV)
            ret, b185 = cv2.threshold(x[0], 185, 255, cv2.THRESH_BINARY_INV)


            cv2.imshow('b195', b195)
            cv2.imshow('b165', b165)
            cv2.imshow('b175', b175)
            cv2.imshow('b185', b185)



            cv2.waitKey(0)

        # tqdm(np.random.shuffle(self.training_data))
        tqdm(np.save("td210.npy", self.training_data))

        # for x in tqdm(self.training_data):
        #     ret, x[0] = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                   cv2.THRESH_BINARY, 199, 5)
        #
        # # tqdm(np.random.shuffle(self.training_data))
        # tqdm(np.save("tdgaussian 199.npy", self.training_data))
        #
        # for x in tqdm(self.training_data):
        #     ret, x[0] = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                       cv2.THRESH_BINARY, 199, 5)
        #
        # # tqdm(np.random.shuffle(self.training_data))
        # tqdm(np.save("tdmean 199.npy", self.training_data))

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 36 by 240
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # 18 by 120
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        # 9 by 60
        self.fc1 = nn.Linear(9 * 60 * 64, 1000)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        return out

if __name__ == '__main__':

    if REBUILD_DATA:
        data = dataClass()
        data.read_file()
        data.make_training_data()

    training_data = np.load("training_data.npy", allow_pickle=True)
    print((training_data[0, 1]).transcript)
    cv2.imshow('image', training_data[0, 0])
    cv2.waitKey(0)
