import os
import cv2
import numpy as np
from tqdm import tqdm

REBUILD_DATA = False

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

        for x in tqdm(labels):
            path = "data/line_images/%s/%s-%s/%s-%s-%s.png" % (
                x.rootdir, x.rootdir, x.subdir, x.rootdir, x.subdir, x.id)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)  # reduce image size to only 10%
            h, w = img.shape[0:2]
            heights.append(h)
            widths.append(w)

            self.training_data.append([np.array(img), x])

        max_height = 36
        max_width = 240

        for x in tqdm(self.training_data):

            if x[0].shape[0] % 2 == 0 and x[0].shape[1] % 2 == 0:
                x[0] = cv2.copyMakeBorder(x[0], int((max_height - x[0].shape[0]) / 2), int((max_height - x[0].shape[0]) / 2), int((max_width - x[0].shape[1]) / 2), int((max_width - x[0].shape[1]) / 2), cv2.BORDER_CONSTANT, value=255)
            elif x[0].shape[0] % 2 != 0 and x[0].shape[1] % 2 == 0:
                x[0] = cv2.copyMakeBorder(x[0], int((max_height - x[0].shape[0]) / 2) + 1, int((max_height - x[0].shape[0]) / 2), int((max_width - x[0].shape[1]) / 2), int((max_width - x[0].shape[1]) / 2), cv2.BORDER_CONSTANT, value=255)
            if x[0].shape[0] % 2 == 0 and x[0].shape[1] % 2 != 0:
                x[0] = cv2.copyMakeBorder(x[0], int((max_height - x[0].shape[0]) / 2), int((max_height - x[0].shape[0]) / 2), int((max_width - x[0].shape[1]) / 2) + 1, int((max_width - x[0].shape[1]) / 2), cv2.BORDER_CONSTANT, value=255)
            if x[0].shape[0] % 2 != 0 and x[0].shape[1] % 2 != 0:
                x[0] = cv2.copyMakeBorder(x[0], int((max_height - x[0].shape[0]) / 2) + 1, int((max_height - x[0].shape[0]) / 2), int((max_width - x[0].shape[1]) / 2) + 1, int((max_width - x[0].shape[1]) / 2), cv2.BORDER_CONSTANT, value=255)


        tqdm(np.random.shuffle(self.training_data))
        tqdm(np.save("training_data.npy", self.training_data))

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)


if __name__ == '__main__':

    if REBUILD_DATA:
        data = dataClass()
        data.read_file()
        data.make_training_data()

    training_data = np.load("training_data.npy", allow_pickle=True)
    print((training_data[0, 1]).transcript)
