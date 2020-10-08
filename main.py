import os
from numpy import argmax
import cv2 as cv
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def one_hot_encode(self, line):
        # define input string
        data = line
        # define universe of possible input values
        alphabet = """|ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/?:;'"-!#&*()+"""
        alphabet = alphabet + " "
        # define a mapping of chars to integers
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        int_to_char = dict((i, c) for i, c in enumerate(alphabet))
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in data]
        print(integer_encoded)
        # one hot encode
        onehot = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            onehot.append(letter)

        # for x in onehot:
        #     print(int_to_char[argmax(x)])

        return onehot

    def read_file(self):
        label_file = open("data/labels.txt","w")

        with open("data/lines.txt") as f:
            raw = f.readlines()

        lines = []

        for x in raw:
            lines.append(x.split())

        for x in lines:
            transcript = ''.join(map(str, x[8:])) #screw the space
            labels.append(lineLabel(x[0], transcript))



    def make_training_data(self):

        widths = []
        heights = []

        count = 0

        for x in tqdm(labels):
            count += 1
            if count == 10:
                break
            path = "data/line_images/%s/%s-%s/%s-%s-%s.png" % (
                x.rootdir, x.rootdir, x.subdir, x.rootdir, x.subdir, x.id)

            img = cv.imread(path, cv.IMREAD_GRAYSCALE)

            ret, img = cv.threshold(img, 180, 255, cv.THRESH_BINARY_INV)

            img = cv.ximgproc.thinning(img, img, cv.ximgproc.THINNING_GUOHALL)

            kernel = np.ones((5, 5), np.uint8)

            img = cv.dilate(img, kernel)

            img = cv.resize(img, (0, 0), fx=0.095, fy=0.095)  # reduce image size to only 9.5%

            h, w = img.shape[0:2]
            heights.append(h)
            widths.append(w)

            self.training_data.append([np.array(img), self.one_hot_encode(x.transcript)])

        max_height = 32
        max_width = 256

        for x in tqdm(self.training_data):

            if x[0].shape[0] % 2 == 0 and x[0].shape[1] % 2 == 0:
                x[0] = cv.copyMakeBorder(x[0], int((max_height - x[0].shape[0]) / 2),
                                          int((max_height - x[0].shape[0]) / 2), int((max_width - x[0].shape[1]) / 2),
                                          int((max_width - x[0].shape[1]) / 2), cv.BORDER_CONSTANT, value=0)
            elif x[0].shape[0] % 2 != 0 and x[0].shape[1] % 2 == 0:
                x[0] = cv.copyMakeBorder(x[0], int((max_height - x[0].shape[0]) / 2) + 1,
                                          int((max_height - x[0].shape[0]) / 2), int((max_width - x[0].shape[1]) / 2),
                                          int((max_width - x[0].shape[1]) / 2), cv.BORDER_CONSTANT, value=0)
            if x[0].shape[0] % 2 == 0 and x[0].shape[1] % 2 != 0:
                x[0] = cv.copyMakeBorder(x[0], int((max_height - x[0].shape[0]) / 2),
                                          int((max_height - x[0].shape[0]) / 2),
                                          int((max_width - x[0].shape[1]) / 2) + 1,
                                          int((max_width - x[0].shape[1]) / 2), cv.BORDER_CONSTANT, value=0)
            if x[0].shape[0] % 2 != 0 and x[0].shape[1] % 2 != 0:
                x[0] = cv.copyMakeBorder(x[0], int((max_height - x[0].shape[0]) / 2) + 1,
                                          int((max_height - x[0].shape[0]) / 2),
                                          int((max_width - x[0].shape[1]) / 2) + 1,
                                          int((max_width - x[0].shape[1]) / 2), cv.BORDER_CONSTANT, value=0)

        # tqdm(np.random.shuffle(self.training_data))
        tqdm(np.save("training_data.npy", self.training_data))





class ConvNetToBiLSTM(nn.Module):
    def __init__(self):
        super(ConvNetToBiLSTM, self).__init__()
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
        self.fc1 = nn.Linear(9 * 60 * 64, 1024)

        self.lstm1 = nn.LSTM(input_size=1024, hidden_size=100, num_layers=1, bidirectional=True)
        self.layer_norm = nn.LayerNorm(1024)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)

        out = self.layer_norm(out)
        out = F.gelu(out)
        out, _ = self.BiGRU(out)
        out = self.dropout(out)
        return out


REBUILD_DATA = True

if __name__ == '__main__':

    if REBUILD_DATA:
        data = dataClass()
        data.read_file()
        data.make_training_data()

    training_data = np.load("training_data.npy", allow_pickle=True)
    # print((training_data[0, 1]).transcript)
    # cv.imshow('image', training_data[0, 0])
    # cv.waitKey(0)

    model = ConvNetToBiLSTM()

    loss_function = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



