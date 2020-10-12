import fnmatch
import os
import torchvision
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


def read_file():
    with open("data/lines.txt") as f:
        raw = f.readlines()

    lines = []

    for x in raw:
        lines.append(x.split())

    labels = []

    for x in lines:
        transcript = ''.join(map(str, x[8:]))  # screw the space
        labels.append(transcript)

    return labels


def one_hot_encode(line):
    # define universe of possible input values
    alphabet = """|ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,/?:;'"-!#&*()+"""
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in line]
    # one hot encode
    one_hot = []
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        one_hot.append(letter)

    return one_hot


def check_accuracy(test_loader, trained_model):
    num_correct = 0
    num_samples = 0
    trained_model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device=device)
            target = target.to(device=device)

            scores = trained_model(data)

            # todo: implement testing
            acc = 0

    trained_model.train()
    return acc


class IamDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.num_img = len(fnmatch.filter(os.listdir(root_dir), '*.png'))
        self.labels = read_file()

    def __len__(self):
        return self.num_img

    def __getitem__(self, index):
        path = "data/line_img/line (%s).png" % (index + 1)

        img = cv.imread(path, cv.IMREAD_GRAYSCALE)

        ret, img = cv.threshold(img, 185, 255, cv.THRESH_BINARY_INV)

        img = cv.ximgproc.thinning(img, img, cv.ximgproc.THINNING_GUOHALL)

        kernel = np.ones((5, 5), np.uint8)

        img = cv.dilate(img, kernel)

        img = cv.resize(img, (0, 0), fx=0.095, fy=0.095)  # reduce image size to only 9.5%

        max_height = 32
        max_width = 256

        if img.shape[0] % 2 == 0 and img.shape[1] % 2 == 0:
            img = cv.copyMakeBorder(img, int((max_height - img.shape[0]) / 2),
                                    int((max_height - img.shape[0]) / 2), int((max_width - img.shape[1]) / 2),
                                    int((max_width - img.shape[1]) / 2), cv.BORDER_CONSTANT, value=0)
        elif img.shape[0] % 2 != 0 and img.shape[1] % 2 == 0:
            img = cv.copyMakeBorder(img, int((max_height - img.shape[0]) / 2) + 1,
                                    int((max_height - img.shape[0]) / 2), int((max_width - img.shape[1]) / 2),
                                    int((max_width - img.shape[1]) / 2), cv.BORDER_CONSTANT, value=0)
        elif img.shape[0] % 2 == 0 and img.shape[1] % 2 != 0:
            img = cv.copyMakeBorder(img, int((max_height - img.shape[0]) / 2),
                                    int((max_height - img.shape[0]) / 2),
                                    int((max_width - img.shape[1]) / 2) + 1,
                                    int((max_width - img.shape[1]) / 2), cv.BORDER_CONSTANT, value=0)
        elif img.shape[0] % 2 != 0 and img.shape[1] % 2 != 0:
            img = cv.copyMakeBorder(img, int((max_height - img.shape[0]) / 2) + 1,
                                    int((max_height - img.shape[0]) / 2),
                                    int((max_width - img.shape[1]) / 2) + 1,
                                    int((max_width - img.shape[1]) / 2), cv.BORDER_CONSTANT, value=0)

        sample = torchvision.transforms.ToTensor()(img)

        label = self.labels[index]

        one_hot_label = np.asarray(one_hot_encode(label))
        one_hot_label = np.pad(one_hot_label, [(0, 70 - one_hot_label.shape[0]), (0, 0)], mode='constant')

        return sample, one_hot_label


class ConvNetToBiLSTM(nn.Module):
    def __init__(self):
        super(ConvNetToBiLSTM, self).__init__()
        # 32 by 256
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # 16 by 128
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        # 8 by 64
        self.fc1 = nn.Linear(8 * 64 * 64, 1024)

        self.layer_norm = nn.LayerNorm(1024)
        self.lstm1 = nn.LSTM(input_size=1024, hidden_size=100, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        print("1", x.shape)
        out = self.layer1(x)
        print("2", out.shape)
        out = self.layer2(out)
        print("3", out.shape)
        out = out.reshape(out.size(0), -1)
        print("4", out.shape)
        out = self.drop_out(out)
        print("5", out.shape)
        out = self.fc1(out)
        print("6", out.shape)

        out = self.layer_norm(out)
        print("7", out.shape)
        out = F.gelu(out)
        print("8", out.shape)
        out = out.unsqueeze(dim=0)
        print("9", out.shape)
        out = self.lstm1(out)
        print("10", out.shape)
        out = self.dropout(out)
        print("11", out.shape)
        return out


if __name__ == '__main__':

    # Set Device
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    # Hyperparameters
    LEARNING_RATE = 0.01
    BATCH_SIZE = 100
    EPOCHS = 2

    # Load Data
    dataset = IamDataset("data/line_img")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [12020, 1333])
    train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=True, num_workers=0)

    # Initialize Network
    model = ConvNetToBiLSTM()

    # Loss and Optimizer
    loss_function = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train Network
    for epoch in range(EPOCHS):
        for batch_index, (data, target) in enumerate(train_loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            target = target.to(device=device)

            # Forward
            scores = model(data)
            loss = loss_function(scores, target)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient Descent or Adam step
            optimizer.step()

    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)


