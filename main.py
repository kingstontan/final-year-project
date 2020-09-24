import os
import cv2
import numpy as np
from tqdm import tqdm

REBUILD_DATA = True


class dataClass():

    path = "data/lines.txt"

    def read_file(self):

        line = []

        with open(self.path, 'r') as f:
            for l in f:
                for word in l.split():
                    line.append(word)
                break

        print(line)





if __name__ == '__main__':

    if REBUILD_DATA:

        data = dataClass()
        data.read_file()

