
# [mnih et al. 2015]
# reprocess

import torch
import numpy as np

from torchvision import transforms
from collections import deque

from config import FRAME_SKIP


trans=transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])


class ProcessUnit(object):

    def __init__(self, length):
        self.length = length * FRAME_SKIP
        self.frame_list = deque(maxlen=self.length)

    def step(self, x):
        # insert in left, so the element of index 0 is newest
        self.frame_list.appendleft(x)

    def to_torch_tensor(self):
        length = len(self.frame_list)
        x_list = []
        i = 0
        while i < length:
            if i == length - 1:
                x_list.append(self.transform(self.frame_list[i]))
            else:
                x = np.maximum(self.frame_list[i], self.frame_list[i+1])
                x_list.append(self.transform(x))
            i += 4
        while len(x_list) < 4:
            x_list.append(x_list[-1])
        return torch.cat(x_list, 1)
        #return torch.cat(x_list[::-1], 1)

    def transform(self, x):
        x = transforms.ToPILImage()(x).convert('RGB')
        x = trans(x)
        x = x.reshape(1, 1, 84, 84)
        return x

