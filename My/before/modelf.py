"""
model : Mnih et al., 2015, Human-level Control ... Nature

"""
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SIMPLE_GAME, GRAPH_GAME
from torchvision import transforms
from vbn import VirtualBatchNorm2D


trans=transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])

class ESNet(nn.Module):
    '''
    input: (N, 4, 84, 84)


    '''
    def __init__(self, CONFIG):
        super(ESNet, self).__init__()
        self.conv1_f = 32
        self.conv2_f = 64
        self.conv3_f = 64
        # output: 20x20x32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        # output: 9x9x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # output: 7x7x64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(self.conv1_f, affine=False)
        self.bn2 = nn.BatchNorm2d(self.conv2_f, affine=False)
        self.bn3 = nn.BatchNorm2d(self.conv3_f, affine=False)

        self.vbn1 = VirtualBatchNorm2D(self.conv1_f)
        self.vbn2 = VirtualBatchNorm2D(self.conv2_f)
        self.vbn3 = VirtualBatchNorm2D(self.conv3_f)

        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, CONFIG['n_action'])

        self.set_parameter_no_grad()
        self._initialize_weights()
        self.status = "bn"
        # self.previous_frame should be PILImage
        self.previous_frame = None

    def forward_bn(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.bn3(self.conv3(x))
        x = F.relu(x)

        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def forward_vbn(self, x):
        x = self.vbn1(self.conv1(x))
        x = F.relu(x)
        x = self.vbn2(self.conv2(x))
        x = F.relu(x)
        x = self.vbn3(self.conv3(x))
        x = F.relu(x)
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def finish_one_game(self):
        self.previous_frame = None

    def switch_to_train(self):
        self.vbn1.set_mean_var_from_bn(self.bn1)
        self.vbn2.set_mean_var_from_bn(self.bn2)
        self.vbn3.set_mean_var_from_bn(self.bn3)
        self.status = 'vbn'
    
    def forward(self, x):
        # preprocess input data
        x_copy = x.copy()
        if self.previous_frame is not None:
            x = np.maximum(x, self.previous_frame)
        self.previous_frame = x_copy
        x = transforms.ToPILImage()(x).convert('RGB') 
        # output of trans : (1, 84, 84)
        x = trans(x)
        x = x.reshape(1, 1, 84, 84)

        if self.status == 'bn':
            return self.forward_bn(x)
        elif self.status == 'vbn':
            return self.forward_vbn(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_parameter_no_grad(self):
        for param in self.parameters():
            param.requires_grad = False
    

class SimpleNet(nn.Module):
    '''
    input: (N, n_feature)
    output: (N)
    '''
    def __init__(self, CONFIG):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(CONFIG['n_feature'], 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, CONFIG['n_action'])

    def forward(self, input):
        input = torch.Tensor(input)
        input = input.view(1, -1)
        input = torch.tanh(self.fc1(input))
        input = torch.tanh(self.fc2(input))
        input = self.fc3(input)
        # input = input[0]
        return F.softmax(input, dim=1)


def build_model(CONFIG):
    gamename = CONFIG['game']
    if gamename in SIMPLE_GAME:
        print("model type")
        return SimpleNet(CONFIG)
    elif gamename in GRAPH_GAME:
        return ESNet(CONFIG)
    else:
        print("please select a correct game!")
        exit()
