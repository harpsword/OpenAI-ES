
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SIMPLE_GAME, GRAPH_GAME
from torchvision import transforms

trans=transforms.Compose(
    [
        transforms.ToTensor(),

    ])

class Net(nn.Module):
    '''
    input: (N, C, H, W)
    output:(N)
    '''
    def __init__(self, CONFIG):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=(0,1))
        self.conv2 = nn.Conv2d(6, 10, kernel_size=3, padding=(1,1))
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3, padding=(0,1))
        # self.conv4 = nn.Conv2d(20, 40, kernel_size=3, padding=(0,1))
        # self.conv3_drop = nn.Dropout2d()
        self.bn1 = nn.BatchNorm2d(6, affine=False)
        self.bn2 = nn.BatchNorm2d(10, affine=False)
        self.bn3 = nn.BatchNorm2d(20, affine=False)
        self.fc1 = nn.Linear(600*20, 50)
        self.fc2 = nn.Linear(50, CONFIG['n_action'])

    def forward(self, x):
        x = trans(x).reshape(1, 3, 250, 160)
        # print(x.shape)
        x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print(x.shape)
        x = F.relu(self.bn2(F.max_pool2d(self.conv2(x), 2)))
        # x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # print(x.shape)
        x = F.relu(self.bn3(F.max_pool2d(self.conv3(x), 2)))
        # x = F.relu(F.max_pool2d(self.conv3(x), 2))
        # print(x.shape)
        x = x.view(-1, 600*20)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

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
        return Net(CONFIG)
    else:
        print("please select a correct game!")
        exit()
