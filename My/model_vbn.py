import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SIMPLE_GAME, GRAPH_GAME
from torchvision import transforms
from vbn import VirtualBatchNorm2D

trans=transforms.Compose(
    [
        transforms.ToTensor(),
    ])

class ESNet(nn.Module):
    '''
    input: (N, C, H, W)
    output:(N)
    '''
    def __init__(self, CONFIG):
        super(ESNet, self).__init__()
        self.conv1_f = 6
        self.conv2_f = 10
        self.conv3_f = 20
        self.conv1 = nn.Conv2d(3, self.conv1_f, kernel_size=3, padding=(0,1))
        self.conv2 = nn.Conv2d(self.conv1_f, self.conv2_f, kernel_size=3, padding=(1,1))
        self.conv3 = nn.Conv2d(self.conv2_f, self.conv3_f, kernel_size=3, padding=(0,1))
        # self.conv4 = nn.Conv2d(20, 40, kernel_size=3, padding=(0,1))
        # self.conv3_drop = nn.Dropout2d()
        self.bn1 = nn.BatchNorm2d(self.conv1_f, affine=False)
        self.bn2 = nn.BatchNorm2d(self.conv2_f, affine=False)
        self.bn3 = nn.BatchNorm2d(self.conv3_f, affine=False)

        self.vbn1 = VirtualBatchNorm2D(self.conv1_f)
        self.vbn2 = VirtualBatchNorm2D(self.conv2_f)
        self.vbn3 = VirtualBatchNorm2D(self.conv3_f)

        self.fc1 = nn.Linear(600*20, 50)
        self.fc2 = nn.Linear(50, CONFIG['n_action'])

        self.set_parameter_no_grad()
        self._initialize_weights()
        self.status = "bn"

    def forward_bn(self, x):
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

    def forward_vbn(self, x):
        x = trans(x).reshape(1, 3, 250, 160)
        # print(x.shape)
        x = F.relu(self.vbn1(F.max_pool2d(self.conv1(x), 2)))
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print(x.shape)
        x = F.relu(self.vbn2(F.max_pool2d(self.conv2(x), 2)))
        # x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # print(x.shape)
        x = F.relu(self.vbn3(F.max_pool2d(self.conv3(x), 2)))
        # x = F.relu(F.max_pool2d(self.conv3(x), 2))
        # print(x.shape)
        x = x.view(-1, 600*20)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def switch_to_train(self):
        self.vbn1.set_mean_var_from_bn(self.bn1)
        self.vbn2.set_mean_var_from_bn(self.bn2)
        self.vbn3.set_mean_var_from_bn(self.bn3)
        self.status = 'vbn'
    
    def forward(self, x):
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
