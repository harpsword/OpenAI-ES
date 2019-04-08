"""
Deep Q-Learning
"""
import click
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp

N_ACTION = 7

class Net(nn.Module):
    def __init__(self, n_action):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=(0,1))
        self.conv2 = nn.Conv2d(6, 10, kernel_size=3, padding=(1,1))
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3, padding=(0,1))
        # self.conv4 = nn.Conv2d(20, 40, kernel_size=3, padding=(0,1))
        # self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(600*20, 50)
        self.fc2 = nn.Linear(50, n_action)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        # print(x.shape)
        x = x.view(-1, 600*20)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class QLearningModel(object):

    def __init__(self, n_action, ):
        self.n_action = n_action
        self.net = Net(self.n_action)
        

@click.command()
def main():
    pass


if __name__ == '__main__':
    main()
