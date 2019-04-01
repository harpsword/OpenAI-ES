
'''
example of distributed ES proposed by OpenAI.
Details can be found in : https://arxiv.org/abs/1703.03864
'''
import gym
import numpy as np
import time
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

CONFIG = [
    dict(game="CartPole-v0",
         n_feature=4, n_action=2, continuous_a=[False], ep_max_step=700, eval_threshold=500),
    dict(game="MountainCar-v0",
         n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
    dict(game="Pendulum-v0",
         n_feature=3, n_action=1, continuous_a=[True, 2.], ep_max_step=200, eval_threshold=-180)
][2]    # choose your game

N_KID = 20         # half of the training population
N_POPULATION = 2*N_KID
N_GENERATION = 5000         # training step
LR = 0.05
SIGMA = 0.05
N_ACTION = 10
N_CORE = mp.cpu_count() * 2 - 8

# def fix_no_leaf(model, pretrained_leaf_dict, prefix=''):
#     for name, param in model._parameters.items():
#         param_name = prefix + ('.' if prefix else '') + name
#         if not param_name in pretrained_leaf_dict.keys():
#             continue
#         if param is not None and pretrained_leaf_dict[param_name] and not param.is_leaf:
#             model._parameters[name] = Variable(param.data, requires_grad = True)
#     for mname, module in model._modules.items():
#         if module is not None:
#             submodule_prefix = prefix + ('.' if prefix else '') + mname
#             fix_no_leaf(module, pretrained_leaf_dict, prefix=submodule_prefix)

class SGD(object): 

    def __init__(self, named_parameters, learning_rate, momentum=0.9):
        self.v = dict()
        self.lr, self.momentum = learning_rate, momentum
        for name, params in named_parameters:
            self.v[name] = torch.zeros_like(params, dtype=torch.float)
            print(name)
            print(params.data.size())
        
    def update_model_parameters(self, model, gradients):
        if not isinstance(gradients, Dict):
            raise TypeError("the gradients must be a dict with key(name)"
                            "value(torch.tensor), Got{}".format(torch.typename(gradients)))
        for name, params in self.v.items():
            self.v[name].mul_(self.momentum).add_(gradients[name]*(1-self.momentum))
        # update parameter of model recursively
        

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, N_ACTION)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def get_reward():
    pass

def train(model, optimizer, utility, pool):
    # pass seed instead whole noise matrix to parallel will save your time
    # mirrored sampling
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)   

    # distribute training in parallel
    jobs = [pool.apply_async(get_reward, (net_shapes, net_params, env, CONFIG['ep_max_step'], CONFIG['continuous_a'],
                                          [noise_seed[k_id], k_id], )) for k_id in range(N_KID*2)]
    rewards = np.array([j.get() for j in jobs])
    kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward

    cumulative_update = {}       # initialize update values
    for ui, k_id in enumerate(kids_rank):
        # reconstruct noise using seed
        np.random.seed(noise_seed[k_id])
        for name, params in model.named_parameters():
            if not name in cumulative_update.keys():
                cumulative_update[name] = torch.zeros_like(params, dtype=torch.float)
            cumulative_updatep[name] += utility[ui] * sign(k_id) * np.random.randn()

    optimizer.update_model_parameters(model, cumulative_update/(2*N_KID*SIGMA))
    return model, rewards

if __name__ == '__main__':
    device = torch.device("cpu")
    model = Net().to(device)
    # model.share_memory()

    # utility instead reward for update parameters (rank transformation)
    base = N_KID * 2    # *2 for mirrored sampling
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base

    # training 
    env = gym.make(CONFIG['game']).unwrapped
    optimizer = SGD(model.named_parameters(), LR)
    pool = mp.Pool(processes=N_CORE)
    mar = None      # moving average reward
    for g in range(N_GENERATION):
        pass