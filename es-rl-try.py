
'''
example of distributed ES proposed by OpenAI.
Details can be found in : https://arxiv.org/abs/1703.03864
'''
import gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from torchvision import transforms

trans=transforms.Compose(
    [
        transforms.ToTensor(),

    ])

CONFIG = [
    dict(game='Assault-v0', observation=(250, 160), n_action=7, ep_max_step=800,eval_threshold=500),
    dict(game="CartPole-v0",
         n_feature=4, n_action=2, continuous_a=[False], ep_max_step=700, eval_threshold=500),
    dict(game="MountainCar-v0",
         n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
    dict(game="Pendulum-v0",
         n_feature=3, n_action=1, continuous_a=[True, 2.], ep_max_step=200, eval_threshold=-180)
][0]    # choose your game

N_KID = 20         # half of the training population
N_POPULATION = 2*N_KID
N_GENERATION = 1000         # training step
LR = 0.05
SIGMA = 0.05
N_ACTION = 7
# N_CORE = mp.cpu_count() * 2 - 8
# N_CORE = mp.cpu_count() - 1
N_CORE = 2

class SGD(object): 

    def __init__(self, named_parameters, learning_rate, momentum=0.9):
        self.v = dict()
        self.lr, self.momentum = learning_rate, momentum
        for name, params in named_parameters:
            self.v[name] = torch.zeros_like(params, dtype=torch.float)
            # print(name)
            # print(params.data.size())
        
    def update_model_parameters(self, model, gradients):
        if not isinstance(gradients, dict):
            raise TypeError("the gradients must be a dict with key(name)"
                            "value(torch.tensor), Got{}".format(torch.typename(gradients)))
        for name, params in self.v.items():
            self.v[name].mul_(self.momentum)
            self.v[name].add_(gradients[name]*(1-self.momentum))
        # update parameter of model recursively
        with torch.no_grad():
            for name, param in model.named_parameters():
                '''
                example : name(conv1.weight)
                model.conv1 is Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1))
                model.conv1.weight is the param
                '''
                tmp = model
                for attr_value in name.split('.'):
                    tmp = getattr(tmp, attr_value)
                tmp.add_(self.lr*self.v[name])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=(0,1))
        self.conv2 = nn.Conv2d(6, 10, kernel_size=3, padding=(1,1))
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3, padding=(0,1))
        # self.conv4 = nn.Conv2d(20, 40, kernel_size=3, padding=(0,1))
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(600*20, 50)
        self.fc2 = nn.Linear(50, N_ACTION)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        # print(x.shape)
        x = x.view(-1, 600*20)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def sign(k_id): return 1. if k_id % 2 == 0 else -1.  # mirrored sampling

def get_reward(model, env, ep_max_step, seed_and_id=None):
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        torch.manual_seed(seed)
        with torch.no_grad():
            for name, params in model.named_parameters():
                tmp = model
                for attr_value in name.split('.'):
                    tmp = getattr(tmp, attr_value)
                tmp.add_(torch.randn_like(params)*SIGMA*sign(k_id))
    observation = env.reset()
    ep_r = 0.
    for step in range(ep_max_step):
        # print(trans(observation).size())
        # print(type(observation))
        action = model(trans(observation).reshape(1, 3, 250, 160)).argmax().item()
        # print(action)
        observation, reward , done, _ = env.step(action)
        ep_r += reward
        if done:
            break
    return ep_r

def train(model, optimizer, utility, pool):
    # pass seed instead whole noise matrix to parallel will save your time
    # mirrored sampling
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)   

    # distribute training in parallel
    jobs = [pool.apply_async(get_reward, ( model, env, CONFIG['ep_max_step'],
                                          [noise_seed[k_id], k_id], )) for k_id in range(N_KID*2)]
    rewards = np.array([j.get() for j in jobs])
    print(rewards)
    # rewards = [get_reward(model, env, CONFIG['ep_max_step'], None,)]
    kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward

    cumulative_update = {}       # initialize update values
    for ui, k_id in enumerate(kids_rank):
        # reconstruct noise using seed
        torch.manual_seed(noise_seed[k_id])
        for name, params in model.named_parameters():
            if not name in cumulative_update.keys():
                cumulative_update[name] = torch.zeros_like(params, dtype=torch.float)
            cumulative_update[name].add_(utility[ui] * sign(k_id) * torch.randn_like(params))
    for name, params in cumulative_update.items():
        cumulative_update[name].mul_(1/(2*N_KID*SIGMA))
    optimizer.update_model_parameters(model, cumulative_update)
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
        t0 = time.time()
        model, kid_rewards = train(model, optimizer, utility, pool)

        # if g % 20 == 0:
        if True:
            mar = 0
            for j in range(5):
                # test trained net without noises
                net_r = get_reward(model, env, CONFIG['ep_max_step'], None,)
                # mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward
                mar += net_r
            mar = mar / 5
            print(
                'Gen: ', g,
                '| Net_R: %.1f' % mar,
                '| Kid_avg_R: %.1f' % np.array(kid_rewards).mean(),
                '| Gen_T: %.2f' % (time.time() - t0),)
        if mar >= CONFIG['eval_threshold']: break
    
    run_times =20

    for j in range(run_times):
        # test trained net without noise
        net_r = get_reward(model, env, CONFIG['ep_max_step'], None,)
        # mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward
        mar += net_r
    mar = mar / run_times
    print("runing 100 times")
    print(
        'Gen: ', g,
        '| Net_R: %.1f' % mar,
        '| Kid_avg_R: %.1f' % kid_rewards.mean(),
        '| Gen_T: %.2f' % (time.time() - t0),)