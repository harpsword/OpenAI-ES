
import numpy as np
import torch
import torch.nn as nn

nn.Parameter
torch.Tensor
torch.autograd.Variable()

N_KID = 10
N_GENERATION = 5000

CONFIG = [
    dict(game="CartPole-v0",
         n_feature=4, n_action=2, continuous_a=[False], ep_max_step=700, eval_threshold=500),
    dict(game="MountainCar-v0",
         n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
    dict(game="Pendulum-v0",
         n_feature=3, n_action=1, continuous_a=[True, 2.], ep_max_step=200, eval_threshold=-180)
][1]    # choose your game

def get_reward():
    pass


def build_net():
    '''
    return parameter's shape, params
    '''
    def linear(n_in, n_out):
        w = np.random.randn(n_in*n_out).astype(np.float32) * .1
        b = np.random.randn(n_out).astype(np.float32) * .1
        return (n_i, n_out), np.concatenate((w,b))
    s0, p0 = linear(CONFIG['n_feature'], 30)
    s1, p1 = linear(30, 20)
    s2, p2 = linear(20, CONFIG['n_action'])
    return [s0, s1, s2], np.concatenate((p0, p1, p2))


def train(net_shapes, net_params, pool):
    # train model
    noise_seed = np.random.randint(0, 2**32-1, size=N_KID, dtype=np.uint32)
    jobs = [pool.apply_async(get_reward, (net_shapes, net_params, env, CONFIG['ep_max_step'], CONFIG['continuous_a'],
                            [noise_seed[k_id], k_id], )) for k_id in range(N_KID*2)]



if __name__ == "__main__":
    build_net()
    for g in range(N_GENERATION):
        train()

