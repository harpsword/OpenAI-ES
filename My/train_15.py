'''
with frame skip
but no limitation of timestep one batch
weight decay
'''
import numpy as np
import torch
from util import sign
from model_15 import build_model
from preprocess import ProcessUnit
import time
from config import FRAME_SKIP

def get_reward(base_model, env, ep_max_step, sigma, CONFIG, seed_and_id=None, test=False):
    # start = time.time()
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        np.random.seed(seed)
        model = build_model(CONFIG)
        model.load_state_dict(base_model.state_dict())
        model.switch_to_train()
        # model = base_model
        with torch.no_grad():
            for name, params in model.named_parameters():
                tmp = model
                for attr_value in name.split('.'):
                    tmp = getattr(tmp, attr_value)
                noise = torch.tensor(np.random.normal(0,1,tmp.size()), dtype=torch.float)
                tmp.add_(noise*sigma*sign(k_id))
    else:
        model = base_model
    env.frameskip = 1
    observation = env.reset()
    break_is_true = False
    ep_r = 0.
    # print('k_id mid:', k_id,time.time()-start)
    if ep_max_step is None:
        raise TypeError("test")
    else:
        ProcessU = ProcessUnit(FRAME_SKIP)
        ProcessU.step(observation)
        
        if test == True:
            ep_max_step = 18000
        no_op_frames = np.random.randint(FRAME_SKIP+1, 30)
        for i in range(no_op_frames):
            # TODO: I think 0 is Null Action
            # but have not found any article about the meaning of every actions
            observation, reward, done, _ = env.step(0)
            ProcessU.step(observation)

        for step in range(ep_max_step):
            action = model(ProcessU.to_torch_tensor())[0].argmax().item()
            for i in range(FRAME_SKIP):
                observation, reward , done, _ = env.step(action)
                ProcessU.step(observation)
                ep_r += reward
                if done:
                    break_is_true = True
            if break_is_true:
                break
    return ep_r, step

def train(model, optimizer, pool, sigma, env, N_KID, CONFIG):
    # pass seed instead whole noise matrix to parallel will save your time
    # mirrored sampling
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)   

    # distribute training in parallel
    jobs = [pool.apply_async(get_reward, ( model, env, CONFIG['ep_max_step'],sigma,CONFIG,
                                          [noise_seed[k_id], k_id], )) for k_id in range(N_KID*2)]
    from config import timesteps_per_batch
    # N_KID means episodes_per_batch
    rewards = []
    timesteps = []
    timesteps_count = 0
    for idx, j in enumerate(jobs):
        rewards.append(j.get()[0])
        timesteps.append(j.get()[1])
        timesteps_count += j.get()[1]
    
    base = len(rewards)
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base

    kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward

    cumulative_update = {}       # initialize update values
    for ui, k_id in enumerate(kids_rank):
        # reconstruct noise using seed
        # torch.manual_seed(noise_seed[k_id])
        np.random.seed(noise_seed[k_id])
        for name, params in model.named_parameters():
            if not name in cumulative_update.keys():
                cumulative_update[name] = torch.zeros_like(params, dtype=torch.float)
            noise = torch.tensor(np.random.normal(0,1,params.size()), dtype=torch.float)
            # cumulative_update[name].add_(utility[ui] * sign(k_id) * torch.randn_like(params))
            cumulative_update[name].add_(utility[ui]*sign(k_id)*noise)
    for name, params in cumulative_update.items():
        cumulative_update[name].mul_(1/(2*N_KID*sigma))
    # weight decay
    for name, params in model.named_parameters():
        tmp = model
        for attr_value in name.split('.'):
            tmp = getattr(tmp, attr_value)
        cumulative_update[name].add_(-CONFIG['l2coeff']*tmp)
    optimizer.update_model_parameters(model, cumulative_update)
    return model, rewards, timesteps_count, len(rewards)


def test(model, pool, env, test_times, CONFIG):
    # distribute training in parallel
    jobs = [pool.apply_async(get_reward, (model, env, CONFIG['ep_max_step'], None, CONFIG, None, True)) for i in range(test_times)]
    from config import timesteps_per_batch
    # N_KID means episodes_per_batch
    rewards = []
    timesteps = []
    timesteps_count = 0
    for idx, j in enumerate(jobs):
        rewards.append(j.get()[0])
        timesteps.append(j.get()[1])
        timesteps_count += j.get()[1]
    
    return rewards, timesteps_count, len(rewards)
