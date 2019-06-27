'''
with vbn
'''
import numpy as np
import torch
from util import sign
from modelf import build_model
import time
from config import FRAME_SKIP

def get_reward(base_model, env, ep_max_step, sigma, CONFIG, seed_and_id=None):
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
                # print(torch.typename(noise))
                # print(torch.typename(tmp))
                tmp.add_(noise*sigma*sign(k_id))
    else:
        model = base_model
    env.frameskip = 1
    observation = env.reset()
    ep_r = 0.
    # print('k_id mid:', k_id,time.time()-start)
    if ep_max_step is None:
        raise TypeError("test")
    else:
        for step in range(ep_max_step):
            # print(trans(observation).size())
            # print(type(observation))
            action = model(observation)[0].argmax().item()
            # print(action)
            observation, reward , done, _ = env.step(action)
            ep_r += reward
            if done:
                break
    # print('k_id final:', k_id,time.time()-start)
    # print('k_id step:', k_id,step)
    return ep_r

def train(model, optimizer, utility, pool, sigma, env, N_KID, CONFIG):
    # pass seed instead whole noise matrix to parallel will save your time
    # mirrored sampling
    # print(type(N_KID))
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)   

    # distribute training in parallel
    jobs = [pool.apply_async(get_reward, ( model, env, CONFIG['ep_max_step'],sigma,CONFIG,
                                          [noise_seed[k_id], k_id], )) for k_id in range(N_KID*2)]
    rewards = np.array([j.get() for j in jobs])
    
    # rewards = [get_reward(model, env, CONFIG['ep_max_step'], sigma, CONFIG, [444, 0],)]

    time.sleep(10)
    # print(rewards)
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
    optimizer.update_model_parameters(model, cumulative_update)
    return model, rewards
