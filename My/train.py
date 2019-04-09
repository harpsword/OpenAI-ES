import numpy as np
import torch
from util import sign


def get_reward(model, env, ep_max_step, sigma, seed_and_id=None):
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        torch.manual_seed(seed)
        with torch.no_grad():
            for name, params in model.named_parameters():
                tmp = model
                for attr_value in name.split('.'):
                    tmp = getattr(tmp, attr_value)
                tmp.add_(torch.randn_like(params)*sigma*sign(k_id))
    observation = env.reset()
    ep_r = 0.
    if ep_max_step is None:
        while True:
        # for step in range(ep_max_step):
            # print(trans(observation).size())
            # print(type(observation))
            action = model(torch.Tensor(observation)).argmax().item()
            # print(action)
            observation, reward , done, _ = env.step(action)
            ep_r += reward
            if done:
                break
    else:
        for step in range(ep_max_step):
            # print(trans(observation).size())
            # print(type(observation))
            action = model(torch.Tensor(observation)).argmax().item()
            # print(action)
            observation, reward , done, _ = env.step(action)
            ep_r += reward
            if done:
                break
    return ep_r

def train(model, optimizer, utility, pool, sigma, env, N_KID, CONFIG):
    # pass seed instead whole noise matrix to parallel will save your time
    # mirrored sampling
    # print(type(N_KID))
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)   

    # distribute training in parallel
    jobs = [pool.apply_async(get_reward, ( model, env, CONFIG['ep_max_step'],sigma,
                                          [noise_seed[k_id], k_id], )) for k_id in range(N_KID*2)]
    rewards = np.array([j.get() for j in jobs])
    
    # rewards = [get_reward(model, env, CONFIG['ep_max_step'], None,)]
    # print(rewards)
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
        cumulative_update[name].mul_(1/(2*N_KID*sigma))
    optimizer.update_model_parameters(model, cumulative_update)
    return model, rewards