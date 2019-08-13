'''
with frame skip
but no limitation of timestep one batch
weight decay
'''
import numpy as np
import torch
from util import sign
from preprocess import ProcessUnit
import time
from config import FRAME_SKIP, reference_batch_size
from noisetable import shared_noise_table 
from model import build_model


def get_reward(base_model, env, ARGS, seed_and_id=None, test=False):
    # start = time.time()
    if seed_and_id is not None:
        index_seed, k_id = seed_and_id
        model = build_model(ARGS)
        model.load_state_dict(base_model.state_dict())
        model_size = model.get_size()
        slice_dict = model.get_name_slice_dict()
        noise_array = shared_noise_table.get(index_seed, model_size)
        with torch.no_grad():
            for name, params in model.named_parameters():
                tmp = model
                for attr_value in name.split('.'):
                    tmp = getattr(tmp, attr_value)
                noise = torch.tensor(noise_array[slice_dict[name][0]:slice_dict[name][1]], dtype=torch.float).reshape(tmp.shape)
                tmp.add_(noise*sigma*sign(k_id))
    else:
        model = base_model

    env.frameskip = 1
    observation = env.reset()
    ep_r = 0.
    frame_count = 0
    if test == True:
        ARGS.timestep_limit_episode = 10000
    for step in range(ARGS.timestep_limit_episode):
        action = model.get_action(observation)
        observation, reward , done, _ = env.step(action)
        frame_count += 1
        ep_r += reward
        if done:
            break
    return ep_r, frame_count


def train(model, optimizer, pool, env, ARGS) 
    # pass seed instead whole noise matrix to parallel will save your time
    # mirrored sampling
    model_size = model.get_size()
    slice_dict = model.get_name_slice_dict()
    stream = np.random.RandomState()
    index_seed = shared_noise_table.sample_index(stream, model_size, N_KID).repeat(2)

    # distribute training in parallel
    jobs = [pool.apply_async(get_reward, (model, env, ARGS, [index_seed[k_id], k_id],)) for k_id in range(ARGS.population_size)]
    rewards = []
    timesteps_count = 0
    for idx, j in enumerate(jobs):
        rewards.append(j.get()[0])
        timesteps_count += j.get()[1]
    
    base = len(rewards)
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base
    kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward

    cumulative_update = {}       # initialize update values
    for ui, k_id in enumerate(kids_rank):
        # reconstruct noise using seed
        noise_array = shared_noise_table.get(index_seed[k_id], model_size)
        for name, params in model.named_parameters():
            if not name in cumulative_update.keys():
                cumulative_update[name] = torch.zeros_like(params, dtype=torch.float)
            noise = torch.tensor(noise_array[slice_dict[name][0]:slice_dict[name][1]], dtype=torch.float).reshape(params.shape)
            cumulative_update[name].add_(utility[ui]*sign(k_id)*noise)
    for name, params in cumulative_update.items():
        cumulative_update[name].mul_(1/(ARGS.population_size*ARGS.sigma))
    # weight decay
    for name, params in model.named_parameters():
        tmp = model
        for attr_value in name.split('.'):
            tmp = getattr(tmp, attr_value)
        cumulative_update[name].add_(-ARGS.l2coeff*tmp)
    optimizer.update_model_parameters(model, cumulative_update)
    return model, rewards, timesteps_count


# TODO
def test(model, pool, env, test_times, CONFIG, reference):
    # distribute training in parallel
    jobs = [pool.apply_async(get_reward, (None,model, env, CONFIG['ARGS.timestep_limit_episode'], None, CONFIG,reference, None, True)) for i in range(test_times)]
    from config import timesteps_per_batch
    # N_KID means episodes_per_batch
    rewards = []
    timesteps = []
    timesteps_count = 0
    for idx, j in enumerate(jobs):
        rewards.append(j.get()[0])
        timesteps.append(j.get()[1])
        timesteps_count += j.get()[1]
    
    return rewards, timesteps_count

def one_explore_for_vbn(env, prob):
    # prob : select probability
    r = []
    env.frameskip = 1
    observation = env.reset()
    break_is_true = False
    ProcessU = ProcessUnit(FRAME_SKIP)
    ProcessU.step(observation)
    ARGS.timestep_limit_episode = 108000
    no_op_frames = np.random.randint(1, 31)
    n_action = env.action_space.n

    for i in range(no_op_frames):
        observation, reward, done, _ = env.step(np.random.randint(n_action))
        ProcessU.step(observation)
        if np.random.rand() <= prob:
            r.append(ProcessU.to_torch_tensor())

    for step in range(ARGS.timestep_limit_episode):
        action = np.random.randint(n_action)
        for i in range(FRAME_SKIP):
            observation, reward , done, _ = env.step(action)
            ProcessU.step(observation)
            if np.random.rand() <= prob:
                r.append(ProcessU.to_torch_tensor())
            if done:
                break_is_true = True
        if break_is_true or len(r) > reference_batch_size:
            break
    return r

def explore_for_vbn(env, prob):
    max_time = 1000
    return_r = []
    for i in range(max_time):
        one_time_r = one_explore_for_vbn(env, prob)
        return_r.extend(one_time_r)
        if len(return_r) > reference_batch_size:
            break
    return return_r[:reference_batch_size]


    
        
    






    
    

