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

def get_reward(modeltype, base_model, env, ep_max_step, sigma, CONFIG, reference, seed_and_id=None, test=False):
    # reference : reference batch torch
    # start = time.time()
    if seed_and_id is not None:
        index_seed, k_id = seed_and_id
        if modeltype == '2015':
            from model_15 import build_model
        elif modeltype == '2013':
            from model_13 import build_model
        model = build_model(CONFIG)
        model.load_state_dict(base_model.state_dict())
        model.switch_to_vbn()
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
    # first send in reference
    # second switch to vbn
    model.switch_to_bn()
    output = model(reference)
    model.switch_to_vbn()

    env.frameskip = 1
    observation = env.reset()
    break_is_true = False
    ep_r = 0.
    frame_count = 0
    # print('k_id mid:', k_id,time.time()-start)
    if ep_max_step is None:
        raise TypeError("test")
    else:
        ProcessU = ProcessUnit(FRAME_SKIP)
        ProcessU.step(observation)
        
        if test == True:
            ep_max_step = 108000
        #no_op_frames = np.random.randint(FRAME_SKIP+1, 30)
        no_op_frames = np.random.randint(1, 31)
        for i in range(no_op_frames):
            # TODO: I think 0 is Null Action
            # but have not found any article about the meaning of every actions
            observation, reward, done, _ = env.step(0)
            ProcessU.step(observation)
            frame_count += 1

        for step in range(ep_max_step):
            action = model(ProcessU.to_torch_tensor())[0].argmax().item()
            for i in range(FRAME_SKIP):
                observation, reward , done, _ = env.step(action)
                ProcessU.step(observation)
                frame_count += 1
                ep_r += reward
                if done:
                    break_is_true = True
            if break_is_true:
                break
    return ep_r, frame_count

def train(model, optimizer, pool, sigma, env, N_KID, CONFIG, modeltype, reference_batch_torch):
    # pass seed instead whole noise matrix to parallel will save your time
    # reference_batch_torch: torch.tensor (128, 4, 84, 84)
    # mirrored sampling
    model_size = model.get_size()
    slice_dict = model.get_name_slice_dict()
    stream = np.random.RandomState()
    index_seed = shared_noise_table.sample_index(stream, model_size, N_KID).repeat(2)

    # distribute training in parallel
    jobs = [pool.apply_async(get_reward, (modeltype ,model, env, CONFIG['ep_max_step'],sigma,CONFIG,
                                          reference_batch_torch,[index_seed[k_id], k_id],)) for k_id in range(N_KID*2)]
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
        noise_array = shared_noise_table.get(index_seed[k_id], model_size)
        for name, params in model.named_parameters():
            if not name in cumulative_update.keys():
                cumulative_update[name] = torch.zeros_like(params, dtype=torch.float)
            noise = torch.tensor(noise_array[slice_dict[name][0]:slice_dict[name][1]], dtype=torch.float).reshape(params.shape)
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
    return model, rewards, timesteps_count


def test(model, pool, env, test_times, CONFIG, reference):
    # distribute training in parallel
    jobs = [pool.apply_async(get_reward, (None,model, env, CONFIG['ep_max_step'], None, CONFIG,reference, None, True)) for i in range(test_times)]
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
    ep_max_step = 108000
    no_op_frames = np.random.randint(1, 31)
    n_action = env.action_space.n

    for i in range(no_op_frames):
        observation, reward, done, _ = env.step(np.random.randint(n_action))
        ProcessU.step(observation)
        if np.random.rand() <= prob:
            r.append(ProcessU.to_torch_tensor())

    for step in range(ep_max_step):
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


    
        
    






    
    

