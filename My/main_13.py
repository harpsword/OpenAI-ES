
import click
import gym
import torch
import time
import pickle
import numpy as np
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from config import N_POPULATION, N_GENERATION, LR, SIGMA
from config import CONFIG
from model_13 import build_model
from optimizer import SGD
from train_13 import train, get_reward

torch.set_num_threads(1)

@click.command()
@click.argument("namemark")
@click.option("--ncpu", default=mp.cpu_count()-1, help='The number of cores', type=int)
@click.option("--batchsize", default=N_POPULATION, help='batch size(population size)', type=int)
@click.option("--generation", default=N_GENERATION, help='the number of generation', type=int)
@click.option("--lr", default=LR, help='learning rate')
@click.option("--sigma", default=SIGMA, help='the SD of perturbed noise')
@click.option("--vbn/--no-vbn",default=False, help='use virtual batch normalization or not')
@click.option("--vbn_test_g", default=10, help='the generation to estimation reference mean and var', type=int)
# @click.option("--simple/--no-simple", default=True, help="use simple model or not")
def main(namemark, ncpu, batchsize, generation, lr, sigma, vbn, vbn_test_g):
    vbn = True
    env = gym.make(CONFIG['game']).unwrapped
    experiment_record = {}
    experiment_record['kid_rewards'] = []
    experiment_record['test_rewards'] = []

    device = torch.device("cpu")
    model = build_model(CONFIG).to(device)
    # print(type(model))
    # model.share_memory()

    # utility instead reward for update parameters (rank transformation)
    base = batchsize   # *2 for mirrored sampling
    if batchsize % 2 == 1:
        print("need an even batch size")
        exit()
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base

    optimizer = SGD(model.named_parameters(), lr)
    pool = mp.Pool(processes=ncpu)
    test_episodes = 15
    # estimate mean and var
    if vbn:
        for g in range(vbn_test_g):
            t0 = time.time()
            model, kid_rewards, _, _ = train(model, optimizer, pool, sigma, env, test_episodes, CONFIG)
            print(
                'Gen: ', g,
                # '| Net_R: %.1f' % mar,
                '| Kid_avg_R: %.1f' % np.array(kid_rewards).mean(),
                '| Gen_T: %.2f' % (time.time() - t0),)
        # reinit model and optimizer
        optimizer.zero_grad()
        model.switch_to_train()
        model._initialize_weights()
    
    # training
    from config import episodes_per_batch
    batchsize = episodes_per_batch
    mar = None      # moving average reward
    for g in range(generation):
        t0 = time.time()
        model, kid_rewards, timestep_count, episodes_number = train(model, optimizer, pool, sigma, env, int(batchsize/2), CONFIG)
        experiment_record['kid_rewards'].append([g, np.array(kid_rewards).mean()])
        # print(
        #         'Gen: ', g,
        #         # '| Net_R: %.1f' % mar,
        #         '| Kid_avg_R: %.1f' % np.array(kid_rewards).mean(),
        #         '| Gen_T: %.2f' % (time.time() - t0),)
        print('Gen:', g,
              'Kid_avg_R: %.1f' % np.array(kid_rewards).mean(),
              'episodes number:', episodes_number,
              'timestep number:', timestep_count)

        if g % 40 == 0:
        # if True:
            test_times = 5
            mar = 0
            for j in range(test_times):
                # test trained net without noises
                net_r, _ = get_reward(model, env, CONFIG['ep_max_step'], sigma, CONFIG, None,)
                # mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward
                mar += net_r
            mar = mar / test_times
            experiment_record['test_rewards'].append([g, mar])
            print(
                'Gen: ', g,
                '| Net_R: %.1f' % mar,
                '| Kid_avg_R: %.1f' % np.array(kid_rewards).mean(),
                '| Gen_T: %.2f' % (time.time() - t0),)
        if mar >= CONFIG['eval_threshold']: break
        
        if (g-1)% 500 == 500 -1:
            CONFIG['ep_max_step'] += 150

        if (g-1) % 1000 == 1000 -1:
            
            torch.save(model.state_dict(), CONFIG['game']+str(namemark)+"genetation"+str(g)+".pt")
            with open("experiment_record"+str(namemark)+"genetation"+str(g)+".pickle", "wb") as f:
                pickle.dump(experiment_record, f)
    
    run_times =20

    for j in range(run_times):
        # test trained net without noise
        net_r = get_reward(model, env, CONFIG['ep_max_step'], sigma, CONFIG, None,)
        # mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward
        mar += net_r
    mar = mar / run_times
    print("runing 100 times")
    print(
        'Gen: ', g,
        '| Net_R: %.1f' % mar,
        '| Kid_avg_R: %.1f' % kid_rewards.mean(),
        '| Gen_T: %.2f' % (time.time() - t0),)

    # ---------------SAVE---------
    torch.save(model.state_dict(), CONFIG['game']+str(namemark)+".pt")
    with open("experiment_record"+str(namemark)+".pickle", "wb") as f:
        pickle.dump(experiment_record, f)


if __name__ == '__main__':
    main()
