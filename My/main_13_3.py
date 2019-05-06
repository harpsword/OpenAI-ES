'''
Frame skip = 4
no limitation of timestep one batch
'''
import os
import click
import gym
import torch
import time
import pickle
import logging
import numpy as np
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from config import N_POPULATION, N_GENERATION, LR, SIGMA, TIMESTEP_LIMIT
from model_13 import build_model
from optimizer import SGD
from train_13_3 import train, get_reward, test

torch.set_num_threads(1)
LogFolder = os.path.join(os.getcwd(), 'log')


def setup_logging(logfile):
    if logfile == 'default.log':
        timenow = time.localtime(time.time())
        logfile = str(timenow.tm_year)+'-'+str(timenow.tm_mon)+'-'+str(timenow.tm_mday)
        indx = 1
        while logfile+'-'+str(indx)+'.log' in os.listdir(LogFolder):
            indx += 1
        logpath = os.path.join(LogFolder, logfile+'-'+str(indx)+'.log')
    else:
        logpath = os.path.join(LogFolder, logfile)
    logging.basicConfig(filename=logpath,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')


@click.command()
@click.argument("namemark")
@click.option("--ncpu", default=mp.cpu_count()-1, help='The number of cores', type=int)
@click.option("--batchsize", default=N_POPULATION, help='batch size(population size)', type=int)
@click.option("--generation", default=N_GENERATION, help='the number of generation', type=int)
@click.option("--lr", default=LR, help='learning rate')
@click.option("--sigma", default=SIGMA, help='the SD of perturbed noise')
@click.option("--vbn/--no-vbn",default=True, help='use virtual batch normalization or not')
@click.option("--vbn_test_g", default=10, help='the generation to estimation reference mean and var', type=int)
@click.option("--gamename", default="Assualt-v0", help="the name of tested game")
@click.option("--logfile", default="default.log", help='the file of log')
# @click.option("--simple/--no-simple", default=True, help="use simple model or not")
def main(namemark, ncpu, batchsize, generation, lr, sigma, vbn, vbn_test_g, gamename, logfile):
    setup_logging(logfile)

    logging.info("learning rate: %s", lr)
    logging.info("sigma: %s", sigma)
    logging.info("Game name: %s", gamename)
    print("learning rate:",lr)
    print("sigma:", sigma)
    print("gamename:", gamename)

    checkpoint_name = gamename + namemark + "-sigma" + str(sigma) +'-lr' + str(lr)

    import json
    configfile = "config.json"
    with open(configfile, "r") as f:
        CONFIG = json.loads(f.read())
    CONFIG = CONFIG[gamename]
    logging.info("Settings: %s", str(CONFIG))

    env = gym.make(gamename)
    CONFIG['n_action'] = env.action_space.n
    CONFIG['game'] = gamename
    experiment_record = {}
    experiment_record['kid_rewards'] = []
    experiment_record['test_rewards'] = []

    device = torch.device("cpu")
    model = build_model(CONFIG).to(device)

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
        logging.info("start test reference batch statistic")
        for g in range(vbn_test_g):
            t0 = time.time()
            model, kid_rewards, _, _ = train(model, optimizer, pool, sigma, env, test_episodes, CONFIG)
            logging.info('Gen: %s | Kid_avg_R: %.1f | Gen_T: %.2f' % (g, np.array(kid_rewards).mean(), time.time()-t0))
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
    mar = None      # moving average reward
    training_timestep_count = 0
    for g in range(generation):
        t0 = time.time()
        model, kid_rewards, timestep_count, episodes_number = train(model, optimizer, pool, sigma, env, int(batchsize/2), CONFIG)
        training_timestep_count += timestep_count
        if training_timestep_count > TIMESTEP_LIMIT:
            logging.info("satisfied timestep limit")
            logging.info("Now timestep %s" % training_timestep_count)
            break
        experiment_record['kid_rewards'].append([g, np.array(kid_rewards).mean()])
        if g % 5 == 0:
            logging.info('Gen: %s | Kid_avg_R: %.1f | Episodes Number: %s | timestep number: %s| Gen_T: %.2f' % (g, np.array(kid_rewards).mean(), episodes_number, timestep_count, time.time()-t0))
            print('Gen:', g,
              '| Kid_avg_R: %.1f' % np.array(kid_rewards).mean(),
              '| episodes number:', episodes_number,
             	  '| timestep number:', timestep_count,
                  '| Gen_T: %.2f' %(time.time() - t0))

        if g % 40 == 0:
        # if True:
            test_times = 100
            test_rewards, timestep_count, episodes_number = test(model, pool, env, test_times, CONFIG)
            test_rewards_mean = np.mean(np.array(test_rewards))
            experiment_record['test_rewards'].append([g, test_rewards])
            logging.info("test model, Reward: %.1f" % test_rewards_mean)
            print(
                'Gen: ', g,
                '| Net_R: %.1f' % test_rewards_mean) 
        if test_rewards_mean >= CONFIG['eval_threshold']: break
        
        if (g-1)% 500 == 500 -1:
            CONFIG['ep_max_step'] += 150
            logging.info("Gen %s | adding max timestep" % g)

        if (g-1) % 1000 == 1000 -1:
            logging.info("Gen %s | storing model" % g)
            torch.save(model.state_dict(), checkpoint_name + 'generation'+str(g)+'.pt')
            with open("experiment_record"+checkpoint_name+'generation'+str(g)+".pickle", "wb") as f:
                pickle.dump(experiment_record, f)
    
    run_times =20
    mar = 0
    for j in range(run_times):
        # test trained net without noise
        net_r = get_reward(model, env, CONFIG['ep_max_step'], sigma, CONFIG, None,)
        mar += net_r
    mar = mar / run_times
    logging.info("test the final model")
    print("runing 100 times")
    print(
        'Gen: ', g,
        '| Net_R: %.1f' % mar,
        '| Kid_avg_R: %.1f' % kid_rewards.mean(),
        '| Gen_T: %.2f' % (time.time() - t0),)

    # ---------------SAVE---------
    torch.save(model.state_dict(), checkpoint_name + '.pt')
    with open("experiment_record"+str(namemark)+".pickle", "wb") as f:
        pickle.dump(experiment_record, f)


if __name__ == '__main__':
    main()
