'''
Frame skip = 4
no limitation of timestep one batch


1. separate ProcessUnit to preprocess.py
2. add shared noise table
'''
import os
import click
import gym
import torch
import time
import pickle
import logging
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from optimizer import SGD
from train import train, test 
from model import build_model

torch.set_num_threads(1)
LogFolder = os.path.join(os.getcwd(), 'log')
model_storage_path = '/home/yyl/model/es-rl/'

Small_value = -1000000

class ARGS(object):
    state_dim = 0
    action_dim = 0
    action_lim = 0
    test_times = 0

    timestep_limit = int(1e7)
    timestep_limit_episode = 5000
    # parameter for input
    namemark = ""
    ncpu = 1
    population_size = 36
    generation = 50000
    lr = 0.02
    sigma = 0.02
    gamename = ""
    logfile = ""

    # parameter not change
    # TODO
    l2coeff = 0.005




def check_env(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high
    action_low = env.action_space.low
    assert len(action_low) == len(action_high)
    for i in range(len(action_low)):
        if abs(action_low[i]) != abs(action_high[i]):
            raise ValueError("Environment Error with wrong action low and high")
    return state_dim, action_dim, action_high[0]


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

 
def main(ARGS):
    setup_logging(ARGS.logfile)

    logging.info("learning rate: %s", ARGS.lr)
    logging.info("sigma: %s", ARGS.sigma)
    logging.info("Game name: %s", ARGS.gamename)
    logging.info("batchsie: %s", ARGS.population_size)
    logging.info("ncpu:%s", ARGS.ncpu)
    logging.info("namemark:%s", ARGS.namemark)

    checkpoint_name = ARGS.gamename + ARGS.namemark + "-sigma" + str(ARGS.sigma) +'-lr' + str(ARGS.lr)

    env = gym.make(ARGS,gamename)
    state_dim, action_dim, action_lim = check_env(env)
    ARGS.test_times = ARGS.ncpu - 1


    experiment_record = {}
    experiment_record['kid_rewards'] = []
    experiment_record['test_rewards'] = []

    device = torch.device("cpu")
    model = build_model(ARGS).to(device)
    model_best = build_model(ARGS)
    best_test_score = Small_value

    # utility instead reward for update parameters (rank transformation)
    base = ARGS.population_size   # *2 for mirrored sampling
    if ARGS.population_size % 2 == 1:
        print("need an even batch size")
        exit()
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base

    optimizer = SGD(model.named_parameters(), ARGS.lr)
    pool = mp.Pool(processes=ARGS.ncpu)

    timestep_count = 0
    test_result_list = []
    for g in range(ARGS.generation):
        t0 = time.time()
        # TODO
        model, kid_rewards, timestep_generation = train(model, optimizer, pool, ARGS.sigma, env, int(population_size/2), ARGS) 
        timestep_count += timestep_generation
        timestep_generation = timestep_count / 4
        if timestep_count > TIMESTEP_LIMIT:
            logging.info("satisfied timestep limit")
            logging.info("Now timestep %s" % timestep_count)
            break
        kid_rewards_mean = np.array(kid_rewards).mean()
        experiment_record['kid_rewards'].append([g, np.array(kid_rewards).mean()])
        if g % 5 == 0:
            logging.info('Gen: %s | Kid_avg_R: %.1f | Episodes Number: %s | timestep number: %s| Gen_T: %.2f' % (g, np.array(kid_rewards).mean(), population_size, timestep_generation, time.time()-t0))
            print('Gen:', g,
              '| Kid_avg_R: %.1f' % np.array(kid_rewards).mean(),
              '| episodes number:', population_size,
             	  '| timestep number:', timestep_generation,
                  '| Gen_T: %.2f' %(time.time() - t0))
        if kid_rewards_mean > best_kid_mean:
            best_kid_mean = kid_rewards_mean
            test_rewards, _= test(model_before, pool, env, test_times, CONFIG, reference_batch_torch)
            test_rewards_mean = np.mean(np.array(test_rewards))
            experiment_record['test_rewards'].append([g, test_rewards])
            logging.info("Gen: %s, test model, Reward: %.1f" % (g, test_rewards_mean))
            #logging.info("train progross %s/%s" % (timestep_count, TIMESTEP_LIMIT))
            print(
                'Gen: ', g,
                '| Net_R: %.1f' % test_rewards_mean) 
            if test_rewards_mean > best_test_score:
                best_test_score = test_rewards_mean
                model_best.load_state_dict(model_before.state_dict())
                # save when found a better model
                #logging.info("Storing Best model")
                torch.save(model_best.state_dict(), model_storage_path+checkpoint_name+'best_model.pt')
        
        if g % 5 == 0:
            test_rewards, timestep_generation = test(model, pool, env, test_times, CONFIG, reference_batch_torch)
            test_rewards_mean = np.mean(np.array(test_rewards))
            experiment_record['test_rewards'].append([g, test_rewards])
            #logging.info("test model, Reward: %.1f" % test_rewards_mean)
            test_result_list.append(test_rewards_mean)
            print(
                'Gen: ', g,
                '| Net_R: %.1f' % test_rewards_mean) 
            if test_rewards_mean > best_test_score:
                best_test_score = test_rewards_mean
                model_best.load_state_dict(model.state_dict())
                # save when found a better model
                #logging.info("Storing Best model")
                torch.save(model_best.state_dict(), model_storage_path+checkpoint_name+'best_model.pt')
        if g % 40 == 0:
            logging.info("train progross %s/%s" % (timestep_count, TIMESTEP_LIMIT))
            logging.info("best test result:%s" % best_test_score)
            logging.info("test result:%s" % str(test_result_list))
            test_result_list = []
        
        if (g-1)% 500 == 500 -1:
            CONFIG['ep_max_step'] += 150
            logging.info("Gen %s | adding max timestep" % g)

        if (g-1) % 1000 == 1000 -1:
            logging.info("Gen %s | storing model" % g)
            torch.save(model.state_dict(), model_storage_path+checkpoint_name + 'generation'+str(g)+'.pt')
            torch.save(model_best.state_dict(), model_storage_path+checkpoint_name+'best_model.pt')
            with open(model_storage_path+"experiment_record"+checkpoint_name+'generation'+str(g)+".pickle", "wb") as f:
                pickle.dump(experiment_record, f)
    
    test_rewards, _ = test(model, pool, env, test_times, CONFIG, reference_batch_torch)
    test_rewards_mean = np.mean(np.array(test_rewards))
    logging.info("test final model, Mean Reward of %s times: %.1f" % (test_times, test_rewards_mean))

    if test_rewards_mean > best_test_score:
        best_test_score = test_rewards_mean
        model_best.load_state_dict(model.state_dict())
        logging.info("storing Best model")

    print("best test results :", best_test_score)
    logging.info("best test results:%s" % best_test_score)
    # ---------------SAVE---------
    torch.save(model_best.state_dict(), model_storage_path+checkpoint_name+'best_model.pt')
    torch.save(model.state_dict(), model_storage_path+checkpoint_name + '.pt')
    with open(model_storage_path+"experiment_record"+str(namemark)+".pickle", "wb") as f:
        pickle.dump(experiment_record, f)



@click.command()
@click.argument("namemark")
@click.option("--ncpu", default=mp.cpu_count()-1, help='The number of cores', type=int)
@click.option("--population_size", default=ARGS.population_size, help='batch size(population size)', type=int)
@click.option("--generation", default=ARGS.generation, help='the number of generation', type=int)
@click.option("--lr", default=ARGS.lr, help='learning rate')
@click.option("--sigma", default=ARGS.sigma, help='the SD of perturbed noise')
@click.option("--gamename", default="Assualt-v0", help="the name of tested game")
@click.option("--logfile", default="default.log", help='the file of log')
def run(namemark, ncpu, population_size, generation, lr, sigma, gamename, logfile):
    ARGS.namemark = namemark
    ARGS.ncpu = ncpu
    ARGS.population_size = population_size
    ARGS.generation = generation
    ARGS.lr = lr
    ARGS.sigma = sigma
    ARGS.gamename = gamename
    ARGS.logfile = logfile
    main(ARGS)


if __name__ == '__main__':
    main()
