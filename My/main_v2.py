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
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from config import N_POPULATION, N_GENERATION, LR, SIGMA, TIMESTEP_LIMIT
from optimizer import SGD
from train_v2 import train, test

torch.set_num_threads(1)
LogFolder = os.path.join(os.getcwd(), 'log')
model_storage_path = '/home/yyl/model/es-rl/'

Small_value = -1000


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
@click.option("--modeltype", type=click.Choice(['2015', '2013']))
def main(namemark, ncpu, batchsize, generation, lr, sigma, vbn, vbn_test_g, gamename, logfile,modeltype):
    if modeltype == '2015':
        from model_15_v2 import build_model
    elif modeltype == '2013':
        from model_13_v2 import build_model
    setup_logging(logfile)

    logging.info("modeltype: %s", modeltype)
    logging.info("learning rate: %s", lr)
    logging.info("sigma: %s", sigma)
    logging.info("Game name: %s", gamename)
    logging.info("batchsie: %s", batchsize)
    logging.info("ncpu:%s", ncpu)
    logging.info("namemark:%s", namemark)
    print("learning rate:",lr)
    print("sigma:", sigma)
    print("gamename:", gamename)
    print("batchsie:", batchsize)
    print("ncpu:", ncpu)
    print("namemark", namemark)

    checkpoint_name = gamename + namemark + "-sigma" + str(sigma) +'-lr' + str(lr)+'-model'+modeltype

    import pandas as pd
    config = pd.read_csv('config.csv')
    CONFIG = dict()
    CONFIG['game'] = gamename + '-v0'
    # it's for training frames
    CONFIG['ep_max_step'] = 1500
    CONFIG['eval_threshold'] = config[config['gamename']==gamename].iloc[0,1]
    CONFIG['l2coeff'] = 0.005
    test_times = ncpu - 1

    logging.info("Settings: %s", str(CONFIG))

    env = gym.make(gamename+'-v0')
    CONFIG['n_action'] = env.action_space.n
    experiment_record = {}
    experiment_record['kid_rewards'] = []
    experiment_record['test_rewards'] = []

    device = torch.device("cpu")
    model = build_model(CONFIG).to(device)
    model_best = build_model(CONFIG)
    model_before = build_model(CONFIG)
    best_test_score = Small_value

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
            model, kid_rewards, _, _ = train(model, optimizer, pool, sigma, env, test_episodes, CONFIG, modeltype)
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
    best_kid_mean = Small_value 
    for g in range(generation):
        t0 = time.time()
        model_before.load_state_dict(model.state_dict())
        model, kid_rewards, timestep_count = train(model, optimizer, pool, sigma, env, int(batchsize/2), CONFIG, modeltype)
        training_timestep_count += timestep_count
        timestep_count = timestep_count / 4
        if training_timestep_count > TIMESTEP_LIMIT:
            logging.info("satisfied timestep limit")
            logging.info("Now timestep %s" % training_timestep_count)
            break
        kid_rewards_mean = np.array(kid_rewards).mean()
        experiment_record['kid_rewards'].append([g, np.array(kid_rewards).mean()])
        if g % 5 == 0:
            logging.info('Gen: %s | Kid_avg_R: %.1f | Episodes Number: %s | timestep number: %s| Gen_T: %.2f' % (g, np.array(kid_rewards).mean(), batchsize, timestep_count, time.time()-t0))
            print('Gen:', g,
              '| Kid_avg_R: %.1f' % np.array(kid_rewards).mean(),
              '| episodes number:', batchsize,
             	  '| timestep number:', timestep_count,
                  '| Gen_T: %.2f' %(time.time() - t0))
        if kid_rewards_mean > best_kid_mean:
            best_kid_mean = kid_rewards_mean
            test_rewards, timestep_count = test(model_before, pool, env, test_times, CONFIG)
            test_rewards_mean = np.mean(np.array(test_rewards))
            experiment_record['test_rewards'].append([g, test_rewards])
            logging.info('Gen: %s | Kid_avg_R: %.1f | Episodes Number: %s | timestep number: %s| Gen_T: %.2f' % (g, np.array(kid_rewards).mean(), batchsize, timestep_count, time.time()-t0))
            
            logging.info("test model, Reward: %.1f" % test_rewards_mean)
            logging.info("train progross %s/%s" % (training_timestep_count, TIMESTEP_LIMIT))
            print(
                'Gen: ', g,
                '| Net_R: %.1f' % test_rewards_mean) 
            if test_rewards_mean > best_test_score:
                best_test_score = test_rewards_mean
                model_best.load_state_dict(model_before.state_dict())
                # save when found a better model
                logging.info("Storing Best model")
                torch.save(model_best.state_dict(), model_storage_path+checkpoint_name+'best_model.pt')
        
        if g % 20 == 0:
            test_rewards, timestep_count = test(model, pool, env, test_times, CONFIG)
            test_rewards_mean = np.mean(np.array(test_rewards))
            experiment_record['test_rewards'].append([g, test_rewards])
            logging.info("test model, Reward: %.1f" % test_rewards_mean)
            logging.info("train progross %s/%s" % (training_timestep_count, TIMESTEP_LIMIT))
            print(
                'Gen: ', g,
                '| Net_R: %.1f' % test_rewards_mean) 
            if test_rewards_mean > best_test_score:
                best_test_score = test_rewards_mean
                model_best.load_state_dict(model.state_dict())
                # save when found a better model
                logging.info("Storing Best model")
                torch.save(model_best.state_dict(), model_storage_path+checkpoint_name+'best_model.pt')
        #if test_rewards_mean >= CONFIG['eval_threshold']: break
        
        if (g-1)% 500 == 500 -1:
            CONFIG['ep_max_step'] += 150
            logging.info("Gen %s | adding max timestep" % g)

        if (g-1) % 1000 == 1000 -1:
            logging.info("Gen %s | storing model" % g)
            torch.save(model.state_dict(), model_storage_path+checkpoint_name + 'generation'+str(g)+'.pt')
            torch.save(model_best.state_dict(), model_storage_path+checkpoint_name+'best_model.pt')
            with open(model_storage_path+"experiment_record"+checkpoint_name+'generation'+str(g)+".pickle", "wb") as f:
                pickle.dump(experiment_record, f)
    
    test_rewards, _ = test(model, pool, env, test_times, CONFIG)
    test_rewards_mean = np.mean(np.array(test_rewards))
    logging.info("test final model, Mean Reward of %s times: %.1f" % (test_times, test_rewards_mean))

    if test_rewards_mean > best_test_score:
        best_test_score = test_rewards_mean
        model_best.load_state_dict(model.state_dict())
        logging.info("storing Best model")

    print("testing results :", test_rewards_mean)
    # ---------------SAVE---------
    torch.save(model_best.state_dict(), model_storage_path+checkpoint_name+'best_model.pt')
    torch.save(model.state_dict(), model_storage_path+checkpoint_name + '.pt')
    with open(model_storage_path+"experiment_record"+str(namemark)+".pickle", "wb") as f:
        pickle.dump(experiment_record, f)


if __name__ == '__main__':
    main()
