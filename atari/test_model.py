
import os
import gym
import click
import torch
import pandas as pd
import numpy as np

from torch import multiprocessing as mp
from train import test
#model_storage_path = '/home/yyl/model/es-rl/'
model_storage_path = ''


@click.command()
@click.option("--model", type=str, help='the name of policy model')
@click.option("--modeltype", type=click.Choice(['2015', '2013']))
@click.option("--gamename", type=str, help='the name of test game')
@click.option("--ncpu", type=int, help="the number of cpu")
def main(model, modeltype, gamename, ncpu):
    if modeltype == '2015':
        from model_15 import build_model
    elif modeltype == '2013':
        from model_13 import build_model
    config = pd.read_csv("config.csv")
    CONFIG = dict()
    CONFIG['game'] = gamename + '-v0'
    CONFIG['ep_max_step'] = 1500
    CONFIG['eval_threshold'] = config[config['gamename']==gamename].iloc[0,1]
    CONFIG['l2coeff'] = 0.005
    pool = mp.Pool(processes=ncpu)
    env = gym.make(CONFIG['game'])
    CONFIG['n_action'] = env.action_space.n

    test_times = 100
    
    test_model =build_model(CONFIG) 
    test_model.load_state_dict(torch.load(os.path.join(model_storage_path, model)))
    test_model.switch_to_vbn()

    test_rewards, _ = test(test_model, pool, env, test_times, CONFIG)

    print("test results:", np.array(test_rewards).mean())
    print(str(test_rewards))

if __name__ == '__main__':
    main()
