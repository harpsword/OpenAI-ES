

N_KID = 20         # half of the training population
N_POPULATION = 2*N_KID
N_GENERATION = 10000         # training step
LR = 0.05
SIGMA = 0.05

CONFIG = [
    dict(game='Assault-v0', n_feature=None, observation=(250, 160), n_action=7, ep_max_step=None,eval_threshold=2000),
    dict(game="CartPole-v0",
         n_feature=4, n_action=2, continuous_a=[False], ep_max_step=700, eval_threshold=500),
    dict(game="MountainCar-v0",
         n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
    dict(game="Pendulum-v0",
         n_feature=3, n_action=1, continuous_a=[True, 2.], ep_max_step=200, eval_threshold=-180)
][1]    # choose your game

SIMPLE_GAME = ['CartPole-v0', 'MountainCar-v0', 'Pendulum-v0']
GRAPH_GAME = ['Assault-v0']
