

N_KID = 18         # half of the training population
N_POPULATION = 2*N_KID
N_GENERATION = 50000         # training step
LR = 0.02
SIGMA = 0.02
#LR = 0.05
#SIGMA = 0.05
FRAME_SKIP = 4
TIMESTEP_LIMIT = 1000000000
# reference batch size
reference_batch_size = 128


CONFIG = [
    dict(game='Assault-v0', n_feature=None, observation=(250, 160), n_action=7, ep_max_step=1000,eval_threshold=2000),
    dict(game="CartPole-v0",
         n_feature=4, n_action=2, continuous_a=[False], ep_max_step=700, eval_threshold=500),
    dict(game="MountainCar-v0",
         n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
    dict(game="Pendulum-v0",
         n_feature=3, n_action=1, continuous_a=[True, 2.], ep_max_step=200, eval_threshold=-180),
    dict(game="Amidar-v0", n_feature=None, observation=(250, 160), n_action=10, ep_max_step=1000, eval_threshold=100)
][4]    # choose your game

SIMPLE_GAME = ['CartPole-v0', 'MountainCar-v0', 'Pendulum-v0']
GRAPH_GAME = ['Assault-v0', 'Amidar-v0']

timesteps_per_batch = 100000
episodes_per_batch = 10000
