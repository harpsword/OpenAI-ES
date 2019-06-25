# It's for ES-RL


## TODO

1. add weight decay, Finished
2. output timestep_all_count, when testing model

add an item to train.py train functioin, cumulative_update.
This item is - coeff * theta.


## Note

Notice:

1. multi-core
2. with batch normalization
3. use np.random.normal to represent torch.randn_like

Please make sure OMP_NUM_THREADS=1
or run 

export OMP_NUM_THREADS=1

