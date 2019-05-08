# It's for ES-RL

## experiment

selected game:

1. Assault-v0, 1673.9
2. Amidar-v0, 112
3. BeamRider-v0, 744
4. Asteroids-v0, 1562.0
5. Boxing-v0, 49.8

## TODO

1. add weight decay, Finished
2. output timestep_all_count, when testing model

add an item to train.py train functioin, cumulative_update.
This item is - coeff * theta.

### selected Game

没有说明，默认是使用vbn。

| Game Name    | objective performance | performance of mine | timestep one batch | lr   | sigma | batch size | Comment         | machine      |
| ------------ | --------------------- | ------------------- | ------------------ | ---- | ----- | ---------- | --------------- | ------------ |
| Assault-v0   | 1673.9                |                     |                    | 0.01 | 0.02  | 140        |                 | dell 99      |
| Amidar-v0    | 112.0                 |                     | 10w                | 0.01 | 0.02  | 400        |                 | lenovo node1 |
| BeamRider-v0 | 744                   | 974.9               | 34w                | 0.01 | 0.02  | 400        | weight decay    | dell 113     |
| Asteroids-v0 | 1562.0                | 140                 | 6w                 | 0.01 | 0.02  | 300        | weight decay    | dell 205     |
| Boxing-v0    | 49.8                  |                     |                    | 0.01 | 0.02  | 400        |                 |              |
| Amidar-v0    | 112.0                 | 110.8               | 10w                | 0.01 | 0.02  | 400        | No weight decay | finished     |


