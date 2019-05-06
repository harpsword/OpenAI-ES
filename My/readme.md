# It's for ES-RL

## experiment

selected game:

1. Assault-v0, 1673.9
2. Amidar-v0, 112
3. BeamRider-v0, 744
4. Asteroids-v0, 1562.0
5. Boxing-v0, 49.8

## TODO

1. add weight decay 

add an item to train.py train functioin, cumulative_update.
This item is - coeff * theta.

### selected Game

| Game Name    | objective performance | performance of mine | timestep one batch | lr   | sigma | batch size | Comment         |
| ------------ | --------------------- | ------------------- | ------------------ | ---- | ----- | ---------- | --------------- |
| Assault-v0 | 1673.9                |                     |                    |      |       |            |                 |
| Amidar-v0    | 112.0                 | 110.8               | 10w                | 0.01 |       |            |                 |
| BeamRider-v0 |  744                  |                     |                    |      |       |            |                 |
| Asteroids-v0 |    1562.0             |                     |                    |      |       |            |                 |
| Boxing-v0    |          49.8         |                     |                    |      |       |            |                 |
|              |                       |                     |                    |      |       |            |                 |
| Amidar-v0    | 112.0                 | 110.8               | 10w                | 0.01 | 0.02  | 400        | No weight decay |




