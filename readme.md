
# Evolution Strategies for Reinforcement Learning

## Note for implementation

1. use numpy.random.normal instead of torch.randn_like
2. create a new model when trying to get rewards


### required package

1. redis
2. hiredis
3. click
4. torch(1.0)
5. torchvision


### selected Game

| Game Name | objective performance | performance of mine | timestep one batch | lr | sigma | batchsize | Comment |
| Amidar-v0 |  112.0                |       110.8         |    10w             |  0.01 | 0.02  |  400      | No weight decay |
  
