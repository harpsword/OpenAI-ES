
# Evolution Strategies for Reinforcement Learning

## Environment

Atari: 
1. My 
2. MyNew
3. MyNew2 

MuJoCo
1. mujoco(MyNew2's mujoco version)

## 机制说明

My文件夹下采用的vbn机制：

fc1无bn层

1. collect reference mean and var: 在训练之前，模型为bn模式，按照训练的方式与环境进行10代，10代之后将模型中的bn层参数(mean,variance)作为reference mean与variance
2. 之后，模型切换到vbn模式，在正常forward时，将当前帧的mean和var与reference mean、var做平均。

MyNew文件夹下采用的vbn机制为

fc1有bn层

1. collect reference: 在训练之前，采用随机策略来与环境进行交互，每一帧按照1%的概率被选入到reference frames set里，集合大小为128.
2. 在forward的时候，先将reference frames set输入(bn模式)，计算reference mean and variance，再切换到vbn模式，在正常forward时，直接使用reference mean and variance。

MyNew2文件夹下采用的vbn机制为

fc1有bn层

1. collect reference: 在训练之前，采用随机策略来与环境进行交互，每一帧按照1%的概率被选入到reference frames set里，集合大小为128.
2. 在forward的时候，先将reference frames set输入(bn模式)，计算reference mean and variance，再切换到vbn模式，在正常forward时，将当前帧的mean和var与reference mean、var做平均。

## Note for implementation

1. use numpy.random.normal instead of torch.randn_like
2. create a new model when trying to get rewards


### required package

1. redis
2. hiredis
3. click
4. torch(1.0)
5. torchvision

## Note

Notice:

1. multi-core
2. with batch normalization
3. use np.random.normal to represent torch.randn_like

Please make sure OMP_NUM_THREADS=1
or run 

export OMP_NUM_THREADS=1

