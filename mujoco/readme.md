# It's for ES-RL

MyNew2文件夹下采用的vbn机制为

1. collect reference: 在训练之前，采用随机策略来与环境进行交互，每一帧按照1%的概率被选入到reference frames set里，集合大小为128.
2. 在forward的时候，先将reference frames set输入，计算reference mean and variance，在正常forward时，将当前帧的mean和var与reference mean、var做平均。

