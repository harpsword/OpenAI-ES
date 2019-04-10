import torch

class SGD(object): 
    '''
    self.v: store the cumulative gradients according to update rule with momentum
    '''
    def __init__(self, named_parameters, learning_rate, momentum=0.9):
        self.v = dict()
        self.lr, self.momentum = learning_rate, momentum
        for name, params in named_parameters:
            self.v[name] = torch.zeros_like(params, dtype=torch.float)
            # print(name)
            # print(params.data.size())
        
    def zero_grad(self):
        for name, params in self.v.item():
            self.v[name] = torch.zeros_like(self.v[name], dtype=torch.float)

    def update_model_parameters(self, model, gradients):
        '''
        update self.v with gradients
        update model's parameter with self.v
        '''
        if not isinstance(gradients, dict):
            raise TypeError("the gradients must be a dict with key(name)"
                            "value(torch.tensor), Got{}".format(torch.typename(gradients)))
        for name, params in self.v.items():
            self.v[name].mul_(self.momentum)
            self.v[name].add_(gradients[name]*(1-self.momentum))
        # update parameter of model recursively
        with torch.no_grad():
            for name, param in model.named_parameters():
                '''
                example : name(conv1.weight)
                model.conv1 is Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1))
                model.conv1.weight is the param
                '''
                tmp = model
                for attr_value in name.split('.'):
                    tmp = getattr(tmp, attr_value)
                tmp.add_(self.lr*self.v[name])