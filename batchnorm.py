
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class VitualBatchNorm2D(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(VitualBatchNorm2D, self).__init__()
        self.named_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))

        self.register_buffer('mean', torch.zeros(num_features))
        self.register_buffer('var', torch.ones(num_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.weight)
        init.zeros_(self.bias)
        self.mean.zero_()
        self.var.fill_(1)
    
    def set_mean_var(self, mean, var):
        self.mean = mean
        self.var = var

    def forward(self, input):
        return F.batch_norm(input, self.mean, self.var, self.weight, self.bias)