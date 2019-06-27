'''
Virtual Batch Normalization
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class VirtualBatchNorm2D(nn.Module):

    def __init__(self, num_features):
        super(VirtualBatchNorm2D, self).__init__()
        self.named_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))

        self.register_buffer('mean', torch.zeros(num_features))
        self.register_buffer('var', torch.ones(num_features))
        # used to calculate the current mean and variance
        self.bn = nn.BatchNorm2d(num_features,momentum=1.0, affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.weight)
        init.zeros_(self.bias)
        self.mean.zero_()
        self.var.fill_(1)
    
    def set_mean_var(self, mean, var):
        self.mean = mean
        self.var = var
    
    def set_mean_var_from_bn(self, bn):
        self.mean = bn.running_mean
        self.var = bn.running_var

    def forward(self, input):
        self._check_input_dim(input)
        batch_size = input.size()[0]
        new_coeff = 1. / (batch_size + 1)
        old_coeff = 1. - new_coeff
        output = self.bn(input)
        new_mean = self.bn.running_mean
        new_var = self.bn.running_var
        mean = new_coeff * new_mean + old_coeff * self.mean
        var = new_coeff * new_var + old_coeff * self.var
        return F.batch_norm(input, mean, var, self.weight, self.bias)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))