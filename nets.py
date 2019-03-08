import os
from collections import namedtuple
import time
from torch.nn import functional as F
import logging
from torch import nn
import torch as t
# from utils import array_tool as at
# from utils.vis_tool import Visualizer

from config import opt

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        """
        num_input_features: the number of input feature maps
        growth_rate:
        grow_rate * bn_size:
        drop_rate:
        """
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm1d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('fc1', nn.Linear(num_input_features, bn_size * growth_rate)),
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('fc2', nn.Linear(bn_size * growth_rate, growth_rate)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return t.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        """
        num_layers: number of dense layers in every block
        """
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

# _Transition, half the number of feature maps
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        """
            num_input_features: the number of input feature maps
            num_output_features:the number of output feature maps, i.e. num_input_features/2
        """
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('fc', nn.Linear(num_input_features, num_output_features))

# a example of densenet
class DenseNet(nn.Module):
    
    def __init__(self, growth_rate=5, block_config=(4, 6, 8),
                 num_init_features=13, bn_size=4, drop_rate=0.5, num_classes=1):
        
        super(DenseNet, self).__init__()

        # first conv
        # self.features = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
        #     ('norm0', nn.BatchNorm2d(num_init_features)),
        #     ('relu0', nn.ReLU(inplace=True)),
        #     ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        # ]))
        self.features = nn.Sequential()
        # every denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        
        #  batch norm
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))
        
        # classifier
        self.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 13)
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.classifier(features)
        return out

# a example of densenet
class DenseNet_in39(nn.Module):
    
    def __init__(self, growth_rate=10, block_config=(4, 6, 8),
                 num_init_features=39, bn_size=4, drop_rate=0.5, num_classes=1):
        
        super(DenseNet_in39, self).__init__()

        # first conv
        # self.features = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
        #     ('norm0', nn.BatchNorm2d(num_init_features)),
        #     ('relu0', nn.ReLU(inplace=True)),
        #     ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        # ]))
        self.features = nn.Sequential()
        # every denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        
        #  batch norm
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))
        
        # classifier
        self.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 13)
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.classifier(features)
        return out