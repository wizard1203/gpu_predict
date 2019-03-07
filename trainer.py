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
from torchnet.meter import ConfusionMeter, AverageValueMeter
from myoptimizer import get_optimizer

class GPUNetTrainer(nn.Module):
    """
    Args:
        
            
    """

    def __init__(self, gpu_net):
        super(GPUNetTrainer, self).__init__()

        self.gpu_net = gpu_net

        # optimizer
        self.optimizer = t.optim.SGD(self.gpu_net.parameters(), lr = opt.lr, weight_decay = opt.weight_decay, momentum=0.9)

        # visdom wrapper
        # self.vis = Visualizer(env=opt.env)

    def forward(self, datas):
        """

        Args:

        Returns:
            
        """
        pred = self.gpu_net(datas)
        return pred

    def train_step(self, label, datas):
        # switch to train mode
        self.gpu_net.train()

        self.optimizer.zero_grad()
        pred = self.forward(datas)
        # print('=======after forward ======pred:{}===='.format(pred))
        loss = F.nll_loss(pred, label)
        # print('=======after cul loss ======loss:{}===='.format(loss))
        loss.backward()
        self.optimizer.step()


        # self.update_meters(losses)
        return loss, pred

    def scale_lr(self):
        lastlr = opt.lr
        opt.lr *= opt.lr_decay
        print("=========*** lr{} change to lr{}==========\n".format(lastlr, opt.lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = opt.lr
        return self.optimizer

    def save(self, save_optimizer=True, better = False, save_path=None):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.gpu_net.state_dict()
        save_dict['config'] = opt._state_dict()
        # save_dict['vis_info'] = self.vis.state_dict()
        save_dict['optimizer'] = self.optimizer.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        # if save_path is None:
        # save_path = 'checkpoints/waternetparams'
        # save_path = opt.save_path
        
        if better:
            save_path = 'cur_best_params'
        else:
            # save_path = opt.save_path
            if opt.customize:
                save_name = 'model' + '_self_' + opt.arch + '_' + opt.optim + opt.kind + 'params.tar'
            else:
                save_name = 'model' + '_default_' + opt.arch + '_' + opt.optim + opt.kind + 'params.tar'
            save_path = os.path.join(opt.save_path, save_name)
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
        print(save_path)
        t.save(save_dict, save_path)
        # self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False):

        # if opt.customize:
        #     load_name = 'model' + '_self_' + opt.arch + '_' + opt.optim + opt.kind + 'params.tar'
        # else:
        #     load_name = 'model' + '_default_' + opt.arch  + '_' + opt.optim + opt.kind + 'params.tar'
        # state_dict = t.load(os.path.join(path, load_name))
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.gpu_net.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.gpu_net.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self
