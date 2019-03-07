import os
import pandas as pd
import numpy as np
import logging
import random
import torch as t
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
from torch.utils import data as data_
from torch.utils.data import Dataset
import torch.utils.data.distributed
from torchvision import transforms as tvtsf
import torchvision.datasets as datasets
import torchvision.models as models
# from data import util

from config import opt
import logging
class gpuDataset:

    def __init__(self, columns, begin_num=0, end_num=500, data_dir='./gtx980-high-dvfs-real-small-workload-features.csv', split='train'):
        """
        Args:
            split:
        
        """
        self.file = data_dir
        self.begin_num = begin_num
        self.end_num = end_num
        self.columns = columns
        self.df = pd.read_csv(self.file, header=0)
        self.li = list(range(self.begin_num, self.end_num + 1))
        random.shuffle(self.li)

    def __len__(self):
        return self.end_num - self.begin_num

    def get_example(self, i):
        """Returns the i-th sample.

        Args:
            i (int): The index of the sample_files.

        Returns:
            a data sample

        """
        # Load a sample
        # label = self.df['avg_power'].loc[self.li[i]]
        # datas = self.df[self.columns].loc[self.li[i]]

        label = df['avg_power'].iloc[self.li[i]]
        datas = df.iloc[self.columns, self.li[i]]

        return label, datas

    __getitem__ = get_example



class TrainDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.db = gpuDataset(config.columns, config.train_begin, config.train_end, config.data_dir, split=split)

    def __getitem__(self, idx):
        label, datas = self.db.get_example(idx)
        label = t.from_numpy(np.array(label))
        datas = np.array(datas)

        datas = t.from_numpy(datas)
        datas = datas.contiguous().view(1,96,16)
        # TODO: check whose stride is negative to fix this instead copy all
        

        return label, datas

    def __len__(self):
        return len(self.db)


class TestDataset(Dataset):
    def __init__(self, config, split='test'):
        self.config = config
        self.db = gpuDataset(config.columns, config.test_begin, config.test_end, config.data_dir, split=split)

    def __getitem__(self, idx):
        label, datas = self.db.get_example(idx)
        label = t.from_numpy(np.array(label))
        datas = np.array(datas)

        datas = t.from_numpy(datas)
        datas = datas.contiguous().view(1,96,16)
        # TODO: check whose stride is negative to fix this instead copy all

        return label, datas

    def __len__(self):
        return len(self.db)



