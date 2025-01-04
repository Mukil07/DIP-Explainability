
from __future__ import print_function, division
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Compose
from torch.utils.data import DataLoader
import numpy as np
import random

import tarfile
import io
import os
import pandas as pd

from torch.utils.data import Dataset
import torch


class OnlineMeanStd:
    def __init__(self):
        pass

    def __call__(self, dataset, batch_size, method='strong'):
        """
        Calculate mean and std of a dataset in lazy mode (online)
        On mode strong, batch size will be discarded because we use batch_size=1 to minimize leaps.

        :param dataset: Dataset object corresponding to your dataset
        :param batch_size: higher size, more accurate approximation
        :param method: weak: fast but less accurate, strong: slow but very accurate - recommended = strong
        :return: A tuple of (mean, std) with size of (3,)
        """

        if method == 'weak':
            loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=0)
            mean = 0.
            std = 0.
            nb_samples = 0.
            for data in loader:
                data = data['y_descreen']
                batch_samples = data.size(0)
                data = data.view(batch_samples, data.size(1), -1)
                mean += data.mean(2).sum(0)
                std += data.std(2).sum(0)
                nb_samples += batch_samples

            mean /= nb_samples
            std /= nb_samples

            return mean, std

        elif method == 'strong':
            loader = DataLoader(dataset=dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=0)
            mean_=[]
            std_=[]
            for j in range(2):
                cnt = 0
                fst_moment = torch.empty(3).cuda()
                snd_moment = torch.empty(3).cuda()
                #import pdb;pdb.set_trace()
                for i,data in enumerate(loader):
                    print(i)
                    video = data[j].permute((0,2,1,3,4)).cuda()
                    for data in torch.unbind(video, dim=1):

                        b, c, h, w = data.shape
                        nb_pixels = b * h * w
                        sum_ = torch.sum(data, dim=[0, 2, 3])
                        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
                        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
                        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

                        cnt += nb_pixels
                mean_.append(fst_moment)
                std_.append(torch.sqrt(snd_moment - fst_moment ** 2))
            return mean_,std_