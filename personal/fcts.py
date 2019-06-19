import os
import torch
import sys
import random
import pandas as pd
import numpy as np
from tqdm import tqdm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(L):
    """
    OR
        e_l = np.exp(L)
        sum_l = sum(e_l)
        new_l = []
        for i in e_l:
            new_l.append(i*1.0/sum_l)
        return new_l
    :param L:
    :return:
    """
    e_l = np.exp(L - np.max(L))
    return e_l / e_l.sum(axis=0)


def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))


def random_seed_everything(seed=10):
    # https://www.kaggle.com/mlwhiz/third-place-model-for-toxic-comments-in-pytorch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

