import os
import sys
import numpy as np
import random
import torch

import random
import copy
import time
import pandas as pd
import numpy as np
import gc
import re
# from torchtext import data
# import spacy
from tqdm import tqdm_notebook, tnrange
from tqdm import tqdm
tqdm.pandas(desc='Progress')
from collections import Counter
# from textblob import TextBlob
from nltk import word_tokenize



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def random_seed_everything(seed=10):
    # https://www.kaggle.com/mlwhiz/third-place-model-for-toxic-comments-in-pytorch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    