# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from re import sub
import torch
import csv
import itertools
import random
from random import shuffle
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split as split_data
import numpy as np
import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import torch
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import pandas as pd 

#IMPORTING LIBRARIES

#from data import Data
from sklearn.metrics import f1_score, average_precision_score
import sklearn

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
import time
import random
from torch import optim
import torch.nn.utils.rnn as rnn
