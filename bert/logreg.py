import os
import sys
import pandas as pd

import torch
from torch.utils.data import DataLoader

# BertのモデルとTokenizer(前処理用)をimport
from transformers import BertTokenizer, BertModel

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import DATA_DIR  # noqa


tweet_df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_airline_tweet.csv"))
