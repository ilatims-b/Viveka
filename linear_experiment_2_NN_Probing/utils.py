import json
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import transformers
from baukit import TraceDict
from datasets import load_dataset
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM