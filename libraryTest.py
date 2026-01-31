from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
from transformers import AutoModel
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import numpy as np
import torch