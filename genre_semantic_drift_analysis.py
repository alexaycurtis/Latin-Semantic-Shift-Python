#do the same thing but genre specific
import numpy as np
import pickle
from gensim.models import Word2Vec
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

