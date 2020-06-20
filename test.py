import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
wine = pd.read_csv(dataset_url)
