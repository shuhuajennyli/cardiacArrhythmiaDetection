import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
import warnings

#reading in the file as a dataframe
train_df=pd.read_csv('/Users/jennyli/Documents/scienceFairDL/cardiacArrhythmiaDetection/finalVersions/processedtest_df2.csv',header=None)
print(train_df)

#taking the annotations and turning them into integers
train_df[288]=train_df[288].astype(int)

#figuring how many there are of each
distribution=train_df[288].value_counts()
print(distribution)
