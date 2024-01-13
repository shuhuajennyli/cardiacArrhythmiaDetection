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


train_df=pd.read_csv('/home/cst2020/ScienceFair/processedtopfive.csv',header=None)

print(train_df)

train_df[288]=train_df[288].astype(int)
equilibre=train_df[288].value_counts()
print(equilibre)


'''final
6     79584
1     11482
8      7971
3      3287
10     1779
2      1381
12     1359
9       915
13      472
11      467
4       422
14      399
5       139
7       111
0        97'''