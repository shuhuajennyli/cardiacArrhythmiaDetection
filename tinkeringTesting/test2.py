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


train_df=pd.read_csv('/home/cst2020/ScienceFair/processedvalidation_df.csv',header=None)

print(train_df)

train_df[288]=train_df[288].astype(int)
equilibre=train_df[288].value_counts()
print(equilibre)






'''
0    56498
2     4956
4      677
1       28
3        2
'''
'''
0    75002
2     7129
4      802
1       33
3        2
'''