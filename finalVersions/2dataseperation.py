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
from sklearn.model_selection import train_test_split
import warnings
from sklearn.utils import resample

#reading in the full dataframe with all the beats
df=pd.read_csv('/Users/jennyli/Documents/scienceFairDL/cardiacArrhythmiaDetection/finalVersions/processednew.csv',header=None)

#counting number of beats for each type of arrhythmia
df[288]=df[288].astype(int)
val_count=df[288].value_counts()

#putting all beats of a type of arrhythmia into their respective variables
df_1=(df[df[288]==1]).sample(n = 79566, random_state = 42)
df_0=(df[df[288]==0]).sample(n = 11478, random_state = 100)
df_2=(df[df[288]==2]).sample(n = 1381, random_state = 88)
df_4=(df[df[288]==4]).sample(n = 472, random_state = 8)
df_3=(df[df[288]==3]).sample(n = 422, random_state = 888)


#splitting the beats of each category into train and test (this makes sure none of the test beats have been used to train)
df_1_train = df_1.sample(frac = 0.8, random_state = 42)
df_1_test = df_1.drop(df_1_train.index)

df_0_train = df_0.sample(frac = 0.8, random_state = 100)
df_0_test = df_0.drop(df_0_train.index)

df_2_train = df_2.sample(frac = 0.8, random_state = 88)
df_2_test = df_2.drop(df_2_train.index)

df_4_train = df_4.sample(frac = 0.8, random_state = 8)
df_4_test = df_4.drop(df_4_train.index)

df_3_train = df_3.sample(frac = 0.8, random_state = 888)
df_3_test = df_3.drop(df_3_train.index)

#putting all the test into one dataframe
test_df = pd.concat([df_0_test, df_1_test, df_2_test, df_3_test, df_4_test])


#splitting the beats of each category in train into actual train and validation
df_1_atrain = df_1_train.sample(frac = 0.8, random_state = 42)
df_1_val = df_1_train.drop(df_1_atrain.index)

df_0_atrain = df_0_train.sample(frac = 0.8, random_state = 42)
df_0_val = df_0_train.drop(df_0_atrain.index)

df_2_atrain = df_2_train.sample(frac = 0.8, random_state = 42)
df_2_val = df_2_train.drop(df_2_atrain.index)

df_4_atrain = df_4_train.sample(frac = 0.8, random_state = 42)
df_4_val = df_4_train.drop(df_4_atrain.index)

df_3_atrain = df_3_train.sample(frac = 0.8, random_state = 42)
df_3_val = df_3_train.drop(df_3_atrain.index)


#combining all the validation samples
validation_df = pd.concat([df_2_val, df_0_val, df_1_val, df_4_val, df_3_val])


#sampling and upsampling the beats for actual train
df_1_sample = (df_1_atrain.sample(n = 30000, random_state = 42))
df_0_upsample = resample(df_0_atrain, n_samples = 30000, random_state = 100)
df_2_upsample = resample(df_2_atrain, n_samples = 30000, random_state = 88)
df_3_upsample = resample(df_3_atrain, n_samples = 30000, random_state = 8)
df_4_upsample = resample(df_4_atrain, n_samples = 30000, random_state = 888)

#combining all the actual train samples
train_df = pd.concat([df_1_sample, df_0_upsample, df_2_upsample, df_3_upsample, df_4_upsample])

#creating files of data
test_df.to_csv(r'/Users/jennyli/Documents/scienceFairDL/cardiacArrhythmiaDetection/FinalVersions/processedtest_df2.csv', header=None, index=None, sep=',', mode='a+')
train_df.to_csv(r'/Users/jennyli/Documents/scienceFairDL/cardiacArrhythmiaDetection/FinalVersions/processedtrain_df2.csv', header=None, index=None, sep=',', mode='a+')
validation_df.to_csv(r'/Users/jennyli/Documents/scienceFairDL/cardiacArrhythmiaDetection/FinalVersions/processedvalidation_df2.csv', header=None, index=None, sep=',', mode='a+')
