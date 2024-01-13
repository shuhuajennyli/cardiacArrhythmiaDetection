import numpy as np
from keras.models import load_model
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
from sklearn.metrics import balanced_accuracy_score
import itertools
from keras import models

#loading in the model and the test file, and defining classes
test_df=pd.read_csv('/home/cst2020/ScienceFair/FinalVersions/processedtest_df2.csv',header=None)
model = load_model('finalmodel.h5')
classes = ['AFIB', 'N', 'AFL', 'BII', 'VFL']

#choosing oine random beat for each type
random = test_df.groupby(288,group_keys=False).apply(lambda train_df : train_df.sample(1))

#choose a class
anyclass = 2
randomplot = random.iloc[anyclass,:287]
random = random.iloc[anyclass,:287].values

#plotting the beat
plt.plot(randomplot)
plt.show()

#making it the right shape to go into model
#                    array rows, colums, number of items in each column
random = np.reshape(random,(1,287,1))

#predicting and outputting
pred_class = model.predict(random)
pred_class = np.argmax(pred_class, axis=1)
print('This is most likely a/an %s beat.' % classes[pred_class[0]])