import warnings
warnings.filterwarnings('ignore')
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

#making data the way it needs to be to go into the model
test_df=pd.read_csv('/Users/jennyli/Documents/scienceFairDL/cardiacArrhythmiaDetection/finalVersions/processedtest_df2.csv',header=None)

target_test=test_df[288]
y_test=to_categorical(target_test)

X_test = test_df.iloc[:,:287].values
X_test = X_test.reshape(len(X_test), X_test.shape[1],1)


#Loading the model in
model = load_model('finalmodel.h5')

#predicting test file
y_pred = model.predict(X_test)

#turning in understandable format
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

#balanced accuracy score as the test set is imbalanced
print(balanced_accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))
model.summary()

#Making confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision = 2)

cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
classes = ['AFIB', 'N', 'AFL', 'BII', 'VFL']

plt.figure(figsize = (8,8))
plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation = 45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	plt.text(j, i, format(cm[i, j], '.2f')),
	horizontalalignment = 'center',
	color = 'white' if cm[i, j] > thresh else 'black'

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()
