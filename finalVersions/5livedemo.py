import warnings
warnings.filterwarnings('ignore')
import numpy as np
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
import warnings
from sklearn.utils import resample
from keras import models


while 1:
	#loading in the model and the test file, and defining classes
	test_df=pd.read_csv('/Users/jennyli/Documents/scienceFairDL/cardiacArrhythmiaDetection/finalVersions/processedtest_df2.csv', header=None)
	model = load_model('finalmodel.h5')
	classes = ['AFIB', 'N', 'AFL', 'BII', 'VFL']

	#choosing one random beat for each type
	random = test_df.groupby(288,group_keys=False).apply(lambda train_df : train_df.sample(1))

	#choose a class
	anyclass = int(input('What type of beat do you want to test? Enter 0:AFIB, 1:N, 2:AFL, 3:BII, 4:VFL '))
	if anyclass > 4 or anyclass < 0:
		print('Invalid input. Exiting...')
		break

	#plotting the beat
	print('Here is the graph of the random beat selected for you. It is a(n) %s beat' % classes[anyclass])
	randomplot = random.iloc[anyclass,:287]
	plt.plot(randomplot)
	plt.show()

	#making it the right shape to go into model
	#                    array rows, columns, number of items in each column
	random = random.iloc[anyclass,:287].values
	random = np.reshape(random,(1,287,1))

	#predicting and outputting
	print('Inputting your beat into the model...')
	pred_class = model.predict(random)
	pred_class = np.argmax(pred_class, axis=1)
	print('It has been predicted to be a(n) %s beat.' % classes[pred_class[0]])
	print('Thank you for playing!')
	break
