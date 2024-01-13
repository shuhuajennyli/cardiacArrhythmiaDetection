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

#reading in all the dataframes 
train_df=pd.read_csv('/Users/jennyli/Documents/scienceFairDL/cardiacArrhythmiaDetection/finalVersions/processedtrain_df2.csv',header=None)
validation_df=pd.read_csv('/Users/jennyli/Documents/scienceFairDL/cardiacArrhythmiaDetection/finalVersions/processedvalidation_df2.csv',header=None)
test_df=pd.read_csv('/Users/jennyli/Documents/scienceFairDL/cardiacArrhythmiaDetection/finalVersions/processedtest_df2.csv',header=None)


#looking at random set of all the types of beats
random = train_df.groupby(288, group_keys=False).apply(lambda train_df : train_df.sample(1))

#plotting them to make sure it is fine
for x in range(5):
    plt.plot(random.iloc[x,:287])
    plt.show()


#making a function that adds gaussian noise to all the waves
def add_gaussian_noise(wave):
    #drawing random samples from a Gaussian distribution
    #                        mean of distribution, width of distribution, output shape
    noise = np.random.normal(0,0.05,287)
    return (wave + noise)

#testing the gaussian noise function
plt.plot(add_gaussian_noise(random.iloc[x,:287]))


#converting dataframes to format for training, taking the annotation of each beat
target_train=train_df[288]
target_validation=validation_df[288]

#turning the annotations into the form that goes into a model
y_train=to_categorical(target_train)
y_validation=to_categorical(target_validation)

#taking the actual beat part of the data
X_train=train_df.iloc[:,:287].values
X_validation=validation_df.iloc[:,:287].values

#adding gaussian noise to every beat (every row)
for i in range(len(X_train)):
    X_train[i,:287] = add_gaussian_noise(X_train[i,:287])

#making the data the right format (three dimensional arrays) for training
#                           array row   columns(288)       items in each column
X_train = X_train.reshape(len(X_train), X_train.shape[1],1)
X_validation = X_validation.reshape(len(X_validation), X_validation.shape[1],1)


#coding the training network
def network(X_train,y_train,X_validation,y_validation):

    #setting variable shape and input
    im_shape=(X_train.shape[1],1)
    #shaping input so that it is (287,1)  
    input_beats=Input(shape=im_shape, name='input_beats')

    #first convolution layer, 64 output filters, length of convolution is 6
    #activation decides if a neuron should be activated by calculating weighted sum, introduces non-linearity
    conv1_1=Convolution1D(64, 6, activation='relu', input_shape=im_shape)(input_beats)
    conv1_1=BatchNormalization()(conv1_1)
    conv1_1=Dropout(0.5)(conv1_1)

    #max pooling, pool size is 3 and strides 2 until next pool
    #same padding applies padding so that it gets fully covered by the filter and stride
    pool1=MaxPool1D(pool_size=3, strides=2, padding="same")(conv1_1)
    #second convolution, 64 output filter, length of convolution is 3
    conv2_1=Convolution1D(64, 3, activation='relu', input_shape=im_shape)(pool1)
    #batchnormalization improves speed, performance, and stability
    conv2_1=BatchNormalization()(conv2_1)

    #flatten, then dense end with 64 neurons
    pool2=MaxPool1D(pool_size=2, strides=2, padding="same")(conv2_1)
    flatten=Flatten()(pool2)
    dense_end = Dense(64, activation='relu')(flatten)
    #dense end with 5 neurons to match output
    main_output = Dense(5, activation='softmax', name='main_output')(dense_end)
    
    #making the actual model
    model = Model(inputs= input_beats, outputs=main_output)

    #compiling it to train with adaptive learning rate optimization
    #cat. cross. loss function for single label categorization
    #evaluate accuracy during training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    
    #saving model checkpoints, etc, added to callbacks
    callbacks = [ModelCheckpoint(filepath='finalmodel.h5', monitor='val_loss', save_best_only=True)]

    #training the model and saving it to a file, and saving history for validation and loss graphs
    history=model.fit(X_train, y_train,epochs=500, callbacks=callbacks, batch_size=100,validation_data=(X_validation,y_validation))
    model.load_weights('finalmodel.h5')
    model.save('finalmodel.h5')
    return(model,history)

#model evaluation function for graphing loss and accuracy
def evaluate_model(history,X_validation,y_validation,model):
    
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    
    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()


from keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout
from keras.layers import Input
from keras.models import Model
from keras.models import load_model
from keras.layers import BatchNormalization
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint


model,history=network(X_train,y_train,X_validation,y_validation)
evaluate_model(history,X_validation,y_validation,model)