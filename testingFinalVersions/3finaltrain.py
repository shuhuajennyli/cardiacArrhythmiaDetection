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
train_df=pd.read_csv('/home/cst2020/ScienceFair/TestingFinalVersions/processedtrain_df2.csv',header=None)
validation_df=pd.read_csv('/home/cst2020/ScienceFair/TestingFinalVersions/processedvalidation_df2.csv',header=None)
test_df=pd.read_csv('/home/cst2020/ScienceFair/TestingFinalVersions/processedtest_df2.csv',header=None)


#looking at random set of all the types of beats
random = train_df.groupby(288,group_keys=False).apply(lambda train_df : train_df.sample(1))

for x in range(5):
    plt.plot(random.iloc[x,:287])
    plt.show()


#making a function that adds gaussian noise to all the waves
def add_gaussian_noise(wave):
    noise = np.random.normal(0,0.05,287)
    return (wave + noise)


#converting dataframes to format for training
target_train=train_df[288]
target_validation=validation_df[288]
target_test=test_df[288]

y_train=to_categorical(target_train)
y_validation=to_categorical(target_validation)
y_test=to_categorical(target_test)

X_train=train_df.iloc[:,:287].values
X_validation=validation_df.iloc[:,:287].values
X_test = test_df.iloc[:,:287].values

for i in range(len(X_train)):
    X_train[i,:287] = add_gaussian_noise(X_train[i,:287])

X_train = X_train.reshape(len(X_train), X_train.shape[1],1)
X_validation = X_validation.reshape(len(X_validation), X_validation.shape[1],1)
X_test = X_test.reshape(len(X_test), X_test.shape[1],1)


#coding the training network
def network(X_train,y_train,X_validation,y_validation):

    #shaping it and setting it as input
    im_shape=(X_train.shape[1],1)
    input_beats=Input(shape=(im_shape), name='input_beats')

    #first convolution layer, with batchnormalization and dropout
    conv1_1=Convolution1D(64, 6, activation='relu', input_shape=im_shape)(input_beats)
    conv1_1=BatchNormalization()(conv1_1)
    conv1_1=Dropout(0.5)(conv1_1)

    #pooling and doing second convolution, batch normalization
    pool1=MaxPool1D(pool_size=3, strides=2, padding="same")(conv1_1)
    conv2_1=Convolution1D(64, 3, activation='relu', input_shape=im_shape)(pool1)
    conv2_1=BatchNormalization()(conv2_1)

    #pooling again then flattening it to the right shape, then plugging into two fully connected layers then one more for the right output shape

    pool2=MaxPool1D(pool_size=2, strides=2, padding="same")(conv2_1)
    flatten=Flatten()(pool2)
    dense_end = Dense(64, activation='relu')(flatten)
    main_output = Dense(5, activation='softmax', name='main_output')(dense_end)
    
    #making the model
    model = Model(inputs= input_beats, outputs=main_output)

    #optimizer adam
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    
    #saving model checkpoints
    callbacks = [ModelCheckpoint(filepath='finalmodel.h5', monitor='val_loss', save_best_only=True)]

    #training the model and saving it to a file, and saving history for validation and loss graphs
    history=model.fit(X_train, y_train,epochs=500, callbacks=callbacks, batch_size=100,validation_data=(X_validation,y_validation))
    model.load_weights('finalmodel.h5')
    model.save('finalmodel.h5')
    return(model,history)

#model evaluation function for graphing loss and accuracy
def evaluate_model(history,X_validation,y_validation,model):
    
    print(history)
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
    target_names=['0','1','2','3','4']
    
    y_true=[]
    for element in y_validation:
        y_true.append(np.argmax(element))
    prediction_proba=model.predict(X_validation)
    prediction=np.argmax(prediction_proba,axis=1)
    cnf_matrix = confusion_matrix(y_true, prediction)


from keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout
from keras.layers import Input
from keras.models import Model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint


model,history=network(X_train,y_train,X_validation,y_validation)
evaluate_model(history,X_validation,y_validation,model)