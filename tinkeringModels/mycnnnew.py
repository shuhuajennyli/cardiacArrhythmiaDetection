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
df=pd.read_csv('/home/cst2020/ScienceFair/processeddiff.csv',header=None)

#counting number of beats for each type of arrhythmia
df[288]=df[288].astype(int)
val_count=df[288].value_counts()

#putting all beats of a type of arrhythmia into their respective variables
df_1=(df[df[288]==1]).sample(n = 79584, random_state = 42)
df_0=(df[df[288]==0]).sample(n = 11482, random_state = 100)
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
df_1_sample = (df_1_atrain.sample(n = 20000, random_state = 42))
df_0_upsample = resample(df_0_atrain, n_samples = 20000, random_state = 100)
df_2_upsample = resample(df_2_atrain, n_samples = 20000, random_state = 88)
df_3_upsample = resample(df_3_atrain, n_samples = 20000, random_state = 8)
df_4_upsample = resample(df_4_atrain, n_samples = 20000, random_state = 888)

#combining all the actual train samples
train_df = pd.concat([df_1_sample, df_0_upsample, df_2_upsample, df_3_upsample, df_4_upsample])


#looking at a beat in group 0
random = train_df.groupby(288,group_keys=False).apply(lambda train_df : train_df.sample(1))
plt.plot(random.iloc[0,:287])
plt.show()



'''

PATIENCE 8 AND BATCH SIZE 100 AND RELU TO SIGMOID

top is sigmoid and new with patience of middle

middle is relu? and diff with patience 20 and batch size 32

bottom is new patience 15 instead of 20, sigmoid, and diff

'''




#making a function that adds gaussian noise to all the waves
def add_gaussian_noise(wave):
    noise = np.random.normal(0,0.05,287)
    return (wave + noise)

target_train=train_df[288]
target_validation=validation_df[288]
y_train=to_categorical(target_train)
y_validation=to_categorical(target_validation)

X_train=train_df.iloc[:,:287].values
X_validation=validation_df.iloc[:,:287].values
for i in range(len(X_train)):
    X_train[i,:287] = add_gaussian_noise(X_train[i,:287])
X_train = X_train.reshape(len(X_train), X_train.shape[1],1)
X_validation = X_validation.reshape(len(X_validation), X_validation.shape[1],1)

def network(X_train,y_train,X_validation,y_validation):

    im_shape=(X_train.shape[1],1)
    inputs_cnn=Input(shape=im_shape, name='inputs_cnn')
    conv1_1=Convolution1D(32, 6, activation='sigmoid', input_shape=im_shape)(inputs_cnn)
    conv1_1=BatchNormalization()(conv1_1)



    conv2_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv1_1)
    conv2_1=BatchNormalization()(conv2_1)
    conv2_1=Dropout(0.6)(conv2_1)
    conv2_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv2_1)
    pool2=MaxPool1D(pool_size=2, strides=2, padding="same")(conv2_1)



    conv3_1=BatchNormalization()(pool2)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    conv3_1=Dropout(0.6)(conv3_1)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv3_1)
    pool3=MaxPool1D(pool_size=2, strides=2, padding="same")(conv3_1)

    conv3_1=BatchNormalization()(pool3)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    conv3_1=Dropout(0.6)(conv3_1)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv3_1)
    pool3=MaxPool1D(pool_size=2, strides=2, padding="same")(conv3_1)

    conv3_1=BatchNormalization()(pool3)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    conv3_1=Dropout(0.6)(conv3_1)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv3_1)
    pool3=MaxPool1D(pool_size=2, strides=2, padding="same")(conv3_1)

    conv3_1=BatchNormalization()(pool2)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    conv3_1=Dropout(0.6)(conv3_1)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv3_1)
    pool3=MaxPool1D(pool_size=2, strides=2, padding="same")(conv3_1)

    conv3_1=BatchNormalization()(pool3)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    conv3_1=Dropout(0.6)(conv3_1)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv3_1)
    pool3=MaxPool1D(pool_size=2, strides=2, padding="same")(conv3_1)

    conv3_1=BatchNormalization()(pool3)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    conv3_1=Dropout(0.6)(conv3_1)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv3_1)
    pool3=MaxPool1D(pool_size=2, strides=2, padding="same")(conv3_1)

    conv3_1=BatchNormalization()(pool2)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    conv3_1=Dropout(0.6)(conv3_1)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv3_1)
    pool3=MaxPool1D(pool_size=2, strides=2, padding="same")(conv3_1)

    conv3_1=BatchNormalization()(pool3)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    conv3_1=Dropout(0.6)(conv3_1)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv3_1)
    pool3=MaxPool1D(pool_size=2, strides=2, padding="same")(conv3_1)

    conv3_1=BatchNormalization()(pool3)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    conv3_1=Dropout(0.6)(conv3_1)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv3_1)
    pool3=MaxPool1D(pool_size=2, strides=2, padding="same")(conv3_1)

    conv3_1=BatchNormalization()(pool2)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    conv3_1=Dropout(0.6)(conv3_1)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv3_1)
    pool3=MaxPool1D(pool_size=2, strides=2, padding="same")(conv3_1)

    conv3_1=BatchNormalization()(pool3)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    conv3_1=Dropout(0.6)(conv3_1)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv3_1)
    pool3=MaxPool1D(pool_size=2, strides=2, padding="same")(conv3_1)

    conv3_1=BatchNormalization()(pool3)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    conv3_1=Dropout(0.6)(conv3_1)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv3_1)
    pool3=MaxPool1D(pool_size=2, strides=2, padding="same")(conv3_1)

    conv3_1=BatchNormalization()(pool2)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    conv3_1=Dropout(0.6)(conv3_1)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv3_1)
    pool3=MaxPool1D(pool_size=2, strides=2, padding="same")(conv3_1)

    conv3_1=BatchNormalization()(pool3)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    conv3_1=Dropout(0.6)(conv3_1)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv3_1)
    pool3=MaxPool1D(pool_size=2, strides=2, padding="same")(conv3_1)

    conv3_1=BatchNormalization()(pool3)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    conv3_1=Dropout(0.6)(conv3_1)
    conv3_1=Convolution1D(32, 3, activation='sigmoid', input_shape=im_shape)(conv3_1)
    pool3=MaxPool1D(pool_size=2, strides=2, padding="same")(conv3_1)



    conv3_1=BatchNormalization()(pool3)
    flatten=Flatten()(pool3)
    #dense_end = Dense(32, activation='relu')(flatten)
    main_output = Dense(5, activation='softmax', name='main_output')(flatten)
    
    





    model = Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    history=model.fit(X_train, y_train,epochs=100,callbacks=callbacks, batch_size=32,validation_data=(X_validation,y_validation))
    model.load_weights('best_model.h5')
    return(model,history)

def evaluate_model(history,X_validation,y_validation,model):
    scores = model.evaluate((X_validation),y_validation, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
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
from keras.layers.normalization import BatchNormalization
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

model,history=network(X_train,y_train,X_validation,y_validation)
evaluate_model(history,X_validation,y_validation,model)
y_pred=model.predict(X_validation)