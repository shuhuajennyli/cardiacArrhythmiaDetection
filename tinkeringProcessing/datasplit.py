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
df=pd.read_csv('/home/cst2020/ScienceFair/processedtopfive.csv',header=None)

#counting number of beats for each type of arrhythmia
df[288]=df[288].astype(int)
val_count=df[288].value_counts()

#putting all beats of a type of arrhythmia into their respective variables
df_2=(df[df[288]==2]).sample(n = 79584, random_state = 42)
df_0=(df[df[288]==0]).sample(n = 11482, random_state = 100)
df_1=(df[df[288]==1]).sample(n = 3287, random_state = 88)
df_4=(df[df[288]==4]).sample(n = 1779, random_state = 8)
df_3=(df[df[288]==3]).sample(n = 1381, random_state = 888)

#splitting the beats of each category into train and test (this makes sure none of the test beats have been used to train)
df_2_train = df_2.sample(frac = 0.8, random_state = 42)
df_2_test = df_2.drop(df_2_train.index)

df_0_train = df_0.sample(frac = 0.8, random_state = 100)
df_0_test = df_0.drop(df_0_train.index)

df_1_train = df_1.sample(frac = 0.8, random_state = 88)
df_1_test = df_1.drop(df_1_train.index)

df_4_train = df_4.sample(frac = 0.8, random_state = 8)
df_4_test = df_4.drop(df_4_train.index)

df_3_train = df_3.sample(frac = 0.8, random_state = 888)
df_3_test = df_3.drop(df_3_train.index)

#putting all the test into one dataframe
test_df = pd.concat([df_0_test, df_1_test, df_2_test, df_3_test, df_4_test])

#upsampling or sampling the training samples to avoid bias
df_2_sample = (df_2_train.sample(n = 20000, random_state = 42))
df_0_upsample = resample(df_0_train, n_samples = 20000, random_state = 100)
df_1_upsample = resample(df_1_train, n_samples = 20000, random_state = 88)
df_3_upsample = resample(df_3_train, n_samples = 20000, random_state = 8)
df_4_upsample = resample(df_4_train, n_samples = 20000, random_state = 888)

#putting all the sampled training into one dataframe
train_and_validation_df = pd.concat([df_2_sample, df_0_upsample, df_1_upsample, df_3_upsample, df_4_upsample])

#splitting train into actual train and validation
train_df = train_and_validation_df.sample(frac = 0.8, random_state = 9)
validation_df = train_and_validation_df.drop(train_df.index)


print(train_and_validation_df)
print(train_df)
print(validation_df)


train_df[288]=train_df[288].astype(int)
val_count=train_df[288].value_counts()
print(val_count)


#looking at a beat in group 0
random = train_df.groupby(288,group_keys=False).apply(lambda train_df : train_df.sample(1))
plt.plot(random.iloc[0,:287])
plt.show()












#STARTING HERE NOT MY WORK


def add_gaussian_noise(signal):
    noise=np.random.normal(0,0.05,287)
    return (signal + noise)

target_train=train_df[288]
target_test=test_df[288]
y_train=to_categorical(target_train)
y_test=to_categorical(target_test)

X_train=train_df.iloc[:,:287].values
X_test=test_df.iloc[:,:287].values
for i in range(len(X_train)):
    X_train[i,:287] = add_gaussian_noise(X_train[i,:287])
X_train = X_train.reshape(len(X_train), X_train.shape[1],1)
X_test = X_test.reshape(len(X_test), X_test.shape[1],1)

def network(X_train,y_train,X_test,y_test):

    im_shape=(X_train.shape[1],1)
    inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')
    conv1_1=Convolution1D(64, (6), activation='relu', input_shape=im_shape)(inputs_cnn)
    conv1_1=BatchNormalization()(conv1_1)
    pool1=MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
    conv2_1=Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool1)
    conv2_1=BatchNormalization()(conv2_1)
    pool2=MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv2_1)
    conv3_1=Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    pool3=MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv3_1)
    flatten=Flatten()(pool3)
    dense_end1 = Dense(64, activation='relu')(flatten)
    dense_end2 = Dense(32, activation='relu')(dense_end1)
    main_output = Dense(5, activation='softmax', name='main_output')(dense_end2)
    
    
    model = Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=8),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    history=model.fit(X_train, y_train,epochs=20,callbacks=callbacks, batch_size=100,validation_data=(X_test,y_test))
    model.load_weights('best_model.h5')
    return(model,history)

def evaluate_model(history,X_test,y_test,model):
    scores = model.evaluate((X_test),y_test, verbose=0)
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
    for element in y_test:
        y_true.append(np.argmax(element))
    prediction_proba=model.predict(X_test)
    prediction=np.argmax(prediction_proba,axis=1)
    cnf_matrix = confusion_matrix(y_true, prediction)


from keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout
from keras.layers import Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

model,history=network(X_train,y_train,X_test,y_test)
evaluate_model(history,X_test,y_test,model)
y_pred=model.predict(X_test)


'''
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
np.set_printoptions(precision=2)


plt.figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix, classes=['N', 'S', 'V', 'F', 'Q'],normalize=True,
                      title='Confusion matrix, with normalization')
plt.show()
'''