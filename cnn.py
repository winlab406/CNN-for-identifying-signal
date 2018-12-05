#######################################load data and label them#####################################################
signal_data_path = 'D:/Python35/Scripts/signal_classification/noise_signal_black/signal_noise_black/1'
black_data_path = 'D:/Python35/Scripts/signal_classification/noise_signal_black/signal_noise_black/-1'
noise_data_path = 'D:/Python35/Scripts/signal_classification/noise_signal_black/signal_noise_black/0'
import os
import json
import numpy as np

signal_dir_list = os.listdir(signal_data_path)
noise_dir_list = os.listdir(noise_data_path)
black_dir_list = os.listdir(black_data_path)
data = []
label = []
time_of_spe = 3
for i in signal_dir_list:
    signal_data_filepath = os.path.join(signal_data_path, i)
    signal_data_file = open(signal_data_filepath, 'r')
    signal_data_info = json.load(signal_data_file)
    data.append(signal_data_info['1'][:time_of_spe])
    label.append(1)
print(len(label))
for k in black_dir_list:
    black_data_filepath = os.path.join(black_data_path, k)
    black_data_file = open(black_data_filepath, 'r')
    black_data_info = json.load(black_data_file)
    data.append(black_data_info['1'][:time_of_spe])
    label.append(1)
for i in noise_dir_list:
    noise_data_filepath = os.path.join(noise_data_path, i)
    noise_data_file = open(noise_data_filepath, 'r')
    noise_data_info = json.load(noise_data_file)
    data.append(noise_data_info['1'][:time_of_spe])
    label.append(0)
print(np.array(data).shape)
print(np.array(label).shape)
####################################################split data#####################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,label, test_size=0.2, random_state=0)
print("hhhhhhhhhhhhhhhhhhhhhhhhhhhh")
print(np.array(X_train).shape)
##################################################################################################################
from keras import layers
from keras import models
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import to_categorical
dr = 0.2 # dropout rate (%)
###############################################preprocessing data###################################################
import matplotlib.pylab as plt
from sklearn import preprocessing
import matplotlib.pylab as pl
nom_X_train=[]
nom_X_test=[]
for i in X_train:
        j=preprocessing.scale(i,axis=1)
        nom_X_train.append(j)
for k in X_test:
        j=preprocessing.scale(k,axis=1)
        nom_X_test.append(j)


print(np.array(nom_X_train).shape)
##############################################################reshape traindata#######################################
nom_X_train=np.array(nom_X_train)
nom_X_test=np.array(nom_X_test)
X_train_nsamples, X_train_nx, X_train_ny =nom_X_train.shape
X_test_nsamples, X_test_nx, X_test_ny =nom_X_test.shape
print(nom_X_train.shape)
nom_X_train=nom_X_train.reshape(X_train_nsamples,X_train_nx,X_train_ny,1)
nom_X_test=nom_X_test.reshape(X_test_nsamples,X_test_nx,X_test_ny,1)
print(nom_X_train.shape)
'''
nom_X_train1 =nom_X_train.reshape((X_train_nsamples,X_train_nx*X_train_ny))
print("reshape of train_data"+",")
print(nom_X_train1.shape)
'''
###########################################################apply pca on trainning data################################
'''
from sklearn.decomposition import PCA
pca = PCA(n_components=100)
new_train=pca.fit_transform(nom_X_train1)
print("PCA done")
print(new_train.shape)

new_train=np.reshape(new_train,(X_train_nsamples,time_of_spe,-1))
'''
##############################################################reshape testdata#######################################
'''
nom_X_test=np.array(nom_X_test)
X_test_nsamples, X_test_nx, X_test_ny =nom_X_test.shape
print(nom_X_test.shape)
nom_X_test1 =nom_X_test.reshape((X_test_nsamples,X_test_nx*X_test_ny))
print("reshape of train_data"+",")
print(nom_X_test1.shape)
'''
#########################################################apply pca on testdata######################################
'''
new_test=pca.transform(nom_X_test1)
new_test=np.reshape(new_test,(X_test_nsamples,time_of_spe,-1))
print("pca on testdata done")
'''
##################### ############################train CNN###################################################################
from keras import layers
y_train=to_categorical(y_train)
model = models.Sequential()

model.add(ZeroPadding2D((0, 2)))
model.add(layers.Conv2D(64, (1, 3),input_shape=(time_of_spe,X_train_ny,1),border_mode='valid', activation="relu", name="conv1", init='glorot_uniform'))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2)))
model.add(layers.Conv2D(256, (3, 3), border_mode="valid", activation="relu", name="conv2", init='glorot_uniform'))
#model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2)))
model.add(layers.Conv2D(80, (1, 3), border_mode="valid", activation="relu", name="conv3", init='glorot_uniform'))
model.add(layers.Flatten())
model.add(layers.Dense(40, activation='relu', init='he_normal', name="dense1"))
model.add(Dropout(dr))

model.add(layers.Dense(2, init='he_normal', name="dense2",activation="sigmoid"))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(np.array(nom_X_train), y_train, epochs=30, batch_size=60)
model.save('cnn_model.h5')
#y_test=model.predict(nom_X_test)
######################################################test accuray#########################################
from sklearn.metrics import accuracy_score
y_predic=model.predict_classes(nom_X_test)
from sklearn.metrics import confusion_matrix
import seaborn as sns
mat = confusion_matrix(y_test,y_predic)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False
            )
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
y_test=to_categorical(y_test)
score=model.evaluate(np.array(nom_X_test),y_test)
print("test accuray:")
print(score)
plt.plot(model.loss)
plt.show()
y_predic=model.predict_classes(nom_X_test)
print(y_predic)

