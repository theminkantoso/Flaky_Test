import os
import pickle
import pandas as pd
import flast_lstm

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, SpatialDropout1D, Embedding
from keras.layers import CuDNNLSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

import time
import warnings

import numpy as np

from scipy import spatial

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric]) 

if __name__ == "__main__":
    projectBasePath = "dataset"
    projectList = [
        "achilles",
        "alluxio-tachyon",
        "ambari",
        "hadoop",
        "jackrabbit-oak",
        "jimfs",
        "ninja",
        "okhttp",
        "oozie",
        "oryx",
        "spring-boot",
        "togglz",
        "wro4j",
    ]

    projectIndex = {
       "achilles": 0,
        "alluxio-tachyon": 1,
        "ambari": 2,
        "hadoop": 3,
        "jackrabbit-oak": 4,
        "jimfs": 5,
        "ninja": 6,
        "okhttp": 7,
        "oozie": 8,
        "oryx": 9,
        "spring-boot": 10,
        "togglz": 11,
        "wro4j": 12
    }

    outDir = "results/"
    outFile = "lstm.csv"
    os.makedirs(outDir, exist_ok=True)
    with open(os.path.join(outDir, outFile), "w") as fo:
        fo.write("dataset,flakyTrain,nonFlakyTrain,flakyTest,nonFlakyTest,loss,accuracy,precision,recall,TP,TN,FP,FN\n")
    
    v0 = time.perf_counter()
    dataPoints, labels = flast_lstm.retrieveDataSpecialLabels(projectBasePath, projectList)
    # dataLabelsList = np.array(labels)
    # print(labels)

    numSplit = 1
    testSetSize = 0.2
    kf = StratifiedShuffleSplit(n_splits=numSplit, test_size=testSetSize)

    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 250
    # This is fixed.
    EMBEDDING_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
    tokenizer.fit_on_texts(dataPoints)
    X = tokenizer.texts_to_sequences(dataPoints)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)    
    word_index = tokenizer.word_index
    # print('Found %s unique tokens.' % len(word_index))

    dataLabelsList = []
    for i in range(len(labels)):
        if(labels[i] % 2 == 0):
            dataLabelsList.append(0)
        else:
            dataLabelsList.append(1)
    dataLabelsList = np.array(dataLabelsList)
    Y = pd.get_dummies(dataLabelsList).values
    # Y = dataLabelsList
    # print(Y)
   
    for projectName in projectList:
        print(projectName.upper(), "FLAST")

        for (trnIdx, tstIdx) in kf.split(X, labels):
            valid = []
            for i in range(len(tstIdx)):
                if(labels[tstIdx[i]] == (2 * projectIndex[projectName]) or labels[tstIdx[i]] == (2 * projectIndex[projectName] + 1)):
                    valid.append(tstIdx[i])
            valid = np.array(valid)
            # print(trnIdx)
            # print(valid)
            X_train, X_test = X[trnIdx], X[valid]
            Y_train, Y_test = Y[trnIdx], Y[valid]
            flaky_train = 0
            nonflaky_train = 0
            flaky_test = 0
            nonflaky_test = 0
            for i in range(len(Y_train)):
                # print(np.equal(Y_train[i], np.array([0,1])))
                if(np.array_equal(Y_train[i], np.array([0,1]))):
                    flaky_train = flaky_train + 1
                else:
                    nonflaky_train = nonflaky_train + 1
            
            for i in range(len(Y_test)):
                # print(np.equal(Y_train[i], np.array([0,1])))
                if(np.array_equal(Y_test[i], np.array([0,1]))):
                    flaky_test = flaky_test + 1
                else:
                    nonflaky_test = nonflaky_test + 1
            # print(X_train.shape)
            # print(Y_train.shape)
            # print(X_test.shape)
            # print(Y_test.shape)

            model = Sequential()
            model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
            model.add(SpatialDropout1D(0.2))
            model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(2, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.TruePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
        # print(model.summary())
        
            epochs = 5
            batch_size = 64

            history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
            accr = model.evaluate(X_test,Y_test)
            print(X_test.shape[0])
            # print('Test set\n Loss: {:0.3f}\n FN: {:0.3f}\n FP: {:0.3f}'.format(accr[0],accr[1],accr[2]))
            y_predicted = model.predict(X_test)
            # y_predicted.flatten()
            # y_predicted = y_predicted.flatten()
            # print(y_predicted.shape)
            y_predicted = np.where(y_predicted > 0.5, 1, 0)
            print(y_predicted)
            print(Y_test)

            with open(os.path.join(outDir, outFile), "a") as fo:
                fo.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(projectName,flaky_train,nonflaky_train,flaky_test,nonflaky_test,accr[0],accr[1],accr[2],accr[3],accr[4],accr[5],accr[6],accr[7]))


        

