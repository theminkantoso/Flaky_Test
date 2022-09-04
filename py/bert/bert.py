import os
import pickle
import pandas as pd
import flast_bert

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Dropout, LSTM, SpatialDropout1D, Embedding
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
    outFile = "bert.csv"
    os.makedirs(outDir, exist_ok=True)
    with open(os.path.join(outDir, outFile), "w") as fo:
        fo.write("dataset,flakyTrain,nonFlakyTrain,flakyTest,nonFlakyTest,loss,accuracy,precision,recall,TP,TN,FP,FN\n")
    
    v0 = time.perf_counter()
    # dataPoints, labels = flast_bert.retrieveDataSpecialLabels(projectBasePath, projectList)

    # X = np.array(dataPoints)

    numSplit = 1
    testSetSize = 0.2
    kf = StratifiedShuffleSplit(n_splits=numSplit, test_size=testSetSize)

    # dataLabelsList = []
    # for i in range(len(labels)):
    #     if(labels[i] % 2 == 0):
    #         dataLabelsList.append(0)
    #     else:
    #         dataLabelsList.append(1)
    # dataLabelsList = np.array(dataLabelsList)
    # # Y = pd.get_dummies(dataLabelsList).values
    # Y = dataLabelsList
    # # print(Y)

    preci = []
    recal = []
   
    for projectName in projectList:
        print(projectName.upper(), "FLAST")
        dataPoints, labels, count_flaky, count_nonflaky = flast_bert.getDataPointsBERT(projectBasePath, projectName)
        X = np.array(dataPoints)
        Y = np.array(labels)

        weight_for_0 = 1.0 / count_nonflaky
        weight_for_1 = 1.0 / count_flaky
        class_weight = {0: weight_for_0, 1: weight_for_1}

        for (trnIdx, tstIdx) in kf.split(dataPoints, labels):
            X_train, X_test = X[trnIdx], X[tstIdx]
            Y_train, Y_test = Y[trnIdx], Y[tstIdx]

            flaky_train = 0
            nonflaky_train = 0
            flaky_test = 0
            nonflaky_test = 0
            for i in range(len(Y_train)):
                if(Y_train[i] == 1):
                    flaky_train = flaky_train + 1
                else:
                    nonflaky_train = nonflaky_train + 1
            
            for i in range(len(Y_test)):
                # print(np.equal(Y_train[i], np.array([0,1])))
                if(Y_test[i] == 1):
                    flaky_test = flaky_test + 1
                else:
                    nonflaky_test = nonflaky_test + 1

            bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
            bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

            METRICS = [
              tf.keras.metrics.BinaryAccuracy(name='accuracy'),
              tf.keras.metrics.Precision(name='precision'),
              tf.keras.metrics.Recall(name='recall'),
              tf.keras.metrics.TruePositives(),
              tf.keras.metrics.TrueNegatives(),
              tf.keras.metrics.FalsePositives(), 
              tf.keras.metrics.FalseNegatives()
            ]

            # Bert layers
            text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
            preprocessed_text = bert_preprocess(text_input)
            outputs = bert_encoder(preprocessed_text)

            # Neural network layers
            l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
            l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

            # Use inputs and outputs to construct a final model
            model = tf.keras.Model(inputs=[text_input], outputs = [l])

            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=METRICS)

            model.fit(X_train, Y_train, epochs=5,class_weight=class_weight)
            accr = model.evaluate(X_test,Y_test)
            print('Test set\n Loss: {:0.3f}\n Accuracy: {:0.3f}\n Precision: {:0.3f}\n Recall: {:0.3f}'.format(accr[0],accr[1],accr[2],accr[3]))
            print('TP: {:0.3f}\n TN: {:0.3f}\n FP: {:0.3f}\n FN: {:0.3f}'.format(accr[4],accr[5],accr[6],accr[7]))

            preci.append(accr[2])  
            recal.append(accr[3])
            with open(os.path.join(outDir, outFile), "a") as fo:
                fo.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(projectName,flaky_train,nonflaky_train,flaky_test,nonflaky_test,accr[0],accr[1],accr[2],accr[3],accr[4],accr[5],accr[6],accr[7]))

    with open(os.path.join(outDir, outFile), "a") as fo:
        fo.write("bert_no_mix,-,-,-,-,-,-,{},{},-,-,-,-".format(sum(preci)/len(preci),sum(recal)/len(recal)))

        

