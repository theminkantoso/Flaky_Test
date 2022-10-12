import os
import flast_bilstm

import tensorflow as tf

import time
from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from transformers import AutoTokenizer, AutoModel
import torch

model_codebert = AutoModel.from_pretrained("microsoft/codebert-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def tokenCodeBERT(inpu):
    code_tokens = tokenizer.tokenize(inpu)
    tokens_ids=tokenizer.convert_tokens_to_ids(code_tokens)
    # context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
    return tokens_ids

def XCodeBERT(X):
    out = []
    for i in range(len(X)):
        code_tokens = tokenizer.tokenize(X[i])
        tokens_ids=tokenizer.convert_tokens_to_ids(code_tokens)
        out.append(tokens_ids)
    # context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
    return out

def RNN():
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(max_words, 32, input_length=max_len)(inputs)
    layer = Bidirectional(LSTM(32))(layer)
    layer = Dense(128, name='FC1')(layer)
    layer = Activation('tanh')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(32, name='FC2')(layer)
    layer = Activation('relu')(layer)
    layer = Dense(1, name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


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

    outDir = "results/bilstm/codebert"
    outFile = "bilstm_entire.csv"
    os.makedirs(outDir, exist_ok=True)
    with open(os.path.join(outDir, outFile), "w") as fo:
        fo.write("flakyTrain,nonFlakyTrain,flakyTest,nonFlakyTest,loss,accuracy,precision,recall,TP,TN,FP,FN\n")

    v0 = time.perf_counter()
    dataPointsFlaky, dataPointsNonFlaky = flast_bilstm.retrieveAllData(projectBasePath, projectList)
    dataPoints = dataPointsFlaky + dataPointsNonFlaky
    dataLabelsList = np.array([1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky))

    X = dataPoints
    X = np.array(X)
    le = LabelEncoder()
    Y = le.fit_transform(dataLabelsList)
    Y = Y.reshape(-1, 1)

    numSplit = 1
    testSetSize = 0.2
    kf = StratifiedShuffleSplit(n_splits=numSplit, test_size=testSetSize)

    max_words = 50000
    max_len = 7000
    model = RNN()
    model.summary()

    class_weight = {
        0: 1.,
        1: 2.5, 
        }

    preci = []
    recal = []

    temp = []
    for i in range(len(X)):
        token = np.array(tokenCodeBERT(X[i]))
        token.resize(max_len)
        temp.append(token)

    X = np.array(temp, dtype=object)

    for (trnIdx, tstIdx) in kf.split(dataPoints, dataLabelsList):
        X_train, X_test = X[trnIdx], X[tstIdx]
        Y_train, Y_test = Y[trnIdx], Y[tstIdx]
        flaky_train = 0
        nonflaky_train = 0
        flaky_test = 0
        nonflaky_test = 0
        for i in range(len(Y_train)):
            if (Y_train[i] == 1):
                flaky_train = flaky_train + 1
            else:
                nonflaky_train = nonflaky_train + 1

        for i in range(len(Y_test)):
            if (Y_test[i] == 1):
                flaky_test = flaky_test + 1
            else:
                nonflaky_test = nonflaky_test + 1

    sequences_matrix = X_train

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                            tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
                            tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])

    sequences_matrix = np.asarray(sequences_matrix).astype(np.float32)

    model.fit(sequences_matrix, Y_train, batch_size=128, epochs=8,
              validation_split=0.2, class_weight=class_weight)

    test_sequences_matrix=X_test
    test_sequences_matrix = np.asarray(test_sequences_matrix).astype(np.float32)

    accr = model.evaluate(test_sequences_matrix, Y_test)
    print(
        'Test set\n Loss: {:0.3f}\nAccuracy: {:0.3f}\nPrecision: {:0.3f}\nRecall: {:0.3f}\nTP: {:0.3f}\nTN: {:0.3f}\nFP: {:0.3f}\nFN: {:0.3f}'.format(
            accr[0], accr[1], accr[2], accr[3], accr[4], accr[5], accr[6], accr[7]))
    preci.append(accr[2])  
    recal.append(accr[3])

    with open(os.path.join(outDir, outFile), "a") as fo:
        fo.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(flaky_train,nonflaky_train,flaky_test,nonflaky_test,accr[0],accr[1],accr[2],accr[3],accr[4],accr[5],accr[6],accr[7]))

    preci.append(accr[2])  
    recal.append(accr[3])

    with open(os.path.join(outDir, outFile), "a") as fo:
        fo.write("-,-,-,-,-,-,{},{},-,-,-,-".format(sum(preci)/len(preci),sum(recal)/len(recal)))
