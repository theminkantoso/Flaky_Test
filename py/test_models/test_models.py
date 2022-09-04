import os
import pickle
import time

import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

import flast3

def flastRF(outDir, dataPoints, labels, index, Z, kf, v_start):
    v0 = time.perf_counter()
    v0 = v0 - v_start
    dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
    dataLabelsList = np.array(labels)
    v1 = time.perf_counter()
    vecTime = v1 - v0

    # storage
    kNN = (dataPointsList, dataLabelsList)
    pickleDumpKNN = os.path.join(outDir, "FlastRF.pickle")
    with open(pickleDumpKNN, "wb") as pickleFile:
        pickle.dump(kNN, pickleFile)
    storage = os.path.getsize(pickleDumpKNN)
    os.remove(pickleDumpKNN)

    avgP, avgR = 0, 0
    avgTPrep, avgTPred = 0, 0
    avgFlakyTrain, avgNonFlakyTrain, avgFlakyTest, avgNonFlakyTest = 0, 0, 0, 0
    successFold, precisionFold = 0, 0
    for (trnIdx, tstIdx) in kf.split(dataPointsList, dataLabelsList):
        valid = []
        for i in range(len(tstIdx)):
            if(dataLabelsList[tstIdx[i]] == (2 * index) or dataLabelsList[tstIdx[i]] == (2 * index + 1)):
                valid.append(tstIdx[i])
        valid = np.array(valid)
        trainData, testData = dataPointsList[trnIdx], dataPointsList[valid]
        trainLabels_temp, testLabels_temp = dataLabelsList[trnIdx], dataLabelsList[valid]
        trainLabels = []
        testLabels = []
        for i in range(len(trainLabels_temp)):
            if(trainLabels_temp[i] % 2 == 0):
                trainLabels.append(0)
            else:
                trainLabels.append(1)
        
        for i in range(len(testLabels_temp)):
            if(testLabels_temp[i] % 2 == 0):
                testLabels.append(0)
            else:
                testLabels.append(1)
        trainLabels = np.array(trainLabels)
        testLabels = np.array(testLabels)

        if sum(trainLabels) == 0 or sum(testLabels) == 0:
            print("Skipping fold...")
            print(" Flaky Train Tests", sum(trainLabels))
            print(" Flaky Test Tests", sum(testLabels))
            continue

        successFold += 1
        avgFlakyTrain += sum(trainLabels)
        avgNonFlakyTrain += len(trainLabels) - sum(trainLabels)
        avgFlakyTest += sum(testLabels)
        avgNonFlakyTest += len(testLabels) - sum(testLabels)

        # prepare the data in the right format for RF
        nSamplesTrainData, nxTrain, nyTrain = trainData.shape
        trainData = trainData.reshape((nSamplesTrainData, nxTrain * nyTrain))
        nSamplesTestData, nxTest, nyTest = testData.shape
        testData = testData.reshape((nSamplesTestData, nxTest * nyTest))
       

         # training
        t0 = time.perf_counter()
        rf = RandomForestClassifier()
        rf.fit(trainData, trainLabels)
        t1 = time.perf_counter()
        trainTime = t1 - t0

        # testing
        p0 = time.perf_counter()
        predictLabels = rf.predict(testData)
        p1 = time.perf_counter()
        testTime = p1 - p0

        preparationTime = (vecTime * len(trainData) / len(dataPoints)) + trainTime
        predictionTime = (vecTime / len(dataPoints)) + (testTime / len(testData))
        (precision, recall) = flast3.computeResults(testLabels, predictLabels)
        print(precision, recall)
        if precision != "-":
            precisionFold += 1
            avgP += precision
        avgR += recall
        avgTPrep += preparationTime
        avgTPred += predictionTime

    if precisionFold == 0:
        avgP = "-"
    else:
        avgP /= precisionFold
    avgR /= successFold
    avgTPrep /= successFold
    avgTPred /= successFold
    avgFlakyTrain /= successFold
    avgNonFlakyTrain /= successFold
    avgFlakyTest /= successFold
    avgNonFlakyTest /= successFold

    return (avgFlakyTrain, avgNonFlakyTrain, avgFlakyTest, avgNonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred)

def flastSVM(outDir, dataPoints, labels, index, Z, kf, v_start):
    v0 = time.perf_counter()
    v0 = v0 - v_start
    dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
    dataLabelsList = np.array(labels)
    v1 = time.perf_counter()
    vecTime = v1 - v0

    # storage
    kNN = (dataPointsList, dataLabelsList)
    pickleDumpKNN = os.path.join(outDir, "FlastSVM.pickle")
    with open(pickleDumpKNN, "wb") as pickleFile:
        pickle.dump(kNN, pickleFile)
    storage = os.path.getsize(pickleDumpKNN)
    os.remove(pickleDumpKNN)

    avgP, avgR = 0, 0
    avgTPrep, avgTPred = 0, 0
    avgFlakyTrain, avgNonFlakyTrain, avgFlakyTest, avgNonFlakyTest = 0, 0, 0, 0
    successFold, precisionFold = 0, 0
    for (trnIdx, tstIdx) in kf.split(dataPointsList, dataLabelsList):
        valid = []
        for i in range(len(tstIdx)):
            if(dataLabelsList[tstIdx[i]] == (2 * index) or dataLabelsList[tstIdx[i]] == (2 * index + 1)):
                valid.append(tstIdx[i])
        valid = np.array(valid)
        trainData, testData = dataPointsList[trnIdx], dataPointsList[valid]
        trainLabels_temp, testLabels_temp = dataLabelsList[trnIdx], dataLabelsList[valid]
        trainLabels = []
        testLabels = []
        for i in range(len(trainLabels_temp)):
            if(trainLabels_temp[i] % 2 == 0):
                trainLabels.append(0)
            else:
                trainLabels.append(1)
        
        for i in range(len(testLabels_temp)):
            if(testLabels_temp[i] % 2 == 0):
                testLabels.append(0)
            else:
                testLabels.append(1)
        trainLabels = np.array(trainLabels)
        testLabels = np.array(testLabels)

        if sum(trainLabels) == 0 or sum(testLabels) == 0:
            print("Skipping fold...")
            print(" Flaky Train Tests", sum(trainLabels))
            print(" Flaky Test Tests", sum(testLabels))
            continue

        successFold += 1
        avgFlakyTrain += sum(trainLabels)
        avgNonFlakyTrain += len(trainLabels) - sum(trainLabels)
        avgFlakyTest += sum(testLabels)
        avgNonFlakyTest += len(testLabels) - sum(testLabels)

        # prepare the data in the right format for SVM
        nSamplesTrainData, nxTrain, nyTrain = trainData.shape
        trainData = trainData.reshape((nSamplesTrainData, nxTrain * nyTrain))
        nSamplesTestData, nxTest, nyTest = testData.shape
        testData = testData.reshape((nSamplesTestData, nxTest * nyTest))
       

         # training
        t0 = time.perf_counter()
        s_v_m = svm.SVC()
        s_v_m.fit(trainData, trainLabels)
        t1 = time.perf_counter()
        trainTime = t1 - t0

        # testing
        p0 = time.perf_counter()
        predictLabels = s_v_m.predict(testData)
        p1 = time.perf_counter()
        testTime = p1 - p0

        preparationTime = (vecTime * len(trainData) / len(dataPoints)) + trainTime
        predictionTime = (vecTime / len(dataPoints)) + (testTime / len(testData))
        (precision, recall) = flast3.computeResults(testLabels, predictLabels)
        print(precision, recall)
        if precision != "-":
            precisionFold += 1
            avgP += precision
        avgR += recall
        avgTPrep += preparationTime
        avgTPred += predictionTime

    if precisionFold == 0:
        avgP = "-"
    else:
        avgP /= precisionFold
    avgR /= successFold
    avgTPrep /= successFold
    avgTPred /= successFold
    avgFlakyTrain /= successFold
    avgNonFlakyTrain /= successFold
    avgFlakyTest /= successFold
    avgNonFlakyTest /= successFold

    return (avgFlakyTrain, avgNonFlakyTrain, avgFlakyTest, avgNonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred)

def flastNB(outDir, dataPoints, labels, index, Z, kf, v_start):
    v0 = time.perf_counter()
    v0 = v0 - v_start
    dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
    dataLabelsList = np.array(labels)
    v1 = time.perf_counter()
    vecTime = v1 - v0

    # storage
    kNN = (dataPointsList, dataLabelsList)
    pickleDumpKNN = os.path.join(outDir, "FlastNB.pickle")
    with open(pickleDumpKNN, "wb") as pickleFile:
        pickle.dump(kNN, pickleFile)
    storage = os.path.getsize(pickleDumpKNN)
    os.remove(pickleDumpKNN)

    avgP, avgR = 0, 0
    avgTPrep, avgTPred = 0, 0
    avgFlakyTrain, avgNonFlakyTrain, avgFlakyTest, avgNonFlakyTest = 0, 0, 0, 0
    successFold, precisionFold = 0, 0
    for (trnIdx, tstIdx) in kf.split(dataPointsList, dataLabelsList):
        valid = []
        for i in range(len(tstIdx)):
            if(dataLabelsList[tstIdx[i]] == (2 * index) or dataLabelsList[tstIdx[i]] == (2 * index + 1)):
                valid.append(tstIdx[i])
        valid = np.array(valid)
        trainData, testData = dataPointsList[trnIdx], dataPointsList[valid]
        trainLabels_temp, testLabels_temp = dataLabelsList[trnIdx], dataLabelsList[valid]
        trainLabels = []
        testLabels = []
        for i in range(len(trainLabels_temp)):
            if(trainLabels_temp[i] % 2 == 0):
                trainLabels.append(0)
            else:
                trainLabels.append(1)
        
        for i in range(len(testLabels_temp)):
            if(testLabels_temp[i] % 2 == 0):
                testLabels.append(0)
            else:
                testLabels.append(1)
        trainLabels = np.array(trainLabels)
        testLabels = np.array(testLabels)

        if sum(trainLabels) == 0 or sum(testLabels) == 0:
            print("Skipping fold...")
            print(" Flaky Train Tests", sum(trainLabels))
            print(" Flaky Test Tests", sum(testLabels))
            continue

        successFold += 1
        avgFlakyTrain += sum(trainLabels)
        avgNonFlakyTrain += len(trainLabels) - sum(trainLabels)
        avgFlakyTest += sum(testLabels)
        avgNonFlakyTest += len(testLabels) - sum(testLabels)

        # prepare the data in the right format for NB
        nSamplesTrainData, nxTrain, nyTrain = trainData.shape
        trainData = trainData.reshape((nSamplesTrainData, nxTrain * nyTrain))
        nSamplesTestData, nxTest, nyTest = testData.shape
        testData = testData.reshape((nSamplesTestData, nxTest * nyTest))
       

         # training
        t0 = time.perf_counter()
        gnb = GaussianNB()
        gnb.fit(trainData, trainLabels)
        t1 = time.perf_counter()
        trainTime = t1 - t0

        # testing
        p0 = time.perf_counter()
        predictLabels = gnb.predict(testData)
        p1 = time.perf_counter()
        testTime = p1 - p0

        preparationTime = (vecTime * len(trainData) / len(dataPoints)) + trainTime
        predictionTime = (vecTime / len(dataPoints)) + (testTime / len(testData))
        (precision, recall) = flast3.computeResults(testLabels, predictLabels)
        print(precision, recall)
        if precision != "-":
            precisionFold += 1
            avgP += precision
        avgR += recall
        avgTPrep += preparationTime
        avgTPred += predictionTime

    if precisionFold == 0:
        avgP = "-"
    else:
        avgP /= precisionFold
    avgR /= successFold
    avgTPrep /= successFold
    avgTPred /= successFold
    avgFlakyTrain /= successFold
    avgNonFlakyTrain /= successFold
    avgFlakyTest /= successFold
    avgNonFlakyTest /= successFold

    return (avgFlakyTrain, avgNonFlakyTrain, avgFlakyTest, avgNonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred)

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

    # projectList = [
    #     "achilles",
    #     "alluxio-tachyon",
    #     "ambari",
    # ]
    # projectIndex = {
    #     "achilles": 0,
    #     "alluxio-tachyon": 1,
    #     "ambari": 2
    # }
      
    outDir = "results/"
    outFile = "flast_rf.csv"
    os.makedirs(outDir, exist_ok=True)
    with open(os.path.join(outDir, outFile), "w") as fo:
        fo.write("dataset,flakyTrain,nonFlakyTrain,flakyTest,nonFlakyTest,precision,recall,storage,preparationTime,predictionTime\n")

    # time preperation
    v0 = time.perf_counter()
    dataPoints, labels = flast3.retrieveDataSpecialLabels(projectBasePath, projectList)

    # CV
    # RandomForest: 20
    # SVM: 4
    # NB: 30
    numSplit = 30
    testSize = 0.2
    kf = StratifiedShuffleSplit(n_splits=numSplit, test_size=testSize)
    dim = 0  # number of dimensions (0: JL with error eps)
    eps = 0.3  # JL eps
    Z = flast3.flastVectorization(dataPoints, dim=dim, eps=eps)
    preci = []
    recal = []

    for projectName in projectList:
        print(projectName.upper(), "FLAST")
        (flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred) = flastRF(outDir, dataPoints, labels, projectIndex[projectName], Z, kf, v0)
        with open(os.path.join(outDir, outFile), "a") as fo:
            fo.write("{},{},{},{},{},{},{},{},{},{}\n".format(projectName, flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred))
        if(avgP != "-"):
            preci.append(avgP)
        else:
            preci.append(0)    
        recal.append(avgR)

    with open(os.path.join(outDir, outFile), "a") as fo:
        fo.write("random forest 100 estimators,-,-,-,-,{},{},-,-,-\n".format(sum(preci)/len(preci),sum(recal)/len(recal)))

