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

import flast2

def flastRF(outDir, projectBasePath, projectList, kf, dim, eps):
    v0 = time.perf_counter()
    dataPointsFlaky = []
    dataPointsNonFlaky = []
    dataPointsFlaky, dataPointsNonFlaky = flast2.retrieveAllData(projectBasePath, projectList)
    dataPoints = dataPointsFlaky + dataPointsNonFlaky
    Z = flast2.flastVectorization(dataPoints, dim=dim, eps=eps)
    dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
    dataLabelsList = np.array([1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky))
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
        trainData, testData = dataPointsList[trnIdx], dataPointsList[tstIdx]
        trainLabels, testLabels = dataLabelsList[trnIdx], dataLabelsList[tstIdx]
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

        # prepare the data in the right format for kNN
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
        (precision, recall) = flast2.computeResults(testLabels, predictLabels)
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

def flastSVM(outDir, projectBasePath, projectList, kf, dim, eps):
    v0 = time.perf_counter()
    dataPointsFlaky = []
    dataPointsNonFlaky = []
    dataPointsFlaky, dataPointsNonFlaky = flast2.retrieveAllData(projectBasePath, projectList)
    dataPoints = dataPointsFlaky + dataPointsNonFlaky
    Z = flast2.flastVectorization(dataPoints, dim=dim, eps=eps)
    dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
    dataLabelsList = np.array([1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky))
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
        trainData, testData = dataPointsList[trnIdx], dataPointsList[tstIdx]
        trainLabels, testLabels = dataLabelsList[trnIdx], dataLabelsList[tstIdx]
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

        # prepare the data in the right format for kNN
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
        (precision, recall) = flast2.computeResults(testLabels, predictLabels)
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

def flastNB(outDir, projectBasePath, projectList, kf, dim, eps):
    v0 = time.perf_counter()
    dataPointsFlaky = []
    dataPointsNonFlaky = []
    dataPointsFlaky, dataPointsNonFlaky = flast2.retrieveAllData(projectBasePath, projectList)
    dataPoints = dataPointsFlaky + dataPointsNonFlaky
    Z = flast2.flastVectorization(dataPoints, dim=dim, eps=eps)
    dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
    dataLabelsList = np.array([1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky))
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
        trainData, testData = dataPointsList[trnIdx], dataPointsList[tstIdx]
        trainLabels, testLabels = dataLabelsList[trnIdx], dataLabelsList[tstIdx]
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

        # prepare the data in the right format for kNN
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
        (precision, recall) = flast2.computeResults(testLabels, predictLabels)
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
      
    outDir = "results/"
    outFile = "flast_rf_noEs_100.csv"
    os.makedirs(outDir, exist_ok=True)
    with open(os.path.join(outDir, outFile), "w") as fo:
        fo.write("flakyTrain,nonFlakyTrain,flakyTest,nonFlakyTest,precision,recall,storage,preparationTime,predictionTime\n")

    dim = 0  # number of dimensions (0: JL with error eps)
    eps = 0.3  # JL eps

    # time preperation
    numSplit = 30
    testSetSize = 0.2
    kf = StratifiedShuffleSplit(n_splits=numSplit, test_size=testSetSize)

    (flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred) = flastRF(outDir, projectBasePath, projectList, kf, dim, eps)
    with open(os.path.join(outDir, outFile), "a") as fo:
            fo.write("{},{},{},{},{},{},{},{},{}\n".format(flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred))

