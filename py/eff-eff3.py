import os
import pickle
import time

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

import flast3


def flastKNN(outDir, dataPoints, labels, index, Z, kf, k, sigma, params, v_start):
    v0 = time.perf_counter()
    v0 = v0 - v_start
    # Z = flast3.flastVectorization(dataPoints, dim=dim, eps=eps)
    dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
    dataLabelsList = np.array(labels)
    v1 = time.perf_counter()
    vecTime = v1 - v0

    # storage
    kNN = (dataPointsList, dataLabelsList)
    pickleDumpKNN = os.path.join(outDir, "flast-k{}-sigma{}.pickle".format(k, sigma))
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

        # prepare the data in the right format for kNN
        nSamplesTrainData, nxTrain, nyTrain = trainData.shape
        trainData = trainData.reshape((nSamplesTrainData, nxTrain * nyTrain))
        nSamplesTestData, nxTest, nyTest = testData.shape
        testData = testData.reshape((nSamplesTestData, nxTest * nyTest))

        trainTime, testTime, predictLabels = flast3.flastClassification(trainData, trainLabels, testData, sigma, k, params)
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
    outDir = "results/"
    outFile = "eff-eff3.csv"
    os.makedirs(outDir, exist_ok=True)
    with open(os.path.join(outDir, outFile), "w") as fo:
        fo.write("dataset,flakyTrain,nonFlakyTrain,flakyTest,nonFlakyTest,k,sigma,precision,recall,storage,preparationTime,predictionTime\n")
    v0 = time.perf_counter()
    dataPoints, labels = flast3.retrieveDataSpecialLabels(projectBasePath, projectList)

    numSplit = 30
    testSetSize = 0.2
    kf = StratifiedShuffleSplit(n_splits=numSplit, test_size=testSetSize)

    # FLAST
    dim = 0  # number of dimensions (0: JL with error eps)
    eps = 0.3  # JL eps
    params = { "algorithm": "brute", "metric": "cosine", "weights": "distance" }

    Z = flast3.flastVectorization(dataPoints, dim=dim, eps=eps)
    
    for k in [3, 7]:
        for sigma in [0.5, 0.95]:
            for projectName in projectList:
                print(projectName.upper(), "FLAST", k, sigma)
                (flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred) = flastKNN(outDir, dataPoints, labels, projectIndex[projectName], Z, kf, k, sigma, params, v0)
                with open(os.path.join(outDir, outFile), "a") as fo:
                    fo.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(projectName, flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, k, sigma, avgP, avgR, storage, avgTPrep, avgTPred))

