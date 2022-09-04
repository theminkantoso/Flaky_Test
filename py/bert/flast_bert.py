import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection

###############################################################################
# read data from file

def getDataPoints(path):
    dataPointsList = []
    for dataPointName in os.listdir(path):
        if dataPointName[0] == ".":
            continue
        with open(os.path.join(path, dataPointName), encoding="utf-8") as fileIn:
            dp = fileIn.read()
        dataPointsList.append(dp)
    # print(dataPointsList)
    return dataPointsList


def getDataPointsInfo(projectBasePath, projectName):
    # get list of tokenized test methods
    projectPath = os.path.join(projectBasePath, projectName)
    flakyPath = os.path.join(projectPath, "flakyMethods")
    nonFlakyPath = os.path.join(projectPath, "nonFlakyMethods")
    return getDataPoints(flakyPath), getDataPoints(nonFlakyPath)

def getDataPointsBERT(projectBasePath, projectName):
    dataPointsFlaky, dataPointsNonFlaky = getDataPointsInfo(projectBasePath, projectName)
    labels = [1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky)
    dataPoints = dataPointsFlaky + dataPointsNonFlaky
    return dataPoints, labels, len(dataPointsFlaky), len(dataPointsNonFlaky)

def retrieveDataSpecialLabels(projectBasePath, projectList):
    i = 0
    flakies = []
    non_flakies = []
    labels_flakies = []
    labels_nonflakies = []
    for projectName in projectList:
        dataPointsFlaky, dataPointsNonFlaky = getDataPointsInfo(projectBasePath, projectName)
        for k in range(len(dataPointsFlaky)):
            labels_flakies.append(i+1)
        for k in range(len(dataPointsNonFlaky)):
            labels_nonflakies.append(i)
        flakies = flakies + dataPointsFlaky
        non_flakies = non_flakies + dataPointsNonFlaky
        i = i + 2
    dataPoints = flakies + non_flakies
    labels = labels_flakies + labels_nonflakies
    return dataPoints, labels

def retrieveEntireDataset(projectBasePath, projectList):
    i = 0
    flakies = []
    non_flakies = []
    labels_flakies = []
    labels_nonflakies = []
    for projectName in projectList:
        dataPointsFlaky, dataPointsNonFlaky = getDataPointsInfo(projectBasePath, projectName)
        for k in range(len(dataPointsFlaky)):
            labels_flakies.append(i+1)
        for k in range(len(dataPointsNonFlaky)):
            labels_nonflakies.append(i)
        flakies = flakies + dataPointsFlaky
        non_flakies = non_flakies + dataPointsNonFlaky
        i = i + 2
    dataPoints = flakies + non_flakies
    labels = labels_flakies + labels_nonflakies
    return dataPoints, labels


###############################################################################
# FLAST

def flastVectorization(dataPoints, dim=0, eps=0.3):
    countVec = CountVectorizer()
    Z_full = countVec.fit_transform(dataPoints)
    if eps == 0:
        Z = Z_full
    else:
        if dim <= 0:
            dim = johnson_lindenstrauss_min_dim(Z_full.shape[0], eps=eps)
        srp = SparseRandomProjection(n_components=dim)
        Z = srp.fit_transform(Z_full)
    return Z