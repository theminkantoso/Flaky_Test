import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection

def retrieveAllData(projectBasePath, projectList):
    flakies = []
    non_flakies = []
    for projectName in projectList:
        dataPointsFlaky, dataPointsNonFlaky = getDataPointsInfo(projectBasePath, projectName)
        # print("DEBUG FLAKY", len(dataPointsFlaky))
        # print("DEBUG NONFLAKY", len(dataPointsNonFlaky))
        flakies = flakies + dataPointsFlaky
        non_flakies = non_flakies + dataPointsNonFlaky
    return flakies, non_flakies

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