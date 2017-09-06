import numpy
import logging
import math
import os
import shutil
import argparse
import jieba
import random
import pickle
import sys

numClass = 40
trainRatio = 0.8

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
fh = logging.FileHandler('run.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)
wordToIndex = {}

if 1 == 1:
    trainingList = []
    validationList = []
    testingList = []
    trainingClassList = []
    validationClassList = []
    testingClassList = []
    # Get wordList
    wordCount = {}
    numWord = 0
    wordListFile = open("data/wordList.txt", "w")
    with open("dataProcessed/dataList.txt", "r") as dataListFile:
        data = dataListFile.readline()
        while data:
            data = data.strip()
            for feature in data.split("|||")[0].split("|"):
                if feature not in wordCount:
                    wordCount[feature] = 1
                else:
                    wordCount[feature] += 1
            data = dataListFile.readline()
    dataListFile.close()
    countWord = {}
    removeSet = set()
    for key, value in wordCount.items():
        if value > 2:
            numWord += 1
            wordListFile.write(key + "\n")
            if value in countWord:
                countWord[value].append(key)
            else:
                countWord[value] = [key]
        else:
            removeSet.add(key)
    wordListFile.close()
    print("before remove: " + str(len(wordCount)))
    for i in removeSet: wordCount.pop(i, None)
    print("after remove: " + str(len(wordCount)))
    index = 0
    for key, value in wordCount.items():
        wordToIndex[key] = index
        index += 1

    '''
    values = list(countWord.keys())
    values.sort()
    for i in values:
        print(i)
        print(countWord[i])
        print("-----")
    '''
    # Get tf-idf lists
    classDistribution = numpy.zeros(numClass)
    matrixList = []
    labelList = []
    with open("dataProcessed/dataList.txt", "r") as dataListFile:
        data = dataListFile.readline().strip()
        while data:
            if data:
                data = data.strip()
                indexList = []
                label = int(data.split("|||")[1])
                classDistribution[label] += 1
                for feature in data.split("|||")[0].split("|"):
                    if (feature in wordToIndex):
                        indexList.append(wordToIndex[feature])
                indexList.sort()
                if (len(indexList) > 0):
                    matrixList.append(indexList)
                    labelList.append(label)
                data = dataListFile.readline()
    dataListFile.close()
    logger.info("word index lists generated")

    # Split training, validation, and testing data
    trainingDistribution = classDistribution * (trainRatio)
    validationDistribution = classDistribution * (trainRatio + (1 - trainRatio) / 2.0)
    batchCount = 0
    for index in range(0, len(matrixList)):
        label = labelList[index]
        if (classDistribution[label] >= validationDistribution[label]):
            testingList.append(matrixList[index])
            testingClassList.append(labelList[index])
            classDistribution[label] -= 1
        elif (classDistribution[label] >= trainingDistribution[label]):
            validationList.append(matrixList[index])
            validationClassList.append(labelList[index])
            classDistribution[label] -= 1
        else:
            trainingList.append(matrixList[index])
            trainingClassList.append(labelList[index])
    logger.info("Data Splitted")

    # Release memory
    matrixList = []
    labelList = []
    dataBatch = None
    labelBatch = None

    logger.info("Training data size %d" % len(trainingList))
    logger.info("Validation data size %d" % len(validationList))
    logger.info("Tesing data size %d" % len(testingList))
    assert (len(trainingList) == len(trainingClassList))
    assert (len(validationList) == len(validationClassList))
    assert (len(testingList) == len(testingClassList))

    # Serialize splits
    numpy.save("dataProcessed/trainingBOWListData.npy", trainingList)
    numpy.save("dataProcessed/trainingBOWListClass.npy", trainingClassList)
    trainingList = []
    trainingClassList = []

    numpy.save("dataProcessed/validationBOWListData.npy", validationList)
    numpy.save("dataProcessed/validationBOWListClass.npy", validationClassList)
    validationList = []
    validationClassList = []

    numpy.save("dataProcessed/testingBOWListData.npy", testingList)
    numpy.save("dataProcessed/testingBOWListClass.npy", testingClassList)
    testingList = []
    testingClassList = []
    logger.info("Saved to npy files")

# Load training batch
trainingData = numpy.load("dataProcessed/trainingBOWListData.npy")
trainingClass = numpy.load("dataProcessed/trainingBOWListClass.npy")
logger.info("Built training list")
trainingSize = len(trainingData)

# Load validation tensors
validData = numpy.load("dataProcessed/validationBOWListData.npy")
validClass = numpy.load("dataProcessed/validationBOWListClass.npy")
logger.info("Built validation list")
validationSize = len(validData)

# Load testing tensors
testData = numpy.load("dataProcessed/testingBOWListData.npy")
testClass = numpy.load("dataProcessed/testingBOWListClass.npy")
logger.info("Built testing list")
testingSize = len(testData)

logger.info("Training data size %d" % trainingSize)
logger.info("Validation data size %d" % validationSize)
logger.info("Tesing data size %d" % testingSize)

wordListFile = open("data/wordList.txt", "r")
numWord = len(wordListFile.readlines())
wordListFile.close()

# Get tf table
tfTable = numpy.zeros((numWord, numClass), dtype=float)
for i in range(0, trainingSize):
    l = trainingData[i]
    label = trainingClass[i]
    for wordIndex in l:
        tfTable[wordIndex][label] += 1.0

print("get tf table")

# Get df table
dfTable = numpy.zeros(numWord, dtype=float)
for i in range(0, numWord):
    df = 0
    for j in range(0, numClass):
        if (tfTable[i][j] > 0): df += 1
    if (df > 0): dfTable[i] = math.log(numClass/df)
    else: dfTable[i] = -1

print("get df table")

# normalize tf table
maxTF = numpy.amax(tfTable, axis=0)
for i in range(0, numWord):
    for j in range(0, numClass):
        tfTable[i][j] = tfTable[i][j] / maxTF[j]

print("normalize tf table")

numpy.save("dataProcessed/dfTable.npy", dfTable)
numpy.save("dataProcessed/tfTable.npy", tfTable)
with open('dataProcessed/wordToIndex.pickle', 'wb') as handle:
    pickle.dump(wordToIndex, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Training
corrects = 0
for i in range(0, trainingSize):
    tfidf = numpy.zeros(numClass, dtype=float)
    trainingList = trainingData[i]
    label = trainingClass[i]
    for wordIndex in trainingList:
        for j in range(0, numClass):
            if (dfTable[wordIndex] >= 0):
                tfidf[j] += tfTable[wordIndex][j] * dfTable[wordIndex]
    prediction = numpy.argmax(tfidf)
    if prediction == label: corrects += 1
print('\rvalid acc: {:.4f}%({}/{})'.format(corrects / trainingSize, corrects, trainingSize))

# Validation and Testing
corrects = 0
for i in range(0, validationSize):
    tfidf = numpy.zeros(numClass, dtype=float)
    validList = validData[i]
    label = validClass[i]
    for wordIndex in validList:
        for j in range(0, numClass):
            if (dfTable[wordIndex] >= 0):
                tfidf[j] += tfTable[wordIndex][j] * dfTable[wordIndex]
    prediction = numpy.argmax(tfidf)
    if prediction == label: corrects += 1
print('\rvalid acc: {:.4f}%({}/{})'.format(corrects / validationSize, corrects, validationSize))

corrects = 0
for i in range(0, testingSize):
    tfidf = numpy.zeros(numClass, dtype=float)
    testingList = testData[i]
    label = testClass[i]
    for wordIndex in testingList:
        for j in range(0, numClass):
            if (dfTable[wordIndex] >= 0):
                tfidf[j] += tfTable[wordIndex][j] * dfTable[wordIndex]
    prediction = numpy.argmax(tfidf)
    if prediction == label: corrects += 1
print('\rtest acc: {:.4f}%({}/{})'.format(corrects / testingSize, corrects, testingSize))