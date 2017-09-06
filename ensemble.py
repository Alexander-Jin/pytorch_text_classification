import numpy
import pickle
import random
import logging
import torch
import jieba
import os
import shutil
import sys
import argparse
import datetime
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='ensemble text classificer')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.003]')
parser.add_argument('-epochs', type=int, default=200, help='number of epochs for train [default: 100]')
parser.add_argument('-batchSize', type=int, default=192, help='batch size for training [default: 100]')
parser.add_argument('-log-interval', type=int, default=50,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-valid-interval', type=int, default=500,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=2000, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-wvFile', type=str, default="data/wv.npy", help='wordvector file')
parser.add_argument('-vFile', type=str, default="data/vocab.txt", help='vocab file')
parser.add_argument('-trainRatio', type=float, default=0.8, help='ratio of training data')
parser.add_argument('-debug', action='store_true', default=False, help='print intermediate data')
parser.add_argument('-classFile', type=str, default="data/classes.txt", help="class file")
parser.add_argument('-dataFile', type=str, default="data/data.txt", help="data file")
parser.add_argument('-seed', type=float, default=369, help="random generator seed")
parser.add_argument('-trimLength', type=int, default=80, help="maximum length of each sentence")
parser.add_argument('-rebuild', action='store_true', default=False, help="whether rebuild data matrix and resplit data")
parser.add_argument('-embeddingDimension', type=int, default=64, help="dimension of embedding")
parser.add_argument('-resume', type=str, default=None, help="resume file path")
parser.add_argument('-hiddenSize', type=int, default=128, help="hidden size")
parser.add_argument('-lstmLayerSize', type=int, default=1, help="lstm number of layer")
parser.add_argument('-clip', type=float, default=10.0, help="clip of optimizer gradient")
parser.add_argument('-retest', action='store_true', default=False, help="perform test")
args = parser.parse_args()

wordVectorFile = args.wvFile
vocabFile = args.vFile
debug = args.debug
trainRatio = args.trainRatio
batchSize = args.batchSize
classFilePath = args.classFile
dataFilePath = args.dataFile
sentenceLength = args.trimLength
rebuild = args.rebuild
embeddingDimension = args.embeddingDimension
hiddenSize = args.hiddenSize
lstmLayerSize = args.lstmLayerSize
retest = args.retest

args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
random.seed(args.seed)

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
fh = logging.FileHandler('run.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)
logFile = open("accuracy.txt", "w")

# Load class to index map
classToIndex = {}
with open(classFilePath) as classFile:
    classIndex = 0
    line = classFile.readline()
    while (len(line) > 0):
        classToIndex[line.strip()] = classIndex
        classIndex += 1
        line = classFile.readline()
args.numClass = len(classToIndex)

indexToClass = {}
for key, value in classToIndex.items():
    indexToClass[value] = key

class BiRNN_Mean(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN_Mean, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # 2 for bidirection
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1.weight.data.normal_(0.1, 0.5)
        self.fc1.bias.data.normal_(0.1, 0.5)
        self.fc2.weight.data.normal_(0.1, 0.3)
        self.fc2.bias.data.normal_(0.1, 0.3)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout1(x)

        # Set initial states
        h0 = autograd.Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))  # 2 for bidirection
        c0 = autograd.Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))
        h0 = h0.cuda()
        c0 = c0.cuda()
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of average of steps
        outMean = torch.mean(out, 1)

        #outMax = torch.max(out, 1)[0]

        #out = torch.cat((outMean, outMax), 1)

        #outLast = out[:, 20, :]

        out = self.fc1(outMean)

        out = self.bn(out)

        out = F.relu(out)

        out = self.dropout2(out)

        out = self.fc2(out)

        out = F.log_softmax(out)

        return out

class BiRNN_Max(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN_Max, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # 2 for bidirection
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1.weight.data.normal_(0.1, 0.5)
        self.fc1.bias.data.normal_(0.1, 0.5)
        self.fc2.weight.data.normal_(0.1, 0.3)
        self.fc2.bias.data.normal_(0.1, 0.3)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout1(x)
        h0 = autograd.Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))  # 2 for bidirection
        c0 = autograd.Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))
        h0 = h0.cuda()
        c0 = c0.cuda()
        out, _ = self.lstm(x, (h0, c0))
        outMax = torch.max(out, 1)[0]
        out = self.fc1(outMax)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.log_softmax(out)
        return out

class BiRNN_MeanMax(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN_MeanMax, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 4, hidden_size)  # 2 for bidirection
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1.weight.data.normal_(0.1, 0.5)
        self.fc1.bias.data.normal_(0.1, 0.5)
        self.fc2.weight.data.normal_(0.1, 0.3)
        self.fc2.bias.data.normal_(0.1, 0.3)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout1(x)
        h0 = autograd.Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))  # 2 for bidirection
        c0 = autograd.Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))
        h0 = h0.cuda()
        c0 = c0.cuda()
        out, _ = self.lstm(x, (h0, c0))
        outMean = torch.mean(out, 1)
        outMax = torch.max(out, 1)[0]
        out = torch.cat((outMean, outMax), 1)
        out = self.fc1(out)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.log_softmax(out)
        return out

class BiRNN_Last(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN_Last, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # 2 for bidirection
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1.weight.data.normal_(0.1, 0.5)
        self.fc1.bias.data.normal_(0.1, 0.5)
        self.fc2.weight.data.normal_(0.1, 0.3)
        self.fc2.bias.data.normal_(0.1, 0.3)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout1(x)
        h0 = autograd.Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))  # 2 for bidirection
        c0 = autograd.Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))
        h0 = h0.cuda()
        c0 = c0.cuda()
        out, _ = self.lstm(x, (h0, c0))
        outLast = out[:, 20, :]
        out = self.fc1(outLast)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.log_softmax(out)
        return out

def prediction(data, dclass, models, args):
    dataSize = len(data)
    prediction = numpy.zeros((dataSize, len(models) + 1), dtype=int)
    corrects = 0
    for index in range(0, dataSize):
        feature = autograd.Variable(data[index].view(1, data[0].size()[0], -1), volatile=True)
        t = torch.LongTensor([dclass[index]])
        target = autograd.Variable(t, volatile=True)
        feature = feature.cuda()
        target = target.cuda()
        for i in range(0, len(models)):
            model = models[i]
            logit = model(feature)
            prediction[index][i] = torch.max(logit, 1)[1].data.cpu().numpy()[0]
            corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    print(corrects / dataSize)
    return prediction

if retest:
    # Get class distribution
    dataListFile = open("dataProcessed/dataList.txt", "r")
    classDistribution = numpy.zeros(len(classToIndex))
    line = dataListFile.readline()
    while line:
        line = line.strip()
        label = int(line.split("|||")[1])
        classDistribution[label] += 1
        line = dataListFile.readline()
    dataListFile.close()

    # split data
    validFile = open("dataProcessed/validFile.txt", "w")
    testFile = open("dataProcessed/testFile.txt", "w")
    trainingDistribution = classDistribution * (trainRatio)
    validationDistribution = classDistribution * (trainRatio + (1 - trainRatio) / 2.0)
    dataListFile = open("dataProcessed/dataList.txt", "r")
    line = dataListFile.readline()
    while line:
        line = line.strip()
        label = int(line.split("|||")[1])
        featureString = line.split("|||")[0]
        if (classDistribution[label] >= validationDistribution[label]):
            classDistribution[label] -= 1
            validFile.write(featureString + "|||" + indexToClass[label] + "\n")
        elif (classDistribution[label] >= trainingDistribution[label]):
            classDistribution[label] -= 1
            testFile.write(featureString + "|||" + indexToClass[label] + "\n")
        line = dataListFile.readline()
    validFile.close()
    testFile.close()

    logger.info("Data Splitted")

    # Load word vectors and word to index map
    wordVector = numpy.load(wordVectorFile)
    WordToIndex = {}
    with open(vocabFile) as vocabFile:
        line = vocabFile.readline()
        while (len(line) > 0):
            pair = line.split(",")
            WordToIndex[pair[0].strip()] = int(pair[1])
            line = vocabFile.readline()
    vocabFile.close()
    logger.info("Loaded word vectors")

    TFIDFwordToIndex = pickle.load(open("dataProcessed/wordToIndex.pickle", "rb"))
    # Embed word vectors
    assert (embeddingDimension == wordVector.shape[1])
    classDistribution = numpy.zeros(len(classToIndex))
    validData = []
    validClass = []
    validList = []
    with open("dataProcessed/validFile.txt", "r") as dataListFile:
        data = dataListFile.readline()
        while data:
            data = data.strip()
            dataMatrix = numpy.zeros((sentenceLength, embeddingDimension))
            dataList = []
            row = 0
            label = int(classToIndex[data.split("|||")[1]])
            for feature in data.split("|||")[0].split("|"):
                if (row >= sentenceLength): break
                if (feature.strip() in WordToIndex):
                    normalized = wordVector[WordToIndex[feature.strip()]]
                    normalized -= numpy.mean(normalized)
                    normalized = normalized / numpy.linalg.norm(normalized)
                    dataMatrix[row] = normalized
                else:
                    randomGenerated = numpy.random.rand(embeddingDimension)
                    randomGenerated -= numpy.mean(randomGenerated)
                    randomGenerated = randomGenerated / numpy.linalg.norm(randomGenerated)
                    dataMatrix[row] = randomGenerated
                if (feature in TFIDFwordToIndex): dataList.append(TFIDFwordToIndex[feature])
                row += 1
            validData.append(dataMatrix)
            validClass.append(label)
            validList.append(dataList)
            data = dataListFile.readline()
    dataListFile.close()
    logger.info("Valid Word vectors embedded")
    logger.info("Valid data size %d" % (len(validData)))
    assert(len(validData) == len(validClass))
    numpy.save("dataProcessed/ensembleValidData.npy", validData)
    numpy.save("dataProcessed/ensembleValidClass.npy", validClass)
    numpy.save("dataProcessed/ensembleValidList.npy", validList)

    # Release memory
    validData = None
    validClass = None
    validList = None

    testData = []
    testClass = []
    testList = []
    with open("dataProcessed/testFile.txt", "r") as dataListFile:
        data = dataListFile.readline()
        while data:
            data = data.strip()
            dataMatrix = numpy.zeros((sentenceLength, embeddingDimension))
            dataList = []
            row = 0
            label = int(classToIndex[data.split("|||")[1]])
            for feature in data.split("|||")[0].split("|"):
                if (row >= sentenceLength): break
                if (feature.strip() in WordToIndex):
                    normalized = wordVector[WordToIndex[feature.strip()]]
                    normalized -= numpy.mean(normalized)
                    normalized = normalized / numpy.linalg.norm(normalized)
                    dataMatrix[row] = normalized
                else:
                    randomGenerated = numpy.random.rand(embeddingDimension)
                    randomGenerated -= numpy.mean(randomGenerated)
                    randomGenerated = randomGenerated / numpy.linalg.norm(randomGenerated)
                    dataMatrix[row] = randomGenerated
                if (feature in TFIDFwordToIndex): dataList.append(TFIDFwordToIndex[feature])
                row += 1
            testData.append(dataMatrix)
            testClass.append(label)
            testList.append(dataList)
            data = dataListFile.readline()
    dataListFile.close()
    logger.info("Test Word vectors embedded")
    logger.info("Test data size %d" % (len(testData)))
    assert(len(testData) == len(testClass))
    numpy.save("dataProcessed/ensembleTestData.npy", testData)
    numpy.save("dataProcessed/ensembleTestClass.npy", testClass)
    numpy.save("dataProcessed/ensembleTestList.npy", testList)

    # Release memory
    testData = None
    testClass = None
    testList = None
    wordVector = None
    wordToIndex = None
    TFIDFwordToIndex = None

    logger.info("Saved to npy files")

# Load validation batch
validData = torch.from_numpy(numpy.load("dataProcessed/ensembleValidData.npy")).float()
validClass = torch.from_numpy(numpy.load("dataProcessed/ensembleValidClass.npy"))
validList = numpy.load("dataProcessed/ensembleValidList.npy")
logger.info("Built validation tensors")
logger.info("Valid data size %d" % (len(validData)))

# Load testing batch
testData = torch.from_numpy(numpy.load("dataProcessed/ensembleTestData.npy")).float()
testClass = torch.from_numpy(numpy.load("dataProcessed/ensembleTestClass.npy"))
testList = numpy.load("dataProcessed/ensembleTestList.npy")
logger.info("Built testing tensors")
logger.info("Test data size %d" % (len(testData)))

# Load tf and df table
tfTable = numpy.load("dataProcessed/tfTable.npy")
dfTable = numpy.load("dataProcessed/dfTable.npy")
wordToIndex = pickle.load(open("dataProcessed/wordToIndex.pickle", "rb"))

# Initialize model
#checkpoint_max = torch.load("model_128_1_max")
# load gpu checkpoint to cpu
#checkpoint_mean = torch.load("model_128_1_mean", map_location=lambda storage, loc: storage)
checkpoint_mean = torch.load("model_128_1_mean")
#checkpoint_meanmax = torch.load("model_128_1_maxmean")
#checkpoint_last = torch.load("model_128_1_index20")

#model_max = BiRNN_Max(embeddingDimension, hiddenSize, lstmLayerSize, args.numClass)
model_mean = BiRNN_Mean(embeddingDimension, hiddenSize, lstmLayerSize, args.numClass)
#model_meanmax = BiRNN_MeanMax(embeddingDimension, hiddenSize, lstmLayerSize, args.numClass)
#model_last = BiRNN_Last(embeddingDimension, hiddenSize, lstmLayerSize, args.numClass)

#model_max = torch.nn.DataParallel(model_max, device_ids=[0, 1, 2]).cuda()
model_mean = torch.nn.DataParallel(model_mean, device_ids=[0, 1, 2]).cuda()
#model_meanmax = torch.nn.DataParallel(model_meanmax, device_ids=[0, 1, 2]).cuda()
#model_last = torch.nn.DataParallel(model_last, device_ids=[0, 1, 2]).cuda()

#model_max.load_state_dict(checkpoint_max['state_dict'])
''' 
#Load gpu model to cpu
dictd = {}
for key, value in checkpoint_mean['state_dict'].items():
    dictd[key[7:]] = value
model_mean.load_state_dict(dictd)
'''
model_mean.load_state_dict(checkpoint_mean['state_dict'])
#model_meanmax.load_state_dict(checkpoint_meanmax['state_dict'])
#model_last.load_state_dict(checkpoint_last['state_dict'])

#model_max = torch.nn.DataParallel(model_max, device_ids=[0, 1, 2]).cuda()
model_mean = torch.nn.DataParallel(model_mean, device_ids=[0, 1, 2]).cuda()
#model_meanmax = torch.nn.DataParallel(model_meanmax, device_ids=[0, 1, 2]).cuda()
#model_last = torch.nn.DataParallel(model_last, device_ids=[0, 1, 2]).cuda()

models = []

#models.append(model_max)

models.append(model_mean)
#models.append(model_meanmax)
#models.append(model_last)
validData = torch.from_numpy(numpy.load("dataProcessed/validationData.npy")).float()
validClass = torch.from_numpy(numpy.load("dataProcessed/validationClass.npy"))
logger.info("Built validation tensors")
validationNumBatch = len(validClass)

def evaluate(data, dataClass, trainingOrValidation, model, args):
    model.eval()
    numBatch = data.size()[0]
    dataSize = numBatch * data.size()[1]
    corrects = 0
    for batchNumber in range(0, numBatch):
        feature = autograd.Variable(data[batchNumber], volatile=True)
        target = autograd.Variable(dataClass[batchNumber], volatile=True)
        feature, target = feature.cuda(0), target.cuda(0)
        logit = model(feature)
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects / dataSize
    print('{} eval acc: {:.4f}%({}/{}) \n'.format(trainingOrValidation, accuracy, corrects, dataSize))
    logFile.write('\neval acc: {:.4f}%({}/{}) \n'.format(accuracy, corrects, dataSize))
    logFile.write("\n")
    return accuracy

evaluate(validData, validClass, "validation", model_mean, args)

# testPrediction = prediction(validData, validClass, models, args)
for i in range(0, len(validData)):
    tfidf = numpy.zeros(args.numClass, dtype=float)
    l = validList[i]
    for wordIndex in l:
        for j in range(0, args.numClass):
            if (dfTable[wordIndex] >= 0):
                tfidf[j] += tfTable[wordIndex][j] * dfTable[wordIndex]
    prediction = numpy.argmax(tfidf)
    testPrediction[i][len(models)] = prediction

corrects = 0
undecide = 0
for i in range(0, len(validData)):
    label = validClass[i]
    predictionList = testPrediction[i]
    prediction = 0
    count = {}
    for j in predictionList:
        if j in count:
            count[j] += 1
        else:
            count[j] = 1
    countMax = 0
    voted = []
    for key, value in count.items():
        if value > countMax:
            countMax = value
    for key, value in count.items():
        if value == countMax:
            voted.append(key)
    prediction = predictionList[0]
    if (len(voted) > 1): undecide += 1
    if (prediction == validClass[i]): corrects += 1
print(corrects/ len(validData))
print(undecide)

logf = open("vote.txt", "w")
f = open("dataProcessed/validFile.txt", "r")
for i in range(0, len(validData)):
    line = f.readline()
    logf.write(line + "\n")
    for j in testPrediction[i]:
        logf.write(indexToClass[j] + ", ")
    logf.write("\n ---------\n")
logf.close()
f.close()

