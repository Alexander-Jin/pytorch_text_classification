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

parser = argparse.ArgumentParser(description='LSTM text classificer')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.003]')
parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 100]')
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
parser.add_argument('-test', action='store_true', default=False, help="perform test")
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

trainingList = []
validationList = []
testingList = []
trainingClassList = []
validationClassList = []
testingClassList = []

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

def evaluate(data, dataClass, trainingOrValidation, model, args):
    model.eval()
    numBatch = data.size()[0]
    dataSize = numBatch * data.size()[1]
    corrects = 0
    for batchNumber in range(0, numBatch):
        feature = autograd.Variable(data[batchNumber], volatile=True)
        target = autograd.Variable(dataClass[batchNumber], volatile=True)
        feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects / dataSize
    print('{} eval acc: {:.4f}%({}/{}) \n'.format(trainingOrValidation, accuracy, corrects, dataSize))
    logFile.write('\neval acc: {:.4f}%({}/{}) \n'.format(accuracy, corrects, dataSize))
    logFile.write("\n")
    return accuracy

# Rebuild dataProcessed
if rebuild or not os.path.exists("dataProcessed"):
    '''
    if os.path.exists("dataProcessed"):
        shutil.rmtree("dataProcessed")
    os.makedirs("dataProcessed")

    # Load stop word
    stopSet = set()
    with open("data/stopWord.txt") as stopFile:
        stopList = []
        line = stopFile.readline()
        while (len(line) > 0):
            stopList.append(line.strip())
            line = stopFile.readline()
        stopSet = set(stopList)
    stopFile.close()
    logger.info("Loaded stop word")

    # Build dataList, each element is a sequence of tokens
    dataList = []
    dataFile = open(dataFilePath)
    dataFile.readline()
    line = dataFile.readline()
    while line:
        if len(line.strip()) > 0:
            pair = line.split("|")
            features = "|".join([x for x in list(jieba.cut(pair[0])) if x not in stopSet and len(x) > 0])
            dataList.append(features + "|||" + str(classToIndex[pair[1].strip()]))
        line = dataFile.readline()
    dataFile.close()
    logger.info("Loaded dataList")

    # Shuffle data and write to file
    random.shuffle(dataList)
    dataListFile = open("dataProcessed/dataList.txt", "w")
    for data in dataList:
        dataListFile.write(data)
        dataListFile.write("\n")
    dataListFile.close()
    logger.info("Shuffled data")

    # Release memory
    dataList = []
    stopSet = set()
    '''
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

    # Embed word vectors
    assert (embeddingDimension == wordVector.shape[1])
    classDistribution = numpy.zeros(len(classToIndex))
    matrixList = []
    labelList = []
    with open("dataProcessed/dataList.txt", "r") as dataListFile:
        data = dataListFile.readline()
        while data:
            if data:
                data = data.strip()
                dataMatrix = numpy.zeros((sentenceLength, embeddingDimension))
                row = 0
                label = int(data.split("|||")[1])
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
                    row += 1
                classDistribution[label] += 1
                matrixList.append(dataMatrix)
                labelList.append(label)
                data = dataListFile.readline()
    dataListFile.close()
    logger.info("Word vectors embedded")

    # Release memory
    wordVector = None

    # Split training, validation, and testing data
    trainingDistribution = classDistribution * (trainRatio)
    validationDistribution = classDistribution * (trainRatio + (1 - trainRatio) / 2.0)
    batchCount = 0
    dataBatch = numpy.zeros((batchSize, sentenceLength, embeddingDimension))
    labelBatch = numpy.zeros(batchSize,dtype=numpy.int)

    evalSize = 92
    evalBatchCount = 0
    evalBatch = numpy.zeros((evalSize, sentenceLength, embeddingDimension))
    evalLabelBatch = numpy.zeros(evalSize, dtype=numpy.int)
    for index in range(0, len(labelList)):
        label = labelList[index]
        if (classDistribution[label] >= validationDistribution[label]):
            evalBatch[evalBatchCount] = matrixList[index]
            evalLabelBatch[evalBatchCount] = labelList[index]
            evalBatchCount += 1
            classDistribution[label] -= 1
            if (evalBatchCount == evalSize):
                evalBatchCount = 0
                validationList.append(evalBatch)
                validationClassList.append(evalLabelBatch)
                evalBatch = numpy.zeros((evalSize, sentenceLength, embeddingDimension))
                evalLabelBatch = numpy.zeros(evalSize, dtype=numpy.int)
        elif (classDistribution[label] >= trainingDistribution[label]):
            evalBatch[evalBatchCount] = matrixList[index]
            evalLabelBatch[evalBatchCount] = labelList[index]
            evalBatchCount += 1
            classDistribution[label] -= 1
            if (evalBatchCount == evalSize):
                evalBatchCount = 0
                testingList.append(evalBatch)
                testingClassList.append(evalLabelBatch)
                evalBatch = numpy.zeros((evalSize, sentenceLength, embeddingDimension))
                evalLabelBatch = numpy.zeros(evalSize, dtype=numpy.int)
        else:
            dataBatch[batchCount] = matrixList[index]
            labelBatch[batchCount] = labelList[index]
            batchCount += 1
            if (batchCount == batchSize):
                batchCount = 0
                trainingList.append(dataBatch)
                trainingClassList.append(labelBatch)
                dataBatch = numpy.zeros((batchSize, sentenceLength, embeddingDimension))
                labelBatch = numpy.zeros(batchSize, dtype=numpy.int)
    logger.info("Data Splitted")

    # Release memory
    matrixList = []
    dataBatch = None
    labelBatch = None

    logger.info("Training number of batches %d with batch size %d" % (len(trainingList), batchSize))
    logger.info("Validation number of batches %d with batch size %d" % (len(validationList), evalSize))
    logger.info("Tesing number of batches %d with batch size %d" % (len(testingList), evalSize))
    assert(len(trainingList) == len(trainingClassList))
    assert(len(validationList) == len(validationClassList))
    assert(len(testingList) == len(testingClassList))

    # Serialize splits
    numpy.save("dataProcessed/trainingData.npy", trainingList)
    numpy.save("dataProcessed/trainingClass.npy", trainingClassList)
    trainingList = []
    trainingClassList = []

    numpy.save("dataProcessed/validationData.npy", validationList)
    numpy.save("dataProcessed/validationClass.npy", validationClassList)
    validationList = []
    validationClassList = []

    numpy.save("dataProcessed/testingData.npy", testingList)
    numpy.save("dataProcessed/testingClass.npy", testingClassList)
    testingList = []
    testingClassList = []
    logger.info("Saved to npy files")

if (args.test):
    # Load testing batch
    testData = torch.from_numpy(numpy.load("dataProcessed/testingData.npy")).float()
    testClass = torch.from_numpy(numpy.load("dataProcessed/testingClass.npy"))
    logger.info("Built testing tensors")
    testingNumBatch = len(testClass)
    logger.info("Tesing number of batches %d with batch size %d" % (testingNumBatch, testData.size()[1]))

    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model = BiRNN(embeddingDimension, hiddenSize, lstmLayerSize, args.numClass)
    model.load_state_dict(checkpoint['state_dict'])
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2]).cuda()
    evaluate(testData, testClass, "testing", model, args)
    sys.exit()

# Load training batch
trainingData = torch.from_numpy(numpy.load("dataProcessed/trainingData.npy")).float()
trainingClass = torch.from_numpy(numpy.load("dataProcessed/trainingClass.npy"))
logger.info("Built training batches")
trainingNumBatch = len(trainingData)

# Load validation batch
validData = torch.from_numpy(numpy.load("dataProcessed/validationData.npy")).float()
validClass = torch.from_numpy(numpy.load("dataProcessed/validationClass.npy"))
logger.info("Built validation tensors")
validationNumBatch = len(validClass)

# Load testing batch
testData = torch.from_numpy(numpy.load("dataProcessed/testingData.npy")).float()
testClass = torch.from_numpy(numpy.load("dataProcessed/testingClass.npy"))
logger.info("Built testing tensors")
testingNumBatch = len(testClass)

logger.info("Training number of batches %d with batch size %d" % (trainingNumBatch, batchSize))
logger.info("Validation number of batches %d with batch size %d" % (validationNumBatch, validData.size()[1]))
logger.info("Tesing number of batches %d with batch size %d" % (testingNumBatch, testData.size()[1]))

# Initialize model
model = BiRNN_Mean(embeddingDimension, hiddenSize, lstmLayerSize, args.numClass)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2]).cuda()

# Training
lrate = args.lr
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
# optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
tested = False

validationAccuracyList = numpy.zeros(20, dtype=float)
if args.resume and os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])

changedSpan = 0
for epoch in range(1, args.epochs + 1):
    if (tested): break
    corrects = 0
    print("epoch: " + str(epoch))
    for batchNumber in range(0, len(trainingClass)):
        model.train()
        feature = autograd.Variable(trainingData[batchNumber])
        target = autograd.Variable(trainingClass[batchNumber])
        batchSize = target.size()[0]
        feature, target = feature.cuda(), target.cuda()
        optimizer.zero_grad()
        logit = model(feature)

        loss = F.nll_loss(logit, target)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        changedSpan += 1
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

        if (batchNumber % args.log_interval == 0):
            sys.stdout.write('\rEpoch {}, Batch[{}] norm: {})'.format(epoch, batchNumber, str(norm)))

        if batchNumber == len(trainingClass) // 2 or batchNumber == len(trainingClass) - 1:
            validationAccuracy = evaluate(validData, validClass, "validation", model, args)
            if (changedSpan >= len(validationAccuracyList) and validationAccuracy < min(validationAccuracyList) - 0.0001):
                lrate = lrate / 3
                print("=> learning rate reduced to " + str(lrate))
                changedSpan = 0
                if (lrate < 0.0000001):
                    tested = True
                    break
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lrate
            else:
                for i in range(0, len(validationAccuracyList) - 1):
                    validationAccuracyList[i] = validationAccuracyList[i + 1]
                validationAccuracyList[len(validationAccuracyList) - 1] = validationAccuracy

        if batchNumber == len(trainingClass) - 1:
            dataSize = trainingClass.size()[0] * trainingClass.size()[1]
            accuracy = 100.0 * corrects / dataSize
            print('training eval acc: {:.4f}%({}/{}) \n'.format(accuracy, corrects, dataSize))
model.cpu()
torch.save({'state_dict': model.state_dict()}, "checkpoint00")
evaluate(testData, testClass, "testing", model, args)