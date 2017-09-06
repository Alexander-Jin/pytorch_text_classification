import numpy
import pickle
import random
import jieba
import os
import shutil
import sys
import logging
import argparse

parser = argparse.ArgumentParser(description='write back to dataList')
# learning
parser.add_argument('-lr', type=float, default=0.0005, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 300]')
parser.add_argument('-batchSize', type=int, default=100, help='batch size for training [default: 100]')
parser.add_argument('-log-interval', type=int, default=100,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-valid-interval', type=int, default=500,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=2000, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=2.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-kernel-num', type=int, default=60, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default="2,3,4,5,6",
                    help='comma-separated kernel size to use for convolution')
# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-wvFile', type=str, default="data/wv.npy", help='wordvector file')
parser.add_argument('-vFile', type=str, default="data/vocab.txt", help='vocab file')
parser.add_argument('-trainRatio', type=float, default=0.8, help='ratio of training data')
parser.add_argument('-debug', action='store_true', default=False, help='print intermediate data')
parser.add_argument('-classFile', type=str, default="classes.txt", help="class file")
parser.add_argument('-dataFile', type=str, default="dataset.txt", help="data file")
parser.add_argument('-seed', type=float, default=15, help="random generator seed")
parser.add_argument('-trimLength', type=int, default=130, help="maximum length of each sentence")
parser.add_argument('-rebuild', action='store_true', default=False, help="whether rebuild data matrix and resplit data")
parser.add_argument('-embeddingDimension', type=int, default=64, help="dimension of embedding")
parser.add_argument('-resume', type=str, default=None, help="resume file path")
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

random.seed(args.seed)

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
fh = logging.FileHandler('run.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)

# Load class to index map
indexToClass = {}
with open(classFilePath) as classFile:
    classIndex = 0
    line = classFile.readline()
    while (len(line) > 0):
        indexToClass[classIndex] = line.strip()
        classIndex += 1
        line = classFile.readline()
args.numClass = len(indexToClass)

backFile = open("datasetBack.txt", "w")
backFile.write("description|department\n")
dataListFile = open("dataList.txt", "r")
line = dataListFile.readline()
while line:
    line = line.strip()
    sentence = "".join(line.split("|||")[0].split("|"))
    label = indexToClass[int(line.split("|||")[1])]
    backFile.write(sentence + "|" + label + "\n")
    line = dataListFile.readline()
backFile.close()
dataListFile.close() 
