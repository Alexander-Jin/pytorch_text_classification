from gensim.models.keyedvectors import KeyedVectors
import numpy

word_vectors = KeyedVectors.load_word2vec_format('wv.bin', binary=True)
vocabFile = open("dataset_vocab.txt", "w")
for key in word_vectors.vocab.keys():
    vocabFile.write(key)
    vector = word_vectors.word_vec(key)
    vector = vector / numpy.linalg.norm(vector)
    for i in vector:
    	vocabFile.write(" ")
    	vocabFile.write(str(i))
    vocabFile.write("\n")
vocabFile.close()