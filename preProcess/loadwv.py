from gensim.models.keyedvectors import KeyedVectors
import numpy

word_vectors = KeyedVectors.load_word2vec_format('wv.bin', binary=True)
vocabFile = open("vocab.txt", "w")
embeddingDimension = len(word_vectors.word_vec("中文"))
embeddingNumber = len(word_vectors.vocab)
embedding = numpy.zeros((embeddingNumber, embeddingDimension))
count = 0
for key in word_vectors.vocab.keys():
    vocabFile.write(key + "," + str(count))
    vocabFile.write("\n")
    vector = word_vectors.word_vec(key)
    vector = vector / numpy.linalg.norm(vector)
    embedding[count, :] = vector
    count += 1
vocabFile.close()
numpy.save("wv.npy", embedding)