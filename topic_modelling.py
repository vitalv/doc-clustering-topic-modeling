import csv
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB




#read docword.txt into a document x word matrix

#data = np.loadtxt(docword.kos.txt,skiprows=3)

file_name = 'docword.kos.txt'

vocab_file_name = 'vocab.kos.txt'


def read_vocab(vocab_file_name):
	return [w.strip() for w in open(vocab_file_name)]

vocab = read_vocab(vocab_file_name)

def read_docword(file_name):

	file_handle = open(file_name)
	reader = csv.reader(file_handle, delimiter=' ')
	D = int(next(reader)[0])
	W = int(next(reader)[0])
	N = int(next(reader)[0])

	#create DxW numpy matrix
	m = np.empty(shape=[D,W], dtype='int32')
	#instead of creating a sparse matrix and then fill it up, create a numpy matrix
	#and then later convert it to csr -> SparseEfficiencyWarning
	#m = sparse.csr_matrix( (D,W), dtype='int8')

	for row in reader:
		D_i = int(row[0])-1
		W_i = int(row[1])-1
		count = int(row[2])
		m[D_i, W_i] = count

	m = sparse.csr_matrix(m)

	return m


docword = read_docword(file_name)

#term frequency inverse document frequency
#it's a more reliable metric than plain frequency bc it normalizes frequency across documents
#very common (and semantically meaningless) words like articles ('the', 'a', 'an' ...), prepositions, etc... are in this way given less weight
tfidf_transformer = TfidfTransformer()
docword_tfidf = tfidf_transformer.fit_transform(docword)

#split the tfidf into test and train (1/3 and 2/3):
ix_1third = docword_tfidf.shape[0]/3
docword_tfidf_test = docword_tfidf[0:ix_1third,]
docword_tfidf_train = docword_tfidf[ix_1third+1:,]


nbayes_classifier = MultinomialNB().fit(docword_tfidf_test, docword_tfidf_train)