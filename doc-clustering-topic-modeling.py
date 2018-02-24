#!/usr/bin/python

import csv
import numpy as np
import seaborn as sb 
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer




def read_vocab(vocab_file_name):
	return [w.strip() for w in open(vocab_file_name)]


vocab = read_vocab('vocab.kos.txt')



# read docword.txt into a document x word matrix



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

	return D,W,N,m



D,W,N,docword = read_docword('docword.kos.txt')


# tfidf, term frequency inverse document frequency
# is a more reliable metric than plain frequency bc it normalizes frequency across documents
# very common (and semantically meaningless) words like articles ('the', 'a', 'an' ...), prepositions, etc... are in this way given less weight


tfidf_transformer = TfidfTransformer()
docword_tfidf = tfidf_transformer.fit_transform(docword)






#KMeans --------------------------

k = 20
km = KMeans(algorithm='auto',
			copy_x=True,
			init='k-means++',
			max_iter=300,
			n_clusters=k,
			n_init=10,
			n_jobs=1,
			precompute_distances='auto',
			random_state=None,
			tol=0.0001,
			verbose=0)

%time km.fit(docword_tfidf)
clusters = km.labels_.tolist()

#sort cluster centers by proximity to centroid
k_centers = km.cluster_centers_ #Coordinates of cluster centers  [n_clusters, n_features]
order_centroids = k_centers.argsort()[:, ::-1] #argsort returns the indices that would sort an array

for c in range(k):
	print "Cluster %i: " % c + \
			','.join([vocab[i] for i in [ix for ix in order_centroids[c, :5]]])








#Hierarchical Clustering --------------------------

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial.distance as scdist


#Compute distance matrix . Cosine is a good metric
#Pairwise distances between observations in n-dimensional space.

#Option 1 is sklearn.metrics.pairwais cosine_similarity
dist = 1 - cosine_similarity(docword_tfidf)

#Option 2 is scipy.spatial.distance (can't take csr_matrix as input and is slower)
D = scdist.pdist(docword_tfidf.todense(), metric='cosine')
D = scdist.squareform(D)


#Then get linkage matrix

#Option 1 define the linkage_matrix using ward clustering pre-computed distances
linkage_matrix_ward = ward(dist) 

#Option 2 
linkage_matrix_complete = sch.linkage(D, method='complete')#, metric='cosine')


dendro_color_threshold = 0.7 #default
sch.dendrogram(linkage_matrix, orientation='left', color_threshold=dendro_color_threshold*max(linkage_matrix[:,2]))



fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="left");

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout
















#import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl



''' Try PCA or tSNE, MDS takes too long

from sklearn.manifold import MDS, TSNE

#MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
'''


from sklearn.manifold import TSNE

#tsne = TSNE(n_components=2, perplexity=30, learning_rate=1000, n_iter=1000) #learning_rate is also called epsilon

tsne_cos = TSNE(n_components=2, 
				perplexity=30, 
				learning_rate=1000, 
				n_iter=1000, 
				metric='cosine', 
				verbose=1)

tsne_cos.fit_transform(dist)



def scatter(x, colors, nclasses):
	palette = np.array(sb.color_palette("hls", nclasses )) # color palette with seaborn.
	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')
	sc = ax.scatter(x[:,0], x[:,1], linewidth=0, s=40, color=palette[colors.astype(np.int)])
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	#ax.axis('off')
	ax.axis('tight')
	txts = []
	for i in range(nclasses):
		# Position of each label.
		xtext, ytext = np.median(x[colors == i, :], axis=0)
		txt = ax.text(xtext, ytext, str(i), fontsize=18)
		#txt.set_path_effects([
		#	PathEffects.Stroke(linewidth=5, foreground="w"),
		#	PathEffects.Normal()])
		txts.append(txt)
	return f, ax, sc, txts







from sklearn.decomposition import PCA

# Create a PCA model.
pca_2 = PCA(2)
# Fit the PCA model on the numeric columns from earlier.
plot_columns = pca_2.fit_transform(dist)
# Make a scatter plot of each game, shaded according to cluster assignment.
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1])#, c=labels)
# Show the plot.
plt.show()


































ix_1third = docword_tfidf.shape[0]/3
docword_tfidf_test = docword_tfidf[0:ix_1third,]
docword_tfidf_train = docword_tfidf[ix_1third+1:,]




nbayes_classifier = MultinomialNB().fit(docword_tfidf_test, docword_tfidf_train)

