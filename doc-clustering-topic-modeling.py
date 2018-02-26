#!/usr/bin/python

import csv
#%matplotlib inline 
import numpy as np
import seaborn as sb
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scdist
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import silhouette_score, silhouette_samples
sb.set_style("whitegrid", {'axes.grid' : False})



def read_vocab(vocab_file_name):
    '''Reads vocabulary file into a list'''
	return [w.strip() for w in open(vocab_file_name)]




def read_docword(file_name):
	'''
	Reads docword txt file into a Document-Term Matrix (DTM)
	The full DTM will be too large to hold in memory if represented	as a dense matrix. Use Scipy sparse instead
	Matrix multiplication involving the sparse representation is rapid thanks to algorithms that avoid explicitly
	performing multiplications by 0 (nNMF or SVD for instance involve matrix multiplication)
	'''

	file_handle = open(file_name)
	reader = csv.reader(file_handle, delimiter=' ')
	D = int(next(reader)[0])
	W = int(next(reader)[0])
	N = int(next(reader)[0])

	#create numpy DTM (Document-Term Matrix)
	m = np.empty(shape=[D,W], dtype='uint8')
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


#DTM: Document-Term Matrix
D,W,N,DTM = read_DTM('data/docword.nips.txt')

vocab = read_vocab('data/vocab.nips.txt')


# tfidf, term frequency inverse document frequency
# is a more reliable metric than plain frequency bc it normalizes frequency across documents
# very common (and semantically meaningless) words like articles ('the', 'a', 'an' ...), prepositions, etc... are in this way given less weight


tfidf_transformer = TfidfTransformer()
DTM_tfidf = tfidf_transformer.fit_transform(DTM)






#Now comes the interesting part
'''
This is an unsupervised document clustering / topic extraction.
No previous knowledge on the number of topics there are in every corpus of documents.
A conventional approach involves an -optional- initial step of LSA (Latent Semantic Analysis) (TruncatedSVD)
for dimensionalty reduction followed by K-Means. 
The downside to this approach in this scenario is that it requires a predefined number of clusters, which is not available

There might not be an optimal number of clusters with complete separation, but there are ways to assess/approximate it.
The 'elbow' method consists of plotting a range of number of clusters on the x axis and the average within-cluster sum of squares
in the y axis (as a measure of within cluster similarity between its elements). Then an inflexion point would be
indicative of a good k

Another option to estimate an initial number of clusters consists of running a hierachical clustering and plot a dendrogram
Depending on the method and metric different results can be achieved. 

If a good candidate for k is found K-Means can be re-run using it as input. In addition, several K-Means runs are advised since
the algorithm might end up in a local optima. 



Another approach would be to use a different clustering algorithm not requiring a predefined number of clusters:
Means-shift, for instance 
.

'''










#KMeans --------------------------------------------------------------------------------------------------------
'''
separate samples in n groups of equal variance
first step chooses the initial centroids (k, the number of clusters)
After initialization, K-means consists of looping between the two other steps:
The first step assigns each sample to its nearest centroid.
The second step creates new centroids by taking the mean value of all of the samples assigned to each previous centroid
The inertia or within-cluster sum-of-squares is minimized
'''

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

%time km.fit(DTM_tfidf)
clusters = km.labels_#.tolist()

#sort cluster centers by proximity to centroid
k_centers = km.cluster_centers_ #Coordinates of cluster centers  [n_clusters, n_features]
order_centroids = k_centers.argsort()[:, ::-1] #argsort returns the indices that would sort an array

cluster_topics = {}
for c in range(k):
	topic = ','.join([vocab[i] for i in [ix for ix in order_centroids[c, :5]]])
	cluster_topics[c] = topic
	print "Cluster %i: " % c + topic



#--------------------------------------------------------------------------------------------------------



# Try to find  optimal number of clusters for k-means. "Elbow" method
k_range = range(2,20)
kms = [KMeans(n_clusters=k, init='k-means++').fit(DTM_tfidf) for k in k_range]
centroids = [X.cluster_centers_ for X in kms]
labels = [km.labels_ for km in kms]
#calculate Euclidean distance from each point to cluster center
k_euclid = [scdist.cdist(DTM_tfidf.todense(), c, 'euclidean') for c in centroids]
dist = [np.min(ke, axis=1) for ke in k_euclid]
#Total within cluster sum of squares
wcss = [sum(d**2) for d in dist]
#average wcss
avwcss = [(sum(d**2))/len(d) for d in dist]
#total sum of squares
tss = sum(scdist.pdist(DTM_tfidf.todense())**2)/DTM_tfidf.shape[0]
#between cluster sum of squares:
bss = tss - wcss
#plot average wcss vs number of clusters "Elbow plot": look for a point where the rate of decrease in wcss sharply shifts
plt.subplots(figsize=(18, 12)) # set size
plt.plot(k_range, avwcss, '-o')
plt.ylabel("average wcss")
plt.xlabel("k")




#Not very clear elbow? Check out the Silhouette scores

from sklearn.metrics import silhouette_score, silhouette_samples

silhouette_avg_scores = [silhouette_score(DTM_tfidf, l) for l in labels]
print silhouette_avg_scores


#--------------------------------------------------------------------------------------------------------
















# Mean Shift --------------------------------------------------------------------------------------------------------
'''
Mean shift has the advantage that it does not require a pre-defined number of clusters
Updates centroid candidates in each iteration so they become the mean of the points within a region of size determined by the paramater bandwidth
The mean shift vector is computed for each centroid and points towards a region of the maximum increase in the density of points
'''

from sklearn.cluster import MeanShift, estimate_bandwidth

#bandwidth dictates the size of the region to search through. (Also called attractive/gravitational interaction length) Can be set manually
#http://scikit-learn.org/stable/modules/generated/sklearn.cluster.estimate_bandwidth.html

bandwidth = estimate_bandwidth(DTM_tfidf.toarray(), quantile=0.5, n_samples=200) #default: quantile=0.3, n_samples= (all samples are used)

#quantile=0.5 means that the median of all pairwise distances is used
#but it takes a default value if bandwidth is not set

#set seeds to k-means centroids to try the kmeans+mean_shift combined approach from http://jamesxli.blogspot.se/2012/03/on-mean-shift-and-k-means-clustering.html
#ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#If bin_seeding=true, initial kernel locations are not locations of all points, but rather the location of the discretized version of points, 
#where points are binned onto a grid whose coarseness corresponds to the bandwidth. Setting this option to True will speed up the algorithm because fewer seeds will be initialized

ms = MeanShift(bandwidth=bandwidth)
#ms = MeanShift()

ms.fit(DTM_tfidf.toarray())
ms_clusters = ms.labels_
ms_cluster_centers = ms.cluster_centers_

ms_cluster_labels_unique = np.unique(ms_clusters)
n_clusters_ = len(ms_cluster_labels_unique)

cluster_idxs = {}
for cluster in range(n_clusters_):
	c = labels == cluster
	cluster_idxs[cluster] = np.where(c == True)




#--------------------------------------------------------------------------------------------------------




















#DBSCAN --------------------------------------------------------------------------------------------------------

from sklearn.cluster import DBSCAN
from sklearn import metrics

#these are all defaults
db = DBSCAN(eps=0.5, 
			min_samples=5, 
			metric='minkowski', 
			metric_params=None, 
			algorithm='auto', 
			leaf_size=30, 
			p=2)

db.fit(DTM_tfidf)


core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print("Silhouette Coefficient: %0.3f"  % metrics.silhouette_score(DTM_tfidf, labels))


#--------------------------------------------------------------------------------------------------------
















#Hierarchical Clustering --------------------------------------------------------------------------------

import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial.distance as scdist


#Compute distance matrix . Cosine is a good metric
#Pairwise distances between observations in n-dimensional space.

#Option 1 is sklearn.metrics.pairwise cosine_similarity
dist = 1 - cosine_similarity(DTM_tfidf)

#Option 2 is scipy.spatial.distance (can't take csr_matrix as input and is slower)
#D = scdist.pdist(DTM_tfidf.todense(), metric='cosine')
#Note converting DTM_tfidf to dense format here might be a bad idea, it requires much more memory
#D = scdist.squareform(D)


#Then get linkage matrix

#Option 1 define the linkage_matrix using ward clustering pre-computed distances
linkage_matrix_ward = sch.ward(dist) 

#Option 2 
linkage_matrix_complete = sch.linkage(D, method='complete')#, metric='cosine')



#And then plot the dendrogram

dendro_color_threshold = 0.7 #default: 0.7

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = sch.dendrogram(linkage_matrix_complete, orientation="left",color_threshold=dendro_color_threshold*max(linkage_matrix[:,2]));

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout
plt.show()




#--------------------------------------------------------------------------------------------------------










#TSNE --------------------------------------------------------------------------------------------------------

from sklearn.manifold import TSNE

tsne_cos = TSNE(n_components=2, 
				perplexity=30, 
				learning_rate=1000, 
				n_iter=1000, 
				metric='cosine', 
				verbose=1)

dist = 1 - cosine_similarity(DTM_tfidf)

tsne_cos_coords = tsne_cos.fit_transform(dist)

def scatter(x, colors, nclasses):
	palette = np.array(sb.color_palette("hls", nclasses )) # color palette with seaborn.
	f = plt.figure(figsize=(12, 12))
	ax = plt.subplot(aspect='equal')
	sc = ax.scatter(x[:,0], x[:,1], linewidth=0, s=40, color=palette[colors])
	#ax.axis('off')
	ax.axis('tight')
	txts = []
	for i in range(nclasses):
		# Position of each label.
		xtext, ytext = np.median(x[colors == i, :], axis=0)
		txt = ax.text(xtext, ytext, str(i), fontsize=18)
		txts.append(txt)
	return f, ax, sc, txts


#Plot KMeans clusters 

scatter(tsne_cos_coords, km.labels_, len(set(km.labels_)))



#--------------------------------------------------------------------------------------------------------























































