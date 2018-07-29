import numpy as np
import sklearn.cluster
import distance


with open('datasets/isl-kvennanofn.txt') as f:
    data = f.read()

words = data.split() #Replace this line
words = np.asarray(words) #So that indexing with a list will work
lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5,verbose=True)
affprop.fit(lev_similarity)

for cluster_id in np.unique(affprop.labels_):
    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    #print(" - *%s:* %s" % (exemplar, cluster_str))
    #print(exemplar, cluster_str)
    print(exemplar)