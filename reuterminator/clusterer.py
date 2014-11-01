from reuterssgmlparser import *
from preprocessorhelper import *
from sklearn import cluster
import time
from sklearn.feature_extraction.text import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pylab


class Clusterer:
    def __init__(self):
        #self.parser = parser
        self.docs = [] #list of the text in each doc
        self.doc_topic = [] #list of topic considered for each doc
        #self.data_set_directory = d



    def tokenize(self, text):
        tokenizer = RegexpTokenizer(r'[A-Za-z\-]{2,}')
        tokenized_body = tokenizer.tokenize(text)
        words_without_stopwords = PreprocessorHelper.get_words_after_stop_word_removal(tokenized_body)
        words_stemmed = PreprocessorHelper.get_stemmed_words(words_without_stopwords)
        return words_stemmed

    #added feature generator here itself. Didn't want to change preprocessor

    def read_feature_files(self, feature_type):
        self.X = PreprocessorHelper.load_csr_matrix(feature_type+"_vect.npz")
        self.X_array = self.X.toarray()
        self.topic_per_doc = PreprocessorHelper.load_file("doc_topics.pickle")

    def cluster_Kmeans(self, n):
        self.n_clusters = n
        self.clusters = {}
        for i in range(0,n): #initialize a dict with cluster no as key to a list
            self.clusters[i] = []
        k_means = cluster.KMeans(n_clusters=n)
        k_means.fit(self.X)
        #adding the actual topics of the cluster to this dict. Mapping of label -> [list of actual topics for the documents clustered in this label]
        for i in range(0,len(self.topic_per_doc)):
            self.clusters[k_means.labels_[i]].append(self.topic_per_doc[i])
        self.k_means_labels = k_means.labels_
        self.k_means_cluster_centers = k_means.cluster_centers_
        #self.plot_cluster()
        print self.clusters

    #under progress. doesn't work yet
    def plot_cluster(self):
        cmap = self.get_cmap(self.n_clusters)
        colors = []
        for i in range(0, self.n_clusters):
            colors.append(cmap(i))
        '''fig = plt.figure(figsize=(8, 3))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
        ax = fig.add_subplot(1, 3, 1)

        for k, col in zip(range(self.n_clusters), colors):
            my_members = self.k_means_labels == k
            cluster_center = self.k_means_cluster_centers[k]
            ax.plot(self.X_array[my_members, 0], self.X_array[my_members, 1], 'w',
                    markerfacecolor=col, marker='.')
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=6)
        plt.show()'''
        pylab.scatter(self.X_array[:,0], self.X_array[:,1], c=colors)
        pylab.show()
        # mark centroids as (X)
        #pylab.scatter(res[:,0],res[:,1], marker='o', s = 500, linewidths=2, c='none')
        #pylab.scatter(res[:,0],res[:,1], marker='x', s = 500, linewidths=2)


    def get_cmap(self, N):
        '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
        RGB color.'''
        color_norm  = colors.Normalize(vmin=0, vmax=N-1)
        scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
        def map_index_to_rgb_color(index):
            return scalar_map.to_rgba(index)
        return map_index_to_rgb_color


def main():
    clusterer = Clusterer()
    clusterer.read_feature_files("tfidf")
    start = time.clock()
    parsed_data = clusterer.cluster_Kmeans(72)
    end = time.clock()
    print end - start, 'seconds to cluster with kmeans'
    clusterer.plot_cluster()
if __name__ == "__main__": main()
