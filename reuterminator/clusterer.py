from reuterssgmlparser import *
from preprocessorhelper import *
from sklearn import cluster
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import time
import sys
from sklearn.feature_extraction.text import *
from sklearn import manifold
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import matplotlib.cm as cmx
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
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


    def read_feature_files(self, feature_type):
        self.X_with_topics = PreprocessorHelper.load_csr_matrix(feature_type+"_with_topics_vect.npz")
        self.X_without_topics = PreprocessorHelper.load_csr_matrix(feature_type+"_without_topics_vect.npz")
        #self.X_array = self.X.toarray()
        self.topic_per_doc = PreprocessorHelper.load_file("doc_topics.pickle")


    def cluster_agglomerative(self, n, with_topic):
        #self.n_clusters = n
        if with_topic is True:
            X = self.X_with_topics
        else:
            X = self.X_without_topics

        z = hierarchy.single(pdist(X.toarray(), 'cityblock'))
        self.clusterizer_labels = hierarchy.fcluster(z,n, criterion = 'maxclust')
        cluster_set = set(self.clusterizer_labels)
        self.n_clusters = len(cluster_set) #- (1 if -1 in self.clusterizer_labels else 0)
        if with_topic is True:
            for i in cluster_set: #initialize a dict with cluster no as key to a list
                self.clusters[i] = []
            #adding the actual topics of the cluster to this dict. Mapping of label -> [list of actual topics for the documents clustered in this label]
            for i in range(0,len(self.topic_per_doc)):
                self.clusters[self.clusterizer_labels[i]].append(self.topic_per_doc[i])
            self.generate_piechart_pdf("hierarchical")
        else:
            self.calculate_and_print_silhouette(self.X_without_topics, self.clusterizer_labels, "DBSCAN")



    def cluster_kmeans(self, n, with_topic):
        self.n_clusters = n
        self.clusters = {}
        for i in range(0,n): #initialize a dict with cluster no as key to a list
            self.clusters[i] = []
        clusterizer = cluster.KMeans(n_clusters=n)
        print self.X_with_topics.shape[0]

        if with_topic is True:
            X = self.X_with_topics
        else:
            X = self.X_without_topics

        #clustering
        clusterizer.fit(X)
        self.clusterizer_labels = clusterizer.labels_
        self.clusterizer_cluster_centers = clusterizer.cluster_centers_

        if with_topic is True:
            #adding the actual topics of the cluster to this dict. Mapping of label -> [list of actual topics for the documents clustered in this label]
            for i in range(0,len(self.topic_per_doc)):
                self.clusters[self.clusterizer_labels[i]].append(self.topic_per_doc[i])
            self.generate_piechart_pdf("kmeans")

        else:
            self.calculate_and_print_silhouette(self.X_without_topics, self.clusterizer_labels, "DBSCAN")

        #self.plot_cluster()
        #print self.clusters

    def cluster_DBSCAN(self, with_topic):

        self.clusters = {}
        clusterizer = cluster.DBSCAN(eps=3, min_samples=10, algorithm="auto", metric="manhattan")


        if with_topic is True:
            X = self.X_with_topics
        else:
            X = self.X_without_topics

        clusterizer.fit(X.toarray())
        self.clusterizer_labels = clusterizer.labels_
        cluster_set = set(self.clusterizer_labels)
        self.n_clusters = len(cluster_set) #- (1 if -1 in self.clusterizer_labels else 0)

        if with_topic is True:
            for i in cluster_set: #initialize a dict with cluster no as key to a list
                self.clusters[i] = []
            #adding the actual topics of the cluster to this dict. Mapping of label -> [list of actual topics for the documents clustered in this label]
            for i in range(0,len(self.topic_per_doc)):
                self.clusters[self.clusterizer_labels[i]].append(self.topic_per_doc[i])
            self.generate_piechart_pdf("DBSCAN")

        else:
            self.calculate_and_print_silhouette(self.X_without_topics, self.clusterizer_labels, "DBSCAN")



    def calculate_and_print_silhouette(self, X, labels, algorithm):
        print "Silhouette score for kmeans is "+str(metrics.silhouette_score(X, labels, metric="euclidean"))


    #generates the pdf file
    def generate_piechart_pdf(self, algorithm):
        pdf = PdfPages('piecharts_'+algorithm+'.pdf')
        for cluster_label, doc_labels in self.clusters.iteritems():
            self.plot_single_piechart(cluster_label, doc_labels, pdf)
        pdf.close()

    #calling this for each cluster (pie chart)
    def plot_single_piechart(self, cluster_no, labels, pdf):
       cluster_label_dict = {}

       for label in labels:
           if label not in cluster_label_dict:
               cluster_label_dict[label] = 1
           else:
               cluster_label_dict[label] += 1
       distinct_labels = []
       label_counts = []
       colors = []
       cmap = self.get_cmap(30)
       i = 0
       for label,value in cluster_label_dict.iteritems():
           colors.append(cmap(i))
           distinct_labels.append(label)
           label_counts.append(value)
           i = i+1
       matplotlib.rcParams['font.size'] = 5

       plt.figure(figsize=(8,8))
       plt.pie(label_counts, explode=None, labels=distinct_labels, colors=colors,
       autopct='%1.1f%%', shadow=True, startangle=90)
       # Set aspect ratio to be equal so that pie is drawn as a circle.
       plt.axis('equal')
       plt.suptitle("Cluster no " + str(cluster_no) + "\nNumber of docs in cluster: " + str(len(labels)),fontsize=10)
       pdf.savefig()
       plt.close()





    #NOT using this. IGNORE
    def plot_cluster(self):
        self.scale_data()
        print "finished scale"
        cmap = self.get_cmap(100)
        colors = []
        for i in range(0, self.n_clusters):
            colors.append(cmap(i))
        fig = plt.figure(figsize=(1,1))
        #fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
        ax = fig.add_subplot(1, 2, 1)

        for k, col in zip(range(self.n_clusters), colors):
            my_members = self.k_means_labels == k
            cluster_center = self.k_means_cluster_centers[k]
            ax.plot(self.X_scaled[my_members, 0], self.X_scaled[my_members, 1], 'w',
                    markerfacecolor=col, marker='.')
            #fig.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            #        markeredgecolor='k', markersize=6)
        plt.show()
        #pylab.scatter(self.X_array[:,0], self.X_array[:,1], c=colors)
        #pylab.show()
        # mark centroids as (X)
        #pylab.scatter(res[:,0],res[:,1], marker='o', s = 500, linewidths=2, c='none')
        #pylab.scatter(res[:,0],res[:,1], marker='x', s = 500, linewidths=2)




    #not using this for now either
    def scale_data(self):
        print "in scale"
        mds = manifold.MDS(n_components=2,max_iter=200, dissimilarity="euclidean", n_jobs=2)
        self.X_scaled = mds.fit_transform(self.X)




    def get_cmap(self, N):
        #Returns a function that maps each index in 0, 1, ... N-1 to a distinctRGB color
        color_norm  = colors.Normalize(vmin=0, vmax=N-1)
        scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
        def map_index_to_rgb_color(index):
            return scalar_map.to_rgba(index)
        return map_index_to_rgb_color


def main():
    sys.setrecursionlimit(10000)
    clusterer = Clusterer()
    clusterer.read_feature_files("tfidf")
    start = time.clock()
    print len(set(clusterer.topic_per_doc))
    clusterer.cluster_kmeans(len(set(clusterer.topic_per_doc)), True)
    end = time.clock()
    print end - start, 'seconds to cluster kmeans with topics'

    start = time.clock()
    clusterer.cluster_kmeans(30, False)
    end = time.clock()
    print end - start, 'seconds to cluster kmeans without topics'

    start = time.clock()
    clusterer.cluster_DBSCAN(True)
    end = time.clock()
    print end - start, 'seconds to cluster DBSCAN with topics'

    start = time.clock()
    clusterer.cluster_DBSCAN(False)
    end = time.clock()
    print end - start, 'seconds to cluster DBSCAN without topics'

    start = time.clock()
    clusterer.cluster_agglomerative(10, True)
    end = time.clock()
    print end - start, 'seconds to cluster hierarchical with topics'

    start = time.clock()
    clusterer.cluster_agglomerative(10, False)
    end = time.clock()
    print end - start, 'seconds to cluster hierarchical without topics'


    #clusterer.plot_cluster()
    #clusterer.scale_data()
if __name__ == "__main__": main()
