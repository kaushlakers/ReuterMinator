from reuterssgmlparser import *
from preprocessorhelper import *
from sklearn import cluster
import time
from sklearn.feature_extraction.text import *
from sklearn import manifold
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
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
        #print metrics.silhouette_score(self.X, self.k_means_labels, metric="euclidean")
        #self.generate_piechart_pdf()
        #self.plot_cluster()
        #print self.clusters

    #generates the pdf file
    def generate_piechart_pdf(self):
        pdf = PdfPages('piecharts.pdf')
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
        cmap = self.get_cmap(self.n_clusters)
        i = 0
        for label,value in cluster_label_dict.iteritems():
            colors.append(cmap(i))
            distinct_labels.append(label)
            label_counts.append(value)
            i = i+1

        plt.figure(figsize=(8,8))
        plt.pie(label_counts, explode=None, labels=distinct_labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')
        plt.title("Cluster no " + str(cluster_no))
        pdf.savefig()
        plt.close()





    #NOT using this. IGNORE
    def plot_cluster(self):
        self.scale_data()
        print "finished scale"
        cmap = self.get_cmap(self.n_clusters)
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
    clusterer = Clusterer()
    clusterer.read_feature_files("tfidf")
    start = time.clock()
    clusterer.cluster_Kmeans(10)
    end = time.clock()
    print end - start, 'seconds to cluster with kmeans'
    clusterer.generate_piechart_pdf()
    #clusterer.plot_cluster()
    #clusterer.scale_data()
if __name__ == "__main__": main()
