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
        self.parsed_data = {}

        #Mapping of doc_id to tokenized body that is stemmed and has no stop words.
        #Creating a separate dict for this data as it is used often
        self.tokenized_body_cleaned_dict = {}


        self.dataset_body_word_dict = {}
        #dict of feature vector with tfidf feature
        self.topic_list = []
        self.places_list = []
        self.docs = [] #list of the text in each doc
        self.doc_topic = [] #list of topic considered for each doc
        #self.data_set_directory = d


    def get_parsed_data(self):
        print 'Retrieving parsed data from datasets'
        self.parser.parse_all_docs()
        self.parsed_data = self.parser.get_parsed_dataset()
        self.topic_list = self.parser.get_all_topics()
        self.topic_freq_dict = self.parser.all_topics_dict #dict with freq of each topic. Used to take most common topic
        self.places_list = self.parser.get_all_places()
        self.topic_wise_dict = {}

    ''' Method that removes stop words and stems tokens '''

    def tokenize(self, text):
        tokenizer = RegexpTokenizer(r'[A-Za-z\-]{2,}')
        tokenized_body = tokenizer.tokenize(text)
        words_without_stopwords = PreprocessorHelper.get_words_after_stop_word_removal(tokenized_body)
        words_stemmed = PreprocessorHelper.get_stemmed_words(words_without_stopwords)
        return words_stemmed

    #added feature generator here itself. Didn't want to change preprocessor
    def feature_generator(self):
        #for doc_id,doc_attributes in self.parsed_data.iteritems():
        #    self.parsed_data[doc_id]['body'] = PreprocessorHelper.convert_to_utf(doc_attributes['body'])
        for doc_id,doc_attributes in self.parsed_data.iteritems():
            #self.docs.append(doc_attributes['body'])
            if len(doc_attributes['topics']) > 0:
                max_freq = 0
                for topic in doc_attributes['topics']:  #taking the most common topic among list of topics for each doc
                    if self.topic_freq_dict[topic] > max_freq:
                        max_freq_topic = topic
                        max_freq = self.topic_freq_dict[topic]

                self.doc_topic.append(max_freq_topic)
                self.docs.append(doc_attributes['body'])
        #print self.docs
        #tfidf vectorizer. We can make similar functions like this
        vectorizer = TfidfVectorizer(min_df=20,decode_error="ignore", tokenizer=self.tokenize)
        self.X = vectorizer.fit_transform(self.docs)

        #print X

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
    '''start = time.clock()
    parsed_data = clusterer.get_parsed_data()
    end = time.clock()
    print end - start, 'seconds to parse all documents'
    start = time.clock()
    parsed_data = clusterer.feature_generator()
    end = time.clock()
    print end - start, 'seconds to generate tfidfs' '''
    start = time.clock()
    parsed_data = clusterer.cluster_Kmeans(72)
    end = time.clock()
    print end - start, 'seconds to cluster with kmeans'
    clusterer.plot_cluster()
if __name__ == "__main__": main()
