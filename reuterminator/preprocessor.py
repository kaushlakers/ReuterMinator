from reuterssgmlparser import *
from preprocessorhelper import *
#import nltk
import string
import json
from nltk import *
from nltk import corpus
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.collocations import *
#from nltk.book import *
import sys
import os
import os.path
import json
import time
import matplotlib
import math
import itertools

##
#   Each parsed_data key-value is of the form:
'''
{
    'doc_id' = #Refers to ID of a News Item
    {
        'title': <title>,
        'dateline':<dateline>,
        'body': <body>, #body gets deleted after cleaning
        'topics': <topics>,
        'places': <places>,
        'tokenized_body_cleaned': <tokenized_body_cleaned>
        'vector': <vector>
    }
}
'''
##


class Preprocessor:
    def __init__(self, parser):
        self.parser = parser
        self.parsed_data = {}

        #Mapping of doc_id to tokenized body that is stemmed and has no stop words.
        #Creating a separate dict for this data as it is used often
        self.tokenized_body_cleaned_dict = {}

        #
        self.dataset_body_word_dict = {}
        #dict of feature vector with tfidf feature
        self.tfidf_dict = {}
        #dict of feature vector with tf feature
        self.tf_dict = {}
        self.topic_list = []
        self.places_list = []
        self.bigram_dict = {}
        self.global_bigram_dict = {}
        #self.data_set_directory = d


    def get_parsed_data(self):
        print 'Retrieving parsed data from datasets'
        self.parser.parse_all_docs()
        self.parsed_data = self.parser.get_parsed_dataset()
        self.topic_list = self.parser.get_all_topics()
        self.places_list = self.parser.get_all_places()
        print sorted(self.places_list)
        self.topic_wise_dict = {topic:{} for topic in self.topic_list}
        print self.topic_wise_dict
        #return self.parsed_data

    ''' Method that removes stop words and stems tokens '''

    def clear_topic_dict(self):
        for topic,docs in self.topic_wise_dict.iteritems():
            self.topic_wise_dict[topic].clear()

    def clean_data(self):
        i=0
        for doc_id,doc_attributes in self.parsed_data.iteritems():
            body = doc_attributes['body']
            body = body.replace('-',' ')
            #tokens = nltk.word_tokenize(body.translate(None, string.punctuation))
            tokenizer = RegexpTokenizer(r'[A-Za-z\-]{2,}')
            tokenized_body = tokenizer.tokenize(body)
            words_without_stopwords = PreprocessorHelper.get_words_after_stop_word_removal(tokenized_body)
            words_stemmed = PreprocessorHelper.get_stemmed_words(words_without_stopwords)
            self.tokenized_body_cleaned_dict[doc_id] = words_stemmed

            #deleting body as we no longer need it as we have tokenized_cleaned_body
            del doc_attributes['body']
            i+=1
            if i%1000 == 0:
                print i, 'documents have been stemmed and cleansed of stop words!'



    def populate_tfidf_feature_vector_dictionary(self):
        #self.calculate_term_frequencies()
        self.calculate_term_and_document_frequencies()
        #self.calculate_tfidf()
        #self.populate_dictionary_with_class_labels(self.tfidf_dict)

    def populate_term_frequencies_feature_vector_dictionary(self):
        self.remove_low_frequency_words()
        #tf_dict now contains list of term frequencies for each document
        self.populate_dictionary_with_class_labels(self.tf_dict)

    def get_best_bigrams_from_doc(self ,doc_id):
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(self.tokenized_body_cleaned_dict[doc_id])
        #using PMI to calculate bigram weights
        bigram_scores = finder.score_ngrams(bigram_measures.raw_freq)

        ordered_bigrams = sorted(bigram_scores, key = lambda t: t[1], reverse = True)[10:610]

        bigram_dict = {}

        for bigram,pmi in ordered_bigrams:
            bigram_dict[bigram[0] + " " + bigram[1]] = 1
        return bigram_dict

    def populate_bigram_feature_vector(self):
        for doc_id, tokens in self.tokenized_body_cleaned_dict.iteritems():
            self.bigram_dict[doc_id] = {}
            self.bigram_dict[doc_id]['bigrams_pmi'] = {}
            self.bigram_dict[doc_id]['bigrams_pmi'] = self.get_best_bigrams_from_doc(doc_id)
        #print self.bigram_dict
        self.populate_dictionary_with_class_labels(self.bigram_dict)



    #wrote separate function to calculate term frequency data matrix
    def calculate_term_frequencies(self):
        vector_template = self.convert_to_utf(json.load(open('word_dict.json')))
        for doc_id,tokenized_body_cleaned_list in self.tokenized_body_cleaned_dict.iteritems():
            tf_data_matrix = {}
            tf_data_matrix = dict.fromkeys(vector_template,0)
            frequency_dist = FreqDist(tokenized_body_cleaned_list)
            self.tf_dict[doc_id] = {'term_frequencies':{}}

            for token in frequency_dist:
                self.tf_dict[doc_id]['term_frequencies'][token] = frequency_dist[token]
                if token in tf_data_matrix:
                    tf_data_matrix[token] = frequency_dist[token]
                self.tf_dict[doc_id]['term_frequencies'] = dict.copy(tf_data_matrix)



    def calculate_term_and_document_frequencies(self):
        #Hashset of documents already counted for that word
        #cleaned_word_list =
        #dataset_freq_dist = FreqDist(itertools.chain(*(self.tokenized_body_cleaned_dict.values())))


        self.documents_containing_token = {}
        for doc_id,tokenized_body_cleaned_list in self.tokenized_body_cleaned_dict.iteritems():
            frequency_dist = FreqDist(tokenized_body_cleaned_list)
            #fdist.plot()
            #emptying the tokenized_body_cleaned tokens and refilling only with tokens which have frequency > 1

            #Adding a mapping of tokens in a particular doc to their frequencies in that doc
            #Also adding a mapping of tokens to their frequency across all docs
            self.tf_dict[doc_id] = {'term_frequencies':{}}

            for token in frequency_dist:
                self.tf_dict[doc_id]['term_frequencies'][token] = frequency_dist[token]

                if token not in self.documents_containing_token:
                    self.documents_containing_token[token] = set()

                if doc_id in self.documents_containing_token[token]:
                    continue
                else:
                    self.documents_containing_token[token].add(doc_id)



    def remove_low_frequency_words(self):
        print 'Removing words with per-news-item low frequency'
        start = time.clock()

        for doc_id in self.tf_dict:
            for word in self.tf_dict[doc_id]:
                if self.tf_dict[doc_id][word] < 2:
                    del(self.tf_dict[doc_id][word])

        end = time.clock()
        print end - start, 'seconds to remove all low frequency words in all documents'



    '''Accepts a dictionary of doc_id-feature mappings and populates it with appropriate class labels from the parsed_data dictionary'''
    #TODO: Change this to accept a list of generic class labels
    def populate_dictionary_with_class_labels(self, feature_dict):
        for doc_id, feature in feature_dict.iteritems():
            for topic in self.parsed_data[doc_id]['topics']:
                all_places = {place:0 for place in self.places_list}
                updated_places = {}
                for place in self.parsed_data[doc_id]['places']:
                    if place in all_places:
                        updated_places[place] = 1
                all_places.update(updated_places)

                feature_dict[doc_id]['places'] = all_places
                self.topic_wise_dict[topic][doc_id] = feature_dict[doc_id]
            #feature_dict[doc_id]['topics'] = self.parsed_data[doc_id]['topics']
            #feature_dict[doc_id]['places'] = self.parsed_data[doc_id]['places']

        #for topic in self.topic_list:

        return feature_dict

    def calculate_tfidf(self):
        i=0
        NUMBER_OF_DOCUMENTS = len(self.tf_dict)
        for doc_id in self.tf_dict:
            self.tfidf_dict[doc_id] = {'tfidf':{}}
            for token, token_tf in self.tf_dict[doc_id]['term_frequencies'].iteritems():
                idf = math.log(NUMBER_OF_DOCUMENTS/len(self.documents_containing_token[token]))
                self.tfidf_dict[doc_id]['tfidf'][token] = token_tf*idf

            i+=1
            if i%1000 == 0:
                print i, ' documents have been checked for words!'
            # self.word_dict = OrderedDict(sorted(self.tfidf_dict[doc_id]['tfidf'].items(), key=lambda t: t[1], reverse = True))


    def convert_to_utf(self, input):
        if isinstance(input, dict):
            return {self.convert_to_utf(key): self.convert_to_utf(value) for key, value in input.iteritems()}
        elif isinstance(input, list):
            return [self.convert_to_utf(element) for element in input]
        elif isinstance(input, unicode):
            return input.encode('utf-8')
        else:
            return input



def main():
    preprocessor = Preprocessor(ReutersSGMLParser())

    start = time.clock()
    parsed_data = preprocessor.get_parsed_data()
    end = time.clock()
    print end - start, 'seconds to parse all documents'


    start = time.clock()
    preprocessor.clean_data()
    end = time.clock()
    print end - start, 'seconds to remove stop words and stem all bodies of documents'
    PreprocessorHelper.write_to_file(preprocessor.tokenized_body_cleaned_dict, "cleaned.json")

    #start = time.clock()
    preprocessor.populate_tfidf_feature_vector_dictionary()
    #end = time.clock()
    #print end - start, 'seconds to populate tfidf feature vector'

    #PreprocessorHelper.write_to_file(preprocessor.topic_wise_dict,"tfidf.json")
    #preprocessor.clear_topic_dict()

    start = time.clock()
    preprocessor.populate_term_frequencies_feature_vector_dictionary()
    end = time.clock()
    print end - start, 'seconds to populate tf feature vector'


    PreprocessorHelper.write_to_file(preprocessor.topic_wise_dict,"term_frequencies.json")
    preprocessor.clear_topic_dict()

    '''start = time.clock()
    preprocessor.populate_bigram_feature_vector()
    end = time.clock()
    print end - start, 'seconds to populate bigram feature vector'


    PreprocessorHelper.write_to_file(preprocessor.topic_wise_dict, "bigrams_pmi_small.json")
    preprocessor.clear_topic_dict()
'''
if __name__ == "__main__": main()
