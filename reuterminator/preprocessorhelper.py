import nltk
import string
import json
from nltk import *
from nltk import corpus
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import sys
import os
import os.path
import json
import time
import numpy as np
import pickle
from scipy.sparse import csr_matrix


JUNK_WORDS = ['<','>',':',"''",'#','cc','',',','s','reuter','note','said','mln','dlr','pct']


''' Class that contains auxiliary (static) helper methods that are used by Preprocessor '''
class PreprocessorHelper:

    @staticmethod
    def write_to_file_json(filename, data):
       #Converts to json and dumps the contents to a file
       with open(filename, 'w') as outfile:
           json.dump(data, outfile, indent=4)
       outfile.close()

    @staticmethod
    def read_file_json(filename):
       #Converts to json and dumps the contents to a file
       with open(filename, 'r') as infile:
          data = json.load(infile)
       infile.close()
       return data


    @staticmethod
    def write_to_file(filename, data):
        #Converts to json and dumps the contents to a file
        with open(filename, 'wb') as outfile:
            #Removing non-unicode characters from the dataset
            #self.parsed_data = unicode(self.parsed_data, errors='ignore')
            pickle.dump(data, outfile)
        outfile.close()

    @staticmethod
    def load_file(filename):
        #Converts to json and dumps the contents to a file
        with open(filename, 'rb') as infile:
            #Removing non-unicode characters from the dataset
            #self.parsed_data = unicode(self.parsed_data, errors='ignore')
            data = pickle.load(infile)
        infile.close()
        return data

    @staticmethod
    def save_csr_matrix(filename, array):
        np.savez(filename,data = array.data ,indices=array.indices,
                 indptr =array.indptr, shape=array.shape )
    @staticmethod
    def load_csr_matrix(filename):
        loader = np.load(filename)
        return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                             shape = loader['shape'])


    @staticmethod
    def get_words_after_stop_word_removal(tokens):
        good_words = [w for w in tokens if w.lower() not in nltk.corpus.stopwords.words('english')]
        better_words = [w for w in good_words if w.lower() not in JUNK_WORDS]
        return better_words

    @staticmethod
    def get_stemmed_words(tokens):
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(w) for w in tokens]
        return stemmed_words

    @staticmethod
    def convert_to_utf(input):
        if isinstance(input, dict):
            temp_dict = {}
            for key,value in input.iteritems():
                temp_dict[PreprocessorHelper.convert_to_utf(key)] = PreprocessorHelper.convert_to_utf(value)
            return temp_dict
            #return {self.convert_to_utf(key): self.convert_to_utf(value) for key, value in input.iteritems()}
        elif isinstance(input, list):
            temp_list = []
            for element in input:
                temp_list.append(PreprocessorHelper.convert_to_utf(element))
            return temp_list
            #return [self.convert_to_utf(element) for element in input]
        elif isinstance(input, unicode):
            return input.encode('utf-8')
        else:
            return input
