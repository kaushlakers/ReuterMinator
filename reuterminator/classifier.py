import json
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import *
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import time
from sklearn.feature_extraction import DictVectorizer
import numpy as np
class Classifier:
    def __init__(self, feature_type, classifier_name, classifier, sparse_flag):
        self.feature_type = feature_type
        self.classifier_name = classifier_name
        self.classifier = classifier
        self.feature_vectors = {}
        self.feature_vectors_tuples_for_train = []
        self.feature_vectors_tuples_for_test = []
        self.vectorizer = None
        self.sparse_flag = sparse_flag


    def prepare_data_for_classify(self):
        self.feature_vectors = self.convert_to_utf(json.load(open(self.feature_type+'.json')))
        print "No of topics is ", len(self.feature_vectors)
        print "preparing data"
        self.preprare_train_and_test_set()


    def train_classifier(self):
        #classifier = SklearnClassifier(classifier_type)
        #self.classifier = classifier_type
        X = np.array([feature_vector[0] for feature_vector in self.feature_vectors_tuples_for_train])
        Y = np.array([feature_vector[1] for feature_vector in self.feature_vectors_tuples_for_train])
        self.vectorizer = DictVectorizer(dtype=float, sparse=self.sparse_flag)
        X = self.vectorizer.fit_transform(X)
        #if self.sparse_flag:
         #   X = X.toarray()
        print "training"
        self.classifier.fit(X,Y)

    def test_classifier(self):
        print "testing classifier"
        X_test = self.vectorizer.transform(np.array([feature_vector[0] for feature_vector in self.feature_vectors_tuples_for_test]))
        predicted_list = self.classifier.predict(X_test)
        correct = 0
        wrong = 0
        #if self.sparse_flag:
         #   X_test = X_test.toarray()
        accuracy = self.classifier.score(X_test, [test_label[1] for test_label in self.feature_vectors_tuples_for_test])
        print accuracy
        classifier_metric_file = open(self.classifier_name+'.mtc','w')
        classifier_metric_file.write("Accuracy - "+str(accuracy))
        classifier_metric_file.write(classification_report([test_label[1] for test_label in self.feature_vectors_tuples_for_test],predicted_list))
        classifier_metric_file.close()

    def classify_decision_tree(self):

        print "training decision tree"
        classifier = DecisionTreeClassifier.train(self.feature_vectors_tuples_for_train, depth_cutoff=200, entropy_cutoff=0.1)
        print "testing classifier"
        classified_labels =  classifier.batch_classify([feature_set_tuple[0] for feature_set_tuple in self.feature_vectors_tuples_for_test])
        correct = 0
        wrong = 0
        for i in range(0, len(classified_labels)):
            if classified_labels[i] is self.feature_vectors_tuples_for_test[i][1]:
                correct += 1
            else:
                wrong += 1
        print correct/wrong


    #separates train and test data
    def preprare_train_and_test_set(self):
        for topic,topic_vector in self.feature_vectors.iteritems():
            train_data_limit = int(0.7*len(topic_vector))
            i = 0
            for doc_id,feature_vector in topic_vector.iteritems():

                #classify_tuple = (dict(feature_vector[self.feature_type].items()+feature_vector['places'].items()), topic)
                classify_tuple = (feature_vector[self.feature_type], topic)
                #classify_tuple = (feature_vector['places'], topic)

                i = i+1
                if i <= train_data_limit:
                    self.feature_vectors_tuples_for_train.append(classify_tuple)
                else:
                    self.feature_vectors_tuples_for_test.append(classify_tuple)
        print len(self.feature_vectors_tuples_for_train)
        print len(self.feature_vectors_tuples_for_test)


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
    #MultinomialNB(),DecisionTreeClassifier(max_depth=200),KNeighborsClassifier(n_neighbors=5,weights='distance')

    classifier = Classifier('term_frequencies', "Naive_Bayes", MultinomialNB(), True)
    classifier.prepare_data_for_classify()
    start = time.clock()
    classifier.train_classifier()
    end = time.clock()
    print end - start, 'seconds to train MultinomialNB'
    start = time.clock()
    classifier.test_classifier()
    end = time.clock()
    print end - start, 'seconds to test MultinomialNB'

    classifier = Classifier('term_frequencies', "KNeighbors",KNeighborsClassifier(n_neighbors=5,weights='distance'), True)
    classifier.prepare_data_for_classify()
    start = time.clock()
    classifier.train_classifier()
    end = time.clock()
    print end - start, 'seconds to train kneighbors'
    start = time.clock()
    classifier.test_classifier()
    end = time.clock()
    print end - start, 'seconds to test kneighbors'

    classifier = Classifier('term_frequencies', "DecisionTree",DecisionTreeClassifier(max_depth=200), False)
    classifier.prepare_data_for_classify()
    start = time.clock()
    classifier.train_classifier()
    end = time.clock()
    print end - start, 'seconds to train decision tree'
    start = time.clock()
    classifier.test_classifier()
    end = time.clock()
    print end - start, 'seconds to test decision tree'

    #start = time.clock()
    #classifier.classify_naive_bayes(KNeighborsClassifier(n_neighbors=5,weights='distance'))
    #end = time.clock()
    #print end - start, 'seconds to train and classify with n neighbors'
    #code for decision tree. Not sure it works well yet
    #start = time.clock()
    #classifier.classify_decision_tree()
    #end = time.clock()
    #print end - start, 'seconds to train and classify with decision tree'

if __name__ == "__main__": main()
