from apriori import *
from preprocessorhelper import *
import random
FEATURE = "tfidf"
class AprioriClassifier:

    #constructor does all the training and rule extractions
    def __init__(self, transactions, labels, train_test_split=0.80):
        self.transactions = transactions
        self.labels = labels

        self.apr = Apriori(0.04,0.1)

        #adding the labels to the transactions to find frequent sets
        transactions_with_labels = []
        self.label_set = set()
        item_set = set()

        #Add the topics to the transaction set for frequent itemset mining
        for i in range(0, len(self.transactions)):
            self.label_set = self.label_set.union(self.labels[i])
            transactions_with_labels.append(self.transactions[i].union(self.labels[i]))

        #split test and train data
        split_index = int(train_test_split*len(self.transactions))
        self.transactions_train = transactions_with_labels[0:split_index]
        self.labels_train = self.labels[0:split_index]
        self.transactions_test = self.transactions[split_index:]
        self.labels_test = self.labels[split_index:]

        #run Apriori algorithm
        support_dict,rule_list = self.apr.run(self.transactions_train)

        #extract rules of the form w1 w2 -->C1 C2
        rules = self.prune_rules_with_labels_on_RHS(rule_list)

        #sorting rules, first by confidence and for each confidence level, on support
        sorted_rules = sorted(rules, key =lambda x:(x[1],x[2]), reverse=True)

        #prune and remove redundant rules eg: w1 w2-->C1 > w1 w2-->C2 then second rule is redundant
        self.classifier_rules = self.prune_subsumed_rules(sorted_rules)


    #Not using this yet. Ignore
    def train(transactions, labels):

        transactions_with_labels = []
        self.label_set = set()
        item_set = set()
        #Add the topics to the transaction set for frequent itemset mining
        for i in range(0, len(transactions)):
            self.label_set = self.label_set.union(self.labels[i])
            transactions_with_labels.append(transactions[i].union(labels[i]))

        support_dict,rule_list = self.apr.run(transactions_with_labels)

        #prune rules of the form w1 w2 -->C1 C2
        rules = self.prune_rules_with_labels_on_RHS(rule_list)


        sorted_rules = sorted(rules, key =lambda x:(x[1],x[2]), reverse=True)

        #prune and remove redundant rules eg: w1 w2-->C1 > w1 w2-->C2 then second rule is redundant
        self.classifier_rules = self.test_predicting_ability_of_rules(sorted_rules)



    #extract rules of the form w1 w2 -->C1 C2
    def prune_rules_with_labels_on_RHS(self, rules):
        useful_rules = []
        for rule,confidence,support in rules:
            LHS,RHS = rule
            post_set = set([item for item in RHS])
            pre_set = set([item for item in LHS])
            if post_set.issubset(self.label_set):
                if not pre_set.intersection(self.label_set):
                    useful_rules.append((rule,confidence,support))
        return useful_rules


    #prune and remove redundant rules eg: w1 w2-->C1 > w1 w2-->C2 then second rule is redundant
    def prune_subsumed_rules(self, sorted_rules):
        classifier_rules = []
        transactions_train_copy = list(self.transactions_train)
        labels_train_copy = list(self.labels_train)

        for rule,confidence,support in sorted_rules:
            temp = set()
            marked = {}
            LHS,RHS = rule
            for i in range(0, len(transactions_train_copy)):
                if set(LHS).issubset(transactions_train_copy[i]) and set(RHS).issubset(labels_train_copy[i]):
                    temp = temp.union([i])
                    marked[rule] = True

            #remove documents that have been covered by this rule
            if rule in marked:
                transactions_train_copy = [m for n,m in enumerate(transactions_train_copy) if n not in temp]
                labels_train_copy = [m for n,m in enumerate(labels_train_copy) if n not in temp]
                classifier_rules.append(rule)

        #find default label - most common label in remaining data
        default_label_dict = {}
        if transactions_train_copy != []:
            for i in range(0, len(labels_train_copy)):
                for label in labels_train_copy[i]:
                    if label in default_label_dict:
                        default_label_dict[label] += 1
                    else:
                        default_label_dict[label] = 1

            default_label = sorted(default_label_dict.items(), key = lambda x:(x[1]), reverse=True)[0]
            classifier_rules.append(("default",tuple([default_label[0]])))
            print classifier_rules
        return classifier_rules

    def classify(self):
        classified_labels = []
        for i in range(0, len(self.transactions_test)):
            transaction = self.transactions_test[i]
            for rule in self.classifier_rules:
                LHS,RHS = rule
                if set(LHS).issubset(transaction) or LHS is "default":
                    classified_labels.append(list(RHS))
                    break
        return classified_labels

    def accuracy(self, Y_classified):
        correct = 0
        label_compare = []
        for i in range(0, len(Y_classified)):
        #    print Y_classified[i]
            label_compare.append((Y_classified[i], self.labels_test[i]))
            if set(Y_classified[i]).intersection(set(self.labels_test[i])):
                correct += 1
        PreprocessorHelper.write_to_file_json("result.json", label_compare)
        acc = correct/float(len(Y_classified))
        return acc

def split_data(ratio, transactions, labels):
    split_index = int(train_test_split*len(transactions))
    transactions_train = transactions_with_labels[0:split_index]
    labels_train = labels[0:split_index]
    transactions_test = transactions_with_labels[split_index:]
    labels_test = labels[split_index:]
    return ((transactions_train, labels_train),(transactions_test, labels_test))


def get_transactional_data(data):
    transactions = []
    labels = []
    label_set = set()
    for doc_id, vector in data.iteritems():
        if vector["topics"] != []:
            labels.append(vector["topics"])
            transactions.append(set(vector[FEATURE].keys()))
            label_set = label_set.union(vector["topics"])
    return (transactions,labels)

def main():

    feature_vectors = PreprocessorHelper.convert_to_utf(PreprocessorHelper.read_file_json(FEATURE+".json"))
    transactions,labels = get_transactional_data(feature_vectors)
    ar_classifier = AprioriClassifier(transactions, labels)
    classified_labels = ar_classifier.classify()
    print ar_classifier.accuracy(classified_labels)

    #print classified_labels
if __name__ == "__main__":
    main()
