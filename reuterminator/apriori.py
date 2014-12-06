
import sys
from preprocessorhelper import *
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser

class Apriori:
    def __init__(self, minSupport, minConfidence):
        self.minSupport = minSupport
        self.minConfidence = minConfidence
        self.freqSet = defaultdict(int)
        self.largeSet = dict()


    def subsets(self,arr):
        """ Returns non empty subsets of arr"""
        return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


    def returnItemsWithMinSupport(self, itemSet):
            """calculates the support for items in the itemSet and returns a subset
           of the itemSet each of whose elements satisfies the minimum support"""
            #print itemSet
            _itemSet = set()
            localSet = defaultdict(int)

            for item in itemSet:
                    for transaction in self.transactionList:
                            if item.issubset(transaction):
                                    self.freqSet[item] += 1
                                    localSet[item] += 1


            for item, count in localSet.items():
                    support = float(count)/len(self.transactionList)
                    #print support
                    if support >= self.minSupport:
                            _itemSet.add(item)

            return _itemSet


    def joinSet(self, itemSet, length):
            """Join a set with itself and returns the n-element itemsets"""
            return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


    def getItemSetTransactionList(self, data_iterator):
        transactionList = list()
        itemSet = set()
        for record in data_iterator:
            transaction = frozenset(record)
            transactionList.append(transaction)
            for item in transaction:
                itemSet.add(frozenset([item]))              # Generate 1-itemSets
        return itemSet, transactionList




    def getSupport(self, item):
            """local function which Returns the support of an item"""
            return float(self.freqSet[item])/len(self.transactionList)


    def run(self, transactions):
        """
        run the apriori algorithm. data_iter is a record iterator
        Return both:
         - items {tuple : support}
         - rules ((pretuple, posttuple), confidence)
        """
        itemSet, self.transactionList = self.getItemSetTransactionList(transactions)

        assocRules = dict()
        # Dictionary which stores Association Rules

        oneCSet = self.returnItemsWithMinSupport(itemSet)
        #print oneCSet
        currentLSet = oneCSet
        k = 2
        frequentSets = set()
        while(currentLSet != set([])):
            frequentSets = frequentSets.union(currentLSet)
            self.largeSet[k-1] = currentLSet
            currentLSet = self.joinSet(currentLSet, k)
            currentCSet = self.returnItemsWithMinSupport(currentLSet)
            currentLSet = currentCSet
            k = k + 1

        #print self.largeSet

        toRetItems = {}
        for key, value in self.largeSet.items()[1:]:

            #toRetItems.extend([(tuple(item), self.getSupport(item))for item in value])
            for item in value:
                toRetItems[item] = self.getSupport(item)
            #toRetItems[tuple(item)]

        toRetRules = []
        '''
        for key, value in largeSet.items()[1:]:
            for item in value:
                _subsets = map(frozenset, [x for x in self.subsets(item)])
                for element in _subsets:
                    remain = item.difference(element)
                    if len(remain) > 0:
                        confidence = getSupport(item)/getSupport(element)
                        if confidence >= self.minConfidence:
                            toRetRules.append(((tuple(element), tuple(remain)),
                                               confidence))
        '''
        toRetRules = self.pruneRulesWithMinConfidence()
        return toRetItems, toRetRules




    def pruneRulesWithMinConfidence(self):
        toRetRules = []
        for key, value in self.largeSet.items()[1:]:
            for item in value:
                _subsets = map(frozenset, [x for x in self.subsets(item)])
                for element in _subsets:
                    remain = item.difference(element)
                    if len(remain) > 0:
                        support = self.getSupport(item)
                        confidence = support/self.getSupport(element)
                        #if confidence >= self.minConfidence:
                        toRetRules.append(((tuple(element), tuple(remain)),
                                               confidence, support))
        return toRetRules

    def printResults(self, items, rules):
        """prints the generated itemsets and the confidence rules"""
        for item, support in items:
            print "item: %s , %.3f" % (str(item), support)
        print "\n------------------------ RULES:"
        for rule, confidence in rules:
            pre, post = rule
            print "Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence)


    def dataFromFile(self, fname):
            """Function which reads from the file and yields a generator"""
            file_iter = open(fname, 'rU')
            for line in file_iter:
                    line = line.strip().rstrip(',')                         # Remove trailing comma
                    record = frozenset(line.split(','))
                    yield record


if __name__ == "__main__":
    '''
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv',
                         default=None)
    optparser.add_option('-s', '--minSupport',
                         dest='minS',
                         help='minimum support value',
                         default=0.15,
                         type='float')
    optparser.add_option('-c', '--minConfidence',
                         dest='minC',
                         help='minimum confidence value',
                         default=0.6,
                         type='float')

    (options, args) = optparser.parse_args()

    inFile = None
    if options.input is None:
            inFile = sys.stdin
    elif options.input is not None:
            inFile = dataFromFile(options.input)
    else:
            print 'No dataset filename specified, system with exit\n'
            sys.exit('System will exit')

    minSupport = options.minS
    minConfidence = options.minC
    '''

    apr = Apriori(0.10,0.5)
    inFile = apr.dataFromFile("INTEGRATED-DATASET.csv")
    items, rules = apr.run(inFile)
    print rules
    PreprocessorHelper.write_to_file_json("rules.json", rules)
    #printResults(items, rules)
