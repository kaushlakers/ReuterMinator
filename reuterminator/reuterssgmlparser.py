import sgmllib
import sys
import os
import os.path

####
##
## Class that parses all the Rueters SGML files and returns a list with the following attributes:
##
##
####
class ReutersSGMLParser(sgmllib.SGMLParser):

    #DATA_SET_DIRECTORY = '../SmallDataset/'
    DATA_SET_DIRECTORY = '../Datasets/'

    def __init__(self, verbose=1):
        sgmllib.SGMLParser.__init__(self, verbose)

        self.in_title = 0
        """Flag indicating whether or not we're parsing the title."""

        self.in_dateline = 0
        """Flag indicating whether or not we're parsing the dateline"""

        self.in_body = 0
        """Flag indicating whether or not we're parsing the body"""
        self.in_topic = 0

        self.in_places = 0

        self.title = ""
        """Title of the document"""

        self.item_id = 0
        """Document ID"""

        self.dateline = ""
        """Date line for the document"""

        self.body = ""
        """Body of the document"""

        self.topics = []
        self.docs = {}
        self.places = []
        self.all_topics = []
        self.all_topics_dict = {}
        self.all_places = []
        self.all_places_dict = {}
    def parse_all_docs(self):

        """Parse the given string file_string, which is an SGML encoded file."""

        self.docs = {}
        for file_name in os.listdir(ReutersSGMLParser.DATA_SET_DIRECTORY):
            print 'Parsing ' + file_name
            file_path = ReutersSGMLParser.DATA_SET_DIRECTORY + file_name
            f = open(file_path, "r")
            self.file_string = f.read()
            self.feed(self.file_string)
            f.close()
            print 'Items parsed: ', len(self.docs.keys())

        for place,freq in self.all_places_dict.iteritems():
            if freq > 2000:
                self.all_places.remove(place)
        self.close()

    def get_parsed_dataset(self):
        return self.docs

    def get_all_topics(self):
        return self.all_topics

    def get_all_places(self):
        return self.all_places

    #handle_data method is called in between start_<tag> and end_<tag>

    def handle_data(self, data):
        """handles different types of data and populates it to member variables after lowering case."""

        if self.in_body:
            self.body += data.lower()
        elif self.in_title:
            self.title += data.lower()
        elif self.in_dateline:
            self.dateline += data.lower()

        elif self.in_topic:
            self.topics.append(data.lower())
            if data.lower() not in self.all_topics:
                self.all_topics.append(data.lower())
                self.all_topics_dict[data.lower()] = 1
            else:
                self.all_topics_dict[data.lower()] += 1

        elif self.in_places:
            self.places.append(data.lower())
            if data.lower() not in self.all_places:
                self.all_places.append(data.lower())
                self.all_places_dict[data.lower()] = 1
            else:
                self.all_places_dict[data.lower()] += 1
    ####
    # Handle the Reuters tag
    ####
    def start_reuters(self, attributes):
        """Process Reuters tags, which bracket a document. Create a new
        file for each document encountered."""

        for name, value in attributes:
            if name == "newid":
                self.item_id = value

    def end_reuters(self):
        """Reset all variables."""

        # lets cleanup the topics variable a bit by converting to a list
        self.docs[self.item_id] = {'title' : self.title, 'dateline': self.dateline, 'body' : self.body, 'topics' : self.topics, 'places' : self.places}

        # Reset variables
        self.in_title = 0
        self.in_dateline = 0
        self.in_body = 0
        self.in_topic = 0
        self.item_id = 0
        self.in_places = 0

        self.title = ""
        self.body = ""
        self.dateline = ""
        self.topics = []
        self.places = []

    ####
    # Handle TITLE tags
    ####
    def start_title(self, attributes):
        """Indicate that the parser is in the title portion of the document."""

        self.in_title = 1

    def end_title(self):
        """Indicate that the parser is no longer in the title portion of the document."""

        self.in_title = 0

    ####
    # Handle PLACES tags
   	 ####
    def start_places(self, attributes):
        """Indicate that the parser is in the title portion of the document."""

        self.in_places = 1

    def end_places(self):
        """Indicate that the parser is no longer in the title portion of the document."""

        self.in_places = 0

    ####
    # Handle DATELINE tags
    ####
    def start_dateline(self, attributes):
        """Indicate that the parser is in the dateline portion of the document."""

        self.in_dateline = 1

    def end_dateline(self):
        """Indicate that the parser is no longer in the dateline portion of the document."""

        self.in_dateline = 0

    ####
    # Handle BODY tags
    ####
    def start_body(self, attributes):
        """Indicate that the parser is in the body portion of the document.
        """

        self.in_body = 1

    def end_body(self):
        """Indicate that the parser is no longer in the body portion of the document."""

        self.in_body = 0

    ####
    # Handle TOPIC tags
    ####
    def start_topics(self, attributes):
        """Indicate that the parser is in the topics portion of the document."""

        self.in_topic = 1

    def end_topics(self):
        """Indicate that the parser is no longer in the topics portion of the document."""

        self.in_topic = 0
