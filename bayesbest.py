import math
import os
import pickle
import re
import string


class Bayes_Classifier:

    def __init__(self, trainDirectory="movies_reviews/"):
        # This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
        # cache of a trained classifier has been stored, it loads this cache.  Otherwise,
        # the system will proceed through training.  After running this method, the classifier
        # is ready to classify input text.'''

        self.trainDirectory = trainDirectory
        self.probPositiveDocument = 0
        self.probNegativeDocument = 0

        # Variables for unigrams
        self.positiveReviewUnigrams = {}
        self.negativeReviewUnigrams = {}
        self.sumPositiveUnigramFrequencies = 0
        self.sumNegativeUnigramFrequencies = 0

        # Variables for bigrams
        self.positiveReviewBigrams = {}
        self.negativeReviewBigrams = {}
        self.sumPositiveBigramFrequencies = 0
        self.sumNegativeBigramFrequencies = 0

        exists = os.path.isfile('positiveReviewUnigrams') and os.path.isfile('negativeReviewUnigrams') and os.path.isfile(
            "positiveReviewBigrams") and os.path.isfile("negativeReviewBigrams") and os.path.isfile('trainingDataAttributes')

        # If all the pickled files exists, load them
        if exists:
            trainingDataAttributes = self.load('trainingDataAttributes')
            self.probPositiveDocument = trainingDataAttributes.get(
                "probPositiveDocument")
            self.probNegativeDocument = trainingDataAttributes.get(
                "probNegativeDocument")

            # Data for unigrams
            self.positiveReviewUnigrams = self.load('positiveReviewUnigrams')
            self.negativeReviewUnigrams = self.load('negativeReviewUnigrams')
            self.sumPositiveUnigramFrequencies = trainingDataAttributes.get(
                "sumPositiveUnigramFrequencies")
            self.sumNegativeUnigramFrequencies = trainingDataAttributes.get(
                "sumNegativeUnigramFrequencies")

            # Data for bigrams
            self.positiveReviewBigrams = self.load("positiveReviewBigrams")
            self.negativeReviewBigrams = self.load("negativeReviewBigrams")
            self.sumPositiveBigramFrequencies = trainingDataAttributes.get(
                "sumPositiveBigramFrequencies")
            self.sumNegativeBigramFrequencies = trainingDataAttributes.get(
                "sumNegativeBigramFrequencies")
        else:
            self.train()

    def train(self):
        # Trains the Naive Bayes Sentiment Classifier.
        lFileList = []

        for fFileObj in os.walk(self.trainDirectory):
            lFileList = fFileObj[2]

        totalDocuments = len(lFileList)
        positiveReviewCount = 0
        negativeReviewCount = 0

        for filename in lFileList:
            try:
                rating = int(self.getRating(filename))
                # If rating is greater than 2, it's positive
                isPositiveRating = rating > 2
                # Load the text from the file
                text = self.loadFile(self.trainDirectory + filename)

                # Save the counts of positive and negative reviews to calculate prior probabilities later
                if isPositiveRating:
                    positiveReviewCount += 1
                else:
                    negativeReviewCount += 1

                # Get a list of unigrams from the text
                unigrams = self.tokenize(text)
                for unigram in unigrams:
                    if isPositiveRating:
                        self.incrementKeyByOne(
                            self.positiveReviewUnigrams, unigram)
                    else:
                        self.incrementKeyByOne(
                            self.negativeReviewUnigrams, unigram)

                # Bigram features
                bigrams = self.getBigramList(unigrams)
                for bigram in bigrams:
                    if isPositiveRating:
                        self.incrementKeyByOne(
                            self.positiveReviewBigrams, bigram)
                    else:
                        self.incrementKeyByOne(
                            self.negativeReviewBigrams, bigram)

            except:
                pass

        # Calculate the probabilities of documents to be either positive or negative
        self.probPositiveDocument = positiveReviewCount/totalDocuments
        self.probNegativeDocument = negativeReviewCount/totalDocuments

        self.sumPositiveUnigramFrequencies = sum(
            self.positiveReviewUnigrams.values())
        self.sumNegativeUnigramFrequencies = sum(
            self.negativeReviewUnigrams.values())

        self.sumPositiveBigramFrequencies = sum(
            self.positiveReviewBigrams.values())
        self.sumNegativeBigramFrequencies = sum(
            self.negativeReviewBigrams.values())

        self.save(self.positiveReviewUnigrams, "positiveReviewUnigrams")
        self.save(self.negativeReviewUnigrams, "negativeReviewUnigrams")
        self.save(self.positiveReviewBigrams, "positiveReviewBigrams")
        self.save(self.negativeReviewBigrams, "negativeReviewBigrams")
        self.save({"probPositiveDocument": self.probPositiveDocument,
                   "probNegativeDocument": self.probNegativeDocument,
                   "sumPositiveUnigramFrequencies": self.sumPositiveUnigramFrequencies,
                   "sumNegativeUnigramFrequencies": self.sumNegativeUnigramFrequencies,
                   "sumPositiveBigramFrequencies": self.sumPositiveBigramFrequencies,
                   "sumNegativeBigramFrequencies": self.sumNegativeBigramFrequencies}, "trainingDataAttributes")

    def getRating(self, filenameToMatch):
        # Get the rating of a file from the filename
        p = re.compile("(\d+)")
        return p.search(filenameToMatch).group()

    def incrementKeyByOne(self, dictionary, key):
        # Increment the value of the key in the dictionary by one
        value = dictionary.get(key)
        if value == None:
            dictionary[key] = 1
        else:
            dictionary[key] = value + 1

    def classify(self, sText):
        # Given a target string sText, this function returns the most likely document
        # class to which the target string belongs. This function should return one of three
        # strings: "positive", "negative" or "neutral".

        probPositiveDocument = math.log(self.probPositiveDocument)
        probNegativeDocument = math.log(self.probNegativeDocument)

        # Use the unigram feature to calculate the probabilities of being positive and negative
        unigramTokens = self.tokenize(sText)
        unigramFrequency = self.getTokenFrequencyDictionary(unigramTokens)

        for key, frequency in unigramFrequency.items():
            value = 0
            probability = 0

            # Look for it in positive dict
            value = self.positiveReviewUnigrams.get(key)
            if value != None:
                probability = value / self.sumPositiveUnigramFrequencies
            else:
                probability = 1 / \
                    (self.sumPositiveUnigramFrequencies +
                     self.sumNegativeUnigramFrequencies)
            probPositiveDocument += math.log(frequency * probability)

            # Look for it in negative dict
            value = self.negativeReviewUnigrams.get(key)
            if value != None:
                probability = value / self.sumNegativeUnigramFrequencies
            else:
                probability = 1/(self.sumNegativeUnigramFrequencies +
                                 self.sumPositiveUnigramFrequencies)
            probNegativeDocument += math.log(frequency * probability)

        # Use the bigram feature to calculate the probabilities of being positive and negative
        bigramTokens = self.getBigramList(unigramTokens)
        bigramFrequency = self.getTokenFrequencyDictionary(bigramTokens)

        for key, frequency in bigramFrequency.items():
            value = 0
            probability = 0

            # Look for it in positive dict
            value = self.positiveReviewBigrams.get(key)
            if value != None:
                probability = value / self.sumPositiveBigramFrequencies
            else:
                probability = 1 / \
                    (self.sumPositiveBigramFrequencies +
                     self.sumNegativeBigramFrequencies)
            probPositiveDocument += math.log(frequency * probability)

            # Look for it in negative dict
            value = self.negativeReviewBigrams.get(key)
            if value != None:
                probability = value / self.sumNegativeBigramFrequencies
            else:
                probability = 1/(self.sumNegativeBigramFrequencies +
                                 self.sumPositiveBigramFrequencies)
            probNegativeDocument += math.log(frequency * probability)

        # By default the result is neutral
        result = "neutral"
        # If the difference between the logs of the 2 probabilities is greater than 0.3. It's significant enough to be either one
        if abs(probPositiveDocument-probNegativeDocument) >= 0.3:
            if (probNegativeDocument > probPositiveDocument):
                result = "negative"
            else:
                result = "positive"

        return result

    def getTokenFrequencyDictionary(self, list):
        # Return a dictionary of how often a token occurs in the list
        tokenFrequency = {}
        for token in list:
            frequency = tokenFrequency.get(token)
            if frequency != None:
                tokenFrequency[token] += 1
            else:
                tokenFrequency[token] = 1
        return tokenFrequency

    def loadFile(self, sFilename):
        # Given a file name, return the contents of the file as a string.
        f = open(sFilename, "r")
        sTxt = f.read()
        f.close()
        return sTxt

    def save(self, dObj, sFilename):
        # Given an object and a file name, write the object to the file using pickle.
        f = open(sFilename, "wb")
        p = pickle.Pickler(f)
        p.dump(dObj)
        f.close()

    def load(self, sFilename):
        # Given a file name, load and return the object stored in the file.
        f = open(sFilename, "rb")
        u = pickle.Unpickler(f)
        dObj = u.load()
        f.close()
        return dObj

    def tokenize(self, sText):
        # Given a string of text sText, returns a list of the individual tokens that
        # occur in that string (in order).
        lTokens = []
        sToken = ""
        for c in sText.lower():
            if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
                sToken += c
            else:
                if sToken != "":
                    lTokens.append(sToken)
                    sToken = ""
                if c.strip() != "":
                    lTokens.append(str(c.strip()))

        if sToken != "":
            lTokens.append(sToken)

        return lTokens

    def getBigramList(self, unigramList):
        # Get a list of bigrams from the list of unigrams
        lTokens = []
        for i in range(len(unigramList)-1):
            lTokens.append(unigramList[i] + ' ' + unigramList[i+1])
        return lTokens
