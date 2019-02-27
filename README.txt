NaiveBayesReviewClassifier

    NaiveBayesReviewClassifier is a library for classifying whether a review is sentimentally positive, negative, or neutral based on the review's content.
	After a file's location is provided, the classifier will output whether the review is positive, negative, or neutral. 

    Author: Huy "Daniel" Huynh

I. Installation
    1. Requirements
        * Linux
        * Python 3.3 and up

    2. Import Library In Python Files
        ```python
        from bayes import *
	from bayesbest import *
        ```

    3. Files
        * bayesbest.py - library for enhanced classifier
        * bayes.py - library for basic classifier
        * evaluate.py - library for evaluating a classifier
		
		# The following files must be removed before attempting to train the classifier with new training data
		* positiveReviewUnigrams - pickled file containing a dictionary of positive unigrams of last training
		* negativeReviewUnigrams - pickled file containing a dictionary of negative unigrams of last training
		* positiveReviewBigrams - pickled file containing a dictionary of positive bigrams of last training
		* negativeReviewBigrams - pickled file containing a dictionary of negative bigrams of last training
		* trainingDataAttributes - pickled file containing supporting data for the dictionaries above
		
    4. Command line examples
		```python
        # In file evaluate.py, the following variables must be specified: specify which classifier library you want to evaluate in variable "testFile"
		# testFile: which library of classifier you want to evaluate
		# trainDir: folder containing all the documents for training the model. Documents in this folder must have the labels in their filenames.
		# testDir: folder containing all the documents for testing the model. Documents in this folder must have the labels in their filenames.
        python3 evaluate.py
        ```

        ```python
		# If an argument is passed in the terminal, documents in the specified testing folder will be evaluated
		# Documents in the specified folder don't have to have labels specified in them
		# As a consequence, accuracy, precision, recall, and f-measure values will not be printed
        python3 evaluate.py "testing/"
        ```

II. Usage
	1. Simple example:
	```python
	# Bayes_Classifier() takes in the folder containing training data. If no argument is passed in, 'movies_reviews/' is used
	>>> exec(open("bayes.py").read())
	>>> bc = Bayes_Classifier()
	>>> result = bc.classify("I love my AI class!")
	>>> print(result)
	positive
	```
	
	2. Train a classifier with data from a folder
	```python
	>>> exec(open("bayes.py").read())
	>>> bc = Bayes_Classifier('training/')
	```
	
	3. Use enhanced classifier
	```python
	>>> exec(open("bayesbest.py").read())
	>>> bc = Bayes_Classifier('training/')
	```	

III. Documentation
    # Define a Bayes Classifier
    class Bayes_Classifier(string trainingDirectory)
        # Property for the training directory
        trainDirectory

        # Property for the probability that any document can be positive
        probPositiveDocument

        # Property for the probability that any document can be negative
        probNegativeDocument

        # Property for the dictionary of positive unigrams
        positiveReviewUnigrams

        # Property for the dictionary of negative unigrams
        negativeReviewUnigrams

		# Property for the total frequency of all positive unigrams
		sumPositiveUnigramFrequencies
		
		# Property for the total frequency of all negative unigrams
		sumNegativeUnigramFrequencies
		
		# Property for the dictionary of positive bigrams
		positiveReviewBigrams
		
		# Property for the dictionary of negative bigrams
		negativeReviewBigrams
		
		# Property for the total frequency of all positive bigrams
		sumPositiveBigramFrequencies
		
		# Property for the total frequency of all negative bigrams
		sumNegativeBigramFrequencies
		
        # Trains the Naive Bayes Sentiment Classifier
        function train()

        # Get the rating of a file from the filename
        function getRating(string filename)

		# Increment the value of the key in the dictionary by one
		function incrementKeyByOne(dictionary, key)
		
		# Given a target string sText, this function returns the most likely document
        # class to which the target string belongs. This function should return one of three
        # strings: "positive", "negative" or "neutral".
		function classify(string text)
		
		# Return a dictionary of how often a token occurs in the list
		function getTokenFrequencyDictionary(list)
		
		# Given a file name, return the contents of the file as a string.
		function loadFile(string filename)
		
		# Given an object and a file name, write the object to the file using pickle.
		function save(object, string filename)
		
		# Given a file name, load and return the object stored in the file.
		function load(string filename)
		
		# Given a string of text sText, returns a list of the individual tokens that
        # occur in that string (in order).
		function tokenize(string text)
		
		# Get a list of bigrams from the list of unigrams
		function getBigramList(list unigramList)
		
IV. License
    [MIT](https://choosealicense.com/licenses/mit/) © Huy "Daniel" Huynh