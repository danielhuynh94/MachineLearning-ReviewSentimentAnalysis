import sys

testFile = "bayesbest.py"
trainDir = "training/"
testDir = "testing/"

printMeasures = True

# Take the testing directory from the command line argument
if (len(sys.argv) > 1):
    printMeasures = False
    testDir = sys.argv[1]


def calculate_recall_precision(label, prediction):
    # Return the precision, recall, and f_measure from the label and prediction lists
    # This function was retrieved from the site:
    # https://phpcoderblog.wordpress.com/2017/11/02/how-to-calculate-accuracy-precision-recall-and-f1-score-deep-learning-precision-recall-f-score-calculating-precision-recall-python-precision-recall-scikit-precision-recall-ml-metrics-to-use-bi/
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i in range(0, len(label)):
        if prediction[i] == "positive":
            if prediction[i] == label[i]:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if prediction[i] == label[i]:
                true_negatives += 1
            else:
                false_negatives += 1

    # a ratio of correctly predicted observation to the total observations
    accuracy = (true_positives + true_negatives) \
        / (true_positives + true_negatives + false_positives + false_negatives)
    # precision is "how useful the search results are"
    precision = true_positives / (true_positives + false_positives)
    # recall is "how complete the results are"
    recall = true_positives / (true_positives + false_negatives)

    f_measure = 2 / ((1 / precision) + (1 / recall))

    return accuracy, precision, recall, f_measure


exec(open(testFile).read())
bc = Bayes_Classifier(trainDir)

iFileList = []

for fFileObj in os.walk(testDir + "/"):
    iFileList = fFileObj[2]
    break
print('%d test reviews.' % len(iFileList))

results = {"negative": 0, "neutral": 0, "positive": 0}

print("\nFile Classifications:")
labels = []
label = ''
prediction = []
for filename in iFileList:
    try:
        fileText = bc.loadFile(testDir + filename)
        result = bc.classify(fileText)
        print("%s: %s" % (filename, result))
        results[result] += 1

        if printMeasures:
            # Get the rating label of the file
            rating = int(bc.getRating(filename))

            if rating > 3:
                label = "positive"
            elif rating < 3:
                label = "negative"
            else:
                label = "neutral"

            # Save the labels and predictions to calculate precision, recall, and f_measure later
            labels.append(label)
            prediction.append(result)

    except:
        print("error")

print("\nResults Summary:")
for r in results:
    print("%s: %d" % (r, results[r]))

if printMeasures:
    # Calculate the precision, recall, and f-measure
    accuracy, precision, recall, f_measure = calculate_recall_precision(
        labels, prediction)

    print('accuracy: ' + str(accuracy))
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print('f-measure: ' + str(f_measure))
