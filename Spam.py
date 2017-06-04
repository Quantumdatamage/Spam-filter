import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

data = data.append(dataFrameFromDirectory('C:\\Users\\Max\\Desktop\\Data science folder\\DataScience-Python3\\emails\\spam', 'spam'))
data = data.append(dataFrameFromDirectory('C:\\Users\\Max\\Desktop\\Data science folder\\DataScience-Python3\\emails\\ham', 'ham'))

from __future__ import division
from sklearn.cross_validation import train_test_split
import numpy as np

# Training
# Now split data into training (80 %) and test data sets (20 %) - TRAINING
train, test   = train_test_split(data, test_size=0.2) #distro the train and test values
train_counts  = vectorizer.fit_transform(train['message'].values) #turn the train messages into values
targets       = train['class'].values #turn the targets
classifier.fit(train_counts, targets)

# Now test on TEST data
# Testing
examples = test['message']
examples = np.array(examples)
#test_counts = vectorizer.transform(test)
test_counts = vectorizer.transform(examples)
predictions = classifier.predict(test_counts)
#print (predictions)

test         = test[['class']]
test['pred'] = predictions
#Number classified as spam
Length_ClassifiedAsSpam = len(test[test['pred']== 'spam'])

#Probability of being ham given that it is classified as spam
P_ham_given_classified_as_spam = 100. * len(test[(test['pred'] == 'spam') & (test['class'] == 'ham')]) / Length_ClassifiedAsSpam

#print result to screen
print ("P(ham|classified_as_spam) = " , P_ham_given_classified_as_spam, "%")

print ("Length of test data set       : {}".format(len(test)))
print ("% of correct classifications  : {}".format(100*len(test[test['class'] == test['pred']])/len(test)))
print ("% of incorrect classifications: {}".format(100*len(test[test['class'] != test['pred']])/len(test)))
