import pandas as pd
import numpy as np
import math

reviews_df = pd.read_csv("restaurant_reviews_data/reviews_tr.csv")

reviews_df = reviews_df.head(1000)
#label = int(reviews_df.iloc[4]['label'])
#print("type is ", type(label))

test_df = pd.read_csv("restaurant_reviews_data/reviews_te.csv")
test_df = test_df.head(50)
#sprint(reviews_df["text"][0])

#list of dictionaries, each dictionary maps a word to its frequency in that document
allUnigrams = []

#dictionary that maps word to word number
allWords = {}


#dictionary that maps the word number to the log(idf) component of tf-df
documentFrequency = {}

def addToUnigram(unigram, text, colIndex):
    #build the unigram dictionary for the specific document
    for word in text.split():  
        wordNum = allWords[word] 
        #increase the frequency count in the unigram 2d array
        unigram[wordNum][colIndex] += 1
    return unigram


def addToWordDictionary(text):
    count = len(allWords)
    for word in text.split():
        if word not in allWords:
            allWords[word] = count
            count += 1


def createWordDictionary(df):
    for i, row in df.iterrows():
        addToWordDictionary(row['text'])


def createUnigram(df):
    arr = np.zeros([len(allWords), len(df)], dtype=int)
    for i, row in df.iterrows():
        print(i)
        arr = addToUnigram(arr, row['text'], i) 
    return arr

#loop through each row of unigram 2d numpy array (each unique word), where cell value > 0 increment document count 
def createDocumentFrequency(arr):
    ans_arr = arr
    numDocs = arr.shape[1] #number of Documents = columns in 2d array
    #print("number of documents:", numDocs)
    wordNum = 0
    for row in arr:
        doc_freq = 0
        for cell in row: 
            if cell > 0:
                doc_freq += 1
        if doc_freq == 0:
            documentFrequency[wordNum] = 0
        else:
            documentFrequency[wordNum] = math.log10(numDocs / doc_freq)
        ans_arr[wordNum] = ans_arr[wordNum] * documentFrequency[wordNum]
        wordNum += 1
    return ans_arr


    


#createUnigram(reviews_df)
#for elem in allUnigrams:
#    print(elem)
#    print("test")


createWordDictionary(reviews_df)
#we need to get the unseen words in test data and add them to our word dictionary
createWordDictionary(test_df) 
unigram_arr = createUnigram(reviews_df)
tfidf_arr = createDocumentFrequency(unigram_arr)
#print(allWords)
#print(unigram_arr)
#print(documentFrequency)
#print()


#initialize w vector to 0, add extra component at end for bias
w = np.zeros([1, len(allWords)+1], dtype='int')
#transpose matrix so we can loop through each document
transpose = unigram_arr.transpose()
docCount = 0
#run perceptron algorithm on unigram representation to get w vector after first pass
for column in transpose:
    #data lifting
    lift = np.append(column, 1)
    dot = np.dot(w, lift)
    label = int(reviews_df.iloc[docCount]['label'])
    if label == 0: label = -1
    if dot * label <= 0:
        w = w + label*lift
    docCount += 1

#create unigram representation of test data
test_arr = createUnigram(test_df)
transpose_test = test_arr.transpose()
docCount = 0
correct_prediction = 0
#check if sign(w * x) = 1 and record results
for column in transpose_test:
    #datalifting
    lift = np.append(column, 1)
    ans = np.dot(w, lift)
    label = int(test_df.iloc[docCount]['label'])
    if label == 0: label = -1
    if ans * label > 0:
        correct_prediction += 1
    docCount += 1


print(len(transpose_test))
print("accuracy is", correct_prediction/len(transpose_test))


#
#initialize w vector to 0, add extra component at end for bias
w = np.zeros([1, len(allWords)+1], dtype='int')
transpose = tfidf_arr.transpose()
docCount = 0
#run perceptron algorithm on tf-idf representation to get w vector after first pass
for column in transpose:
    #data lifting
    lift = np.append(column, 1)
    dot = np.dot(w, lift)
    label = int(reviews_df.iloc[docCount]['label'])
    if label == 0: label = -1
    if dot * label <= 0:
        w = w + label*lift
    docCount += 1

#create tfidf representation of test data
test_arr_2 = createDocumentFrequency(test_arr)
transpose_test = test_arr_2.transpose()
docCount = 0
correct_prediction = 0
#check if sign(w * x) = 1 and record results
for column in transpose_test:
    #datalifting
    lift = np.append(column, 1)
    ans = np.dot(w, lift)
    label = int(test_df.iloc[docCount]['label'])
    if label == 0: label = -1
    if ans * label > 0:
        correct_prediction += 1
    docCount += 1


print(len(transpose_test))
print("accuracy is", correct_prediction/len(transpose_test))

