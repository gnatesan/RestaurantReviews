import pandas as pd
import numpy as np
import math 
import random
from statistics import mean
from scipy import sparse

reviews_df = pd.read_csv("restaurant_reviews_data/reviews_tr.csv")


test_df = pd.read_csv("restaurant_reviews_data/reviews_te.csv")

#reviews_df = reviews_df.head(100000)
test_df = test_df.head(15000)
reviews_df = reviews_df.sample(frac =.001)

#list of dictionaries, each dictionary maps a word to its frequency in that document
allUnigrams = []

#dictionary that maps word to word number
allWords = {}

#dictionary that maps bigram to bigram number
allBigrams = {}


#dictionary that maps the word number to the log(idf) component of tf-df
documentFrequency = {}

def addToUnigram(unigram_arr, text, colIndex):
    #build the bigram dictionary for the specific document
    for word in text.split(): 
        if word in allWords:  
          wordNum = allWords[word] 
          #increase the frequency count in the unigram 2d array
          unigram_arr[wordNum][colIndex] += 1
    return unigram_arr

def addToBigram(bigram_arr, text, colIndex):
    #build the bigram dictionary for the specific document
    test2 = text.split()
    max = len(test2) - 1
    for i in range(max):
        a = (test2[i], test2[i+1])
        if a in allBigrams:
          bigramNum = allBigrams[a]
          bigram_arr[bigramNum][colIndex] += 1
    return bigram_arr


def addToWordDictionary(text):
    count = len(allWords)
    for word in text.split():
        if word not in allWords:
            allWords[word] = count
            count += 1

def createWordDictionary(df):
    for i, row in df.iterrows():
        addToWordDictionary(row['text'])

def addToWordDictionary2(arr):
    count = len(allWords)
    for word in arr:
      if word not in allWords:
        allWords[word] = count
        count += 1
      #else: print("not unique", word, allWords[word], count)

def createWordDictionary2(tr_df, te_df):
    out_arr = []
    #loop through training data and create list with all words
    for i, row in tr_df.iterrows():
        #addToWordDictionary(row['text'])
        out_arr.extend(row['text'].split())
    #get the unique words from the list
    new_arr, count = np.unique(out_arr, return_counts=True)
    #print("array with duplicates length is", len(out_arr))
    #print("array without duplicates length is", len(new_arr))
    indexes = []
    for i in range(len(new_arr)):
        if (count[i] < 1000):
            indexes.append(i)
    new_arr = np.delete(new_arr, indexes, axis=0)
    del indexes
    #print("array after removing low occuring features length is", len(new_arr))
    del out_arr
    #addToWordDictionary2(new_arr)
    del new_arr

def createWordDictionary3(tr_df, te_df):
    out_arr = []
    #loop through training data and create list with all words
    for i, row in tr_df.iterrows():
        out_arr = (row['text'].split())
        #get the unique words from the list
        new_arr, count = np.unique(out_arr, return_counts=True)
        addToWordDictionary2(new_arr)
        del new_arr

def createUnigram(df):
    arr = np.zeros([len(allWords), len(df)], dtype=int)
    count = 0
    for i, row in df.iterrows():
        #print(i)
        arr = addToUnigram(arr, row['text'], count) 
        count += 1
    return arr

def createBigram(df):
    arr = np.zeros([len(allBigrams), len(df)], dtype=int)
    count = 0
    for i, row in df.iterrows():
        #print(i)
        arr = addToBigram(arr, row['text'], count) 
        count += 1
    return arr

def addToBigramDictionary(bigram):
    count = len(allBigrams)
    if (bigram not in allBigrams):
      allBigrams[bigram] = count
    
def createBigramDictionary(df):
    for i, row in df.iterrows():
        test2 = row['text'].split()
        max = len(test2) - 1
        for i in range(max):
            a = (test2[i], test2[i+1])
            #print(a)
            addToBigramDictionary(a)

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

createBigramDictionary(reviews_df)
#print(len(allBigrams))
#print(allBigrams)
bigram_arr = createBigram(reviews_df)
bigram_arr = sparse.csr_matrix(bigram_arr)

createWordDictionary2(reviews_df, test_df)
#we need to get the unseen words in test data and add them to our word dictionary
#createWordDictionary(test_df) 
print(len(allWords))
unigram_arr = createUnigram(reviews_df)
unigram_arr = sparse.csr_matrix(unigram_arr)
tfidf_arr = createDocumentFrequency(unigram_arr.toarray())
tfidf_arr = sparse.csr_matrix(tfidf_arr)

#initialize w vector to 0
w = np.zeros([1, len(allWords)+1], dtype='int')
unigram_arr = unigram_arr.toarray()
transpose = unigram_arr.transpose()
r = list(range(len(transpose)))
random.shuffle(r)
docCount = 0
#run perceptron algorithm on unigram representation to get w vector after first pass
for i in r:
    lift = np.append(transpose[i], 1)
    dot = np.dot(w, lift)
    label = int(reviews_df.iloc[i]['label'])
    if label == 0: label = -1
    if dot * label <= 0:
        w = w + label*lift
    docCount += 1

w_final = np.zeros([1, len(allWords)+1], dtype='int')
#fin = []
random.shuffle(r)
docCount = 0
#run perceptron algorithm on unigram representation to get w vector after second pass
for i in r:
    lift = np.append(transpose[i], 1)
    dot = np.dot(w, lift)
    label = int(reviews_df.iloc[i]['label'])
    if label == 0: label = -1
    if dot * label <= 0:
        w = w + label*lift
    w_final = w_final + w
    #fin.append(w)
    docCount += 1
unigram_arr = sparse.csr_matrix(unigram_arr)
w_final = w_final / docCount
print(w)
print(w_final)

ind = np.argpartition(w_final[0], -10)[-10:]
key_list = list(allWords.keys())
for i in ind:
  print(key_list[i])

idx = np.argpartition(w_final[0], 10)

for i in range(10):
  print(key_list[idx[i]])

#create unigram representation of test data
test_arr = createUnigram(test_df)
transpose_test = test_arr.transpose()
docCount = 0
correct_prediction = 0
#check if sign(w_final * x) = 1 and record results
for column in transpose_test:
    lift = np.append(column, 1)
    ans = np.dot(w_final, lift)
    label = int(test_df.iloc[docCount]['label'])
    if label == 0: label = -1
    if ans * label > 0:
        correct_prediction += 1
    docCount += 1
test_arr = sparse.csr_matrix(test_arr)

print(len(transpose_test))
print("accuracy of unigram is", correct_prediction/len(transpose_test))

#initialize w vector to 0
w = np.zeros([1, len(allBigrams)+1], dtype='int')
bigram_arr = bigram_arr.toarray()
transpose = bigram_arr.transpose()
r = list(range(len(transpose)))
random.shuffle(r)
docCount = 0
#run perceptron algorithm on unigram representation to get w vector after first pass
for i in r:
    lift = np.append(transpose[i], 1)
    dot = np.dot(w, lift)
    label = int(reviews_df.iloc[i]['label'])
    if label == 0: label = -1
    if dot * label <= 0:
        w = w + label*lift
    docCount += 1

w_final = np.zeros([1, len(allBigrams)+1], dtype='int')
random.shuffle(r)
docCount = 0
#run perceptron algorithm on bigram representation to get w vector after second pass
for i in r:
    lift = np.append(transpose[i], 1)
    dot = np.dot(w, lift)
    label = int(reviews_df.iloc[i]['label'])
    if label == 0: label = -1
    if dot * label <= 0:
        w = w + label*lift
    w_final = w_final + w
    #fin.append(w)
    docCount += 1
bigram_arr = sparse.csr_matrix(bigram_arr)
w_final = w_final / docCount
print(w)
print(w_final)

#create bigram representation of test data
test_arr = createBigram(test_df)
transpose_test = test_arr.transpose()
docCount = 0
correct_prediction = 0
#check if sign(w_final * x) = 1 and record results
for column in transpose_test:
    lift = np.append(column, 1)
    ans = np.dot(w_final, lift)
    label = int(test_df.iloc[docCount]['label'])
    if label == 0: label = -1
    if ans * label > 0:
        correct_prediction += 1
    docCount += 1
test_arr = sparse.csr_matrix(test_arr)

print(len(transpose_test))
print("accuracy of bigram is", correct_prediction/len(transpose_test))

#initialize w vector to 0, add extra component at end for bias
w = np.zeros([1, len(allWords)+1], dtype='int')
tfidf_arr = tfidf_arr.toarray()
transpose = tfidf_arr.transpose()
docCount = 0
random.shuffle(r)
#run perceptron algorithm on tf-idf representation to get w vector after first pass
for i in r:
    #data lifting
    lift = np.append(transpose[i], 1)
    dot = np.dot(w, lift)
    label = int(reviews_df.iloc[docCount]['label'])
    if label == 0: label = -1
    if dot * label <= 0:
        w = w + label*lift
    docCount += 1

w_final = np.zeros([1, len(allWords)+1], dtype='int')
#fin = []
random.shuffle(r)
docCount = 0
#run perceptron algorithm on tf-idf representation to get w vector after second pass
for i in r:
    lift = np.append(transpose[i], 1)
    dot = np.dot(w, lift)
    label = int(reviews_df.iloc[i]['label'])
    if label == 0: label = -1
    if dot * label <= 0:
        w = w + label*lift
    w_final = w_final + w
    #fin.append(w)
    docCount += 1
tfidf_arr = sparse.csr_matrix(tfidf_arr)
w_final = w_final / docCount
print(w)
print(w_final)

test_arr = createUnigram(test_df)
#create tfidf representation of test data 
test_arr_2 = createDocumentFrequency(test_arr)
transpose_test = test_arr_2.transpose()
docCount = 0
correct_prediction = 0
#check if sign(w * x) = 1 and record results
for column in transpose_test:
    #datalifting
    lift = np.append(column, 1)
    ans = np.dot(w_final, lift)
    label = int(test_df.iloc[docCount]['label'])
    if label == 0: label = -1
    if ans * label > 0:
        correct_prediction += 1
    docCount += 1

print(len(transpose_test))
print("accuracy of tfidf is", correct_prediction/len(transpose_test))
