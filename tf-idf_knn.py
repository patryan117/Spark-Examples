
from __future__ import print_function

import sys
import re
from operator import add
import numpy as np
from numpy import dot
from numpy.linalg import norm

from pyspark import SparkContext


# The following function gets a list of dictionaryPos values, and then creates a TF vector
# corresponding to those values... for example, if we get [3, 4, 1, 1, 2] we would in the
# end have [0, 2/5, 1/5, 1/5, 1/5] because 0 appears zero times, 1 appears twice, 2 appears once, etc.

def buildArray (listOfIndices):
   	returnVal = np.zeros (20000)
   	for index in listOfIndices:
   		returnVal[index] = returnVal[index] + 1
   	mysum = np.sum (returnVal)
   	returnVal = np.divide (returnVal, mysum)
   	return returnVal

def stringVector (x):
	returnVal= str(x[0])
	for j in x[1]:
  		returnVal += ','+ str(j)
   	return returnVal
      

if __name__ == "__main__":
    if len(sys.argv) < 3:
        exit(-1)
	
    
    sc = SparkContext(appName="Assignment-2")

	# First load up all of the documents in the corpus
    corpus = sc.textFile(sys.argv[1])

    # Assumption: Each document is stored in one line of the text file. We need this count value for later ...
    numberOfDocs = corpus.count()

    # Each entry in validLines will be a line from the text file
    validLines = corpus.filter(lambda x: 'id' in x and 'url=' in x)

    # Now, we transform it into a set of (docID, text) pairs
    keyAndText = validLines.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))

    # Now, we split the text in each (docID, text) pair into a list of words. After this step, we have a data set with
    # (docID, ["word1", "word2", "word3", ...]) We use a regex here to make sure that the program does not break down on some of the documents

    regex = re.compile('[^a-zA-Z]')
    # # remove all non letter characters
    keyAndListOfWords = keyAndText.map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()))

    # Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])  to ("word1", 1) ("word2", 1)...
    allWords = keyAndListOfWords.flatMap(lambda item: item[1])

    # Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
    allCounts = allWords.map(lambda word: (word, 1)).reduceByKey(lambda x, y: x + y)

    # Get the top 20,000 words in a local array in a sorted format based on frequency
    topWords = allCounts.top(20000, lambda x:x[1])


    # We'll create a RDD that has a set of (word, dictNum) pairs start by creating an RDD that has the number 0 through 20000
    # 20000 is the number of words that will be in our dictionary
    twentyK = sc.parallelize(range(20000))

    # Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1) ("NextMostCommon", 2), ...
    # the number will be the spot in the dictionary used to tell us where the word is located
    dictionary = twentyK.map(lambda x: (topWords[x][0], x))

    # Next step, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]), ("word1", docID), ("word2", docId), ...
    allWords = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

    # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
    allDictionaryWords = allWords.join(dictionary)

    # Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
    justDocAndPos = allDictionaryWords.map(lambda x: (x[1][0], x[1][1]))

    # Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()

    # ['418384,27,8,8,8,13,10449,640,15243,46,3213,45,224,489,183,273,5841,63,32,28,12,12,4062,4062,4062,221,2406,604,2673,2673,161,88,989,228,228,228,228,228,241,241,241,241,241,241,406,25,450,450,114,12279,272,11,11,39,39,281,365,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14508,14508,14508,14508,1285,5,5,5,5,4665,10618,10618,10618,1077,76,5004,38,283,176,2,2,2,2,2,1258,1258,1258,16,22,22,434,20,2998,24,335,15800,247,103,4085,14,14,14,61,15,101,546,1676,381,229,984,16489,6748,134,134,16170,16170,16170,16170,2845,23,1081,33,316,2990,85,242,2761,4126,2858,1049,1352,94,14559,10,10,10,10,10,14650,4256,3646,3646,3646,75,5126,4,4,4,4,2843,2281,3684,53,601,270,3,3,3,3,3,3,3,474,2046,6153,3819,1,1,1,1,1,1,1,1,1,1,1,1,1,2615']

    # TODO: Uncomment this line to save the file.
    # forCSV.saveAsTextFile(sys.argv[2] + "_allDictionaryWordsInEachDoc")

    # The following line this gets us a set of (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    # and converts the dictionary positions to a bag-of-words numpy array...

    allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))

    # Now, create a version of allDocsAsNumpyArrays where, in the array, every entry is either zero or one.
    # A zero means that the word does not occur, and a one means that it does.

    def binarizeArray(in_list):
        out_list = np.where(in_list > 0, 1, 0)
        return out_list


    zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0], binarizeArray(x[1])))

    # Now, add up all of those arrays into a single array, where the i^th entry tells us how many
    # individual documents the i^th word in the dictionary appeared in
    dfArray = zeroOrOne.reduce(lambda x1, x2: (("", np.add(x1[1], x2[1]))))[1]

    # Create an array of 20,000 entries, each entry with the value numberOfDocs (number of docs)
    multiplier = np.full(20000, numberOfDocs)

    # Get the version of dfArray where the i^th entry is the inverse-document frequency for the
    # i^th word in the corpus
    idfArray = np.log(np.divide(multiplier, dfArray))

    # Finally, convert all of the tf vectors in allDocsAsNumpyArrays to tf * idf vectors
    allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))

    # Now we can save the the TF-IDF as file.
    # allDocsAsNumpyArraysTFidf.map(lambda x: stringVector(x)).saveAsTextFile(sys.argv[2]+"_allDocsAsNumpyArraysTFidf")
	
	############## TASK 3

	
	# Finally, we have a function that returns the prediction for the label of a string, using a kNN algorithm
    def getPrediction(textInput, k):
        myDoc = sc.parallelize(('', textInput))
        wordsInThatDoc = myDoc.flatMap(lambda x: ((j, 1) for j in regex.sub(' ', x).lower().split()))
        allDictionaryWordsInThatDoc = dictionary.join(wordsInThatDoc).map(lambda x: (x[1][1], x[1][0])).groupByKey()
        myArray = buildArray(allDictionaryWordsInThatDoc.top(1)[0][1])
        myArray = np.multiply(myArray, idfArray)
        distances = allDocsAsNumpyArraysTFidf.map(lambda x: (x[0], np.dot(x[1], myArray)))
        topK = distances.top(k, lambda x: x[1])
        docIDRepresented = sc.parallelize(topK).map(lambda x: (x[0], 1))
        numTimes = docIDRepresented.reduceByKey(add)
        return numTimes.top(1, lambda x: x[1])


    # print(getPrediction('God Jesus Allah', 10))
    # print('https://en.wikipedia.org/wiki?curid=' + str(getPrediction('God Jesus Allah', 1)[0][0]))
    #  On 1000 Docs small Data output will be https://en.wikipedia.org/wiki?curid=433978

    # print(getPrediction('Sport Basketball Volleyball Soccer', 30))
    # print('https://en.wikipedia.org/wiki?curid=' + str(getPrediction('Sport Basketball Volleyball Soccer', 1)[0][0]))
    # On 1000 Docs small Data output will be  https://en.wikipedia.org/wiki?curid=418388

    # print(getPrediction('How many goals Vancouver score last year?', 1))
    # print('https://en.wikipedia.org/wiki?curid=' + str(getPrediction('How many goals Vancouver score last year?', 1)[0][0]))
    # https://en.wikipedia.org/wiki?curid=454889


    a = sc.parallelize(getPrediction('God Jesus Allah', 10), 1)
    print(a.collect())
    a.saveAsTextFile(sys.argv[2]+"_answer1")

    b = sc.parallelize(getPrediction('Sport Basketball Volleyball Soccer', 30), 1)
    print(b.collect())
    b.saveAsTextFile(sys.argv[2]+"_answer2")

    c = sc.parallelize(getPrediction('How many goals Vancouver score last year?', 1))
    print(c.collect())
    c.saveAsTextFile(sys.argv[2]+"_answer3")


    sc.stop()
