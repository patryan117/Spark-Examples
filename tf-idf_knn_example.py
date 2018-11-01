
from __future__ import print_function

import sys
import re
from operator import add
import numpy as np
from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext




#############################################################################################################################
# BASIC SPARK TF-IDF FOR WIKIPEDIA, DESIGNED FOR RUNNING ON AWS (EMR) / DATAPROC (GOOGLE CLOUD) 
#############################################################################################################################

#TODO: Implement stopword removal and stemming




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

    corpus = sc.textFile(sys.argv[1])

    numberOfDocs = corpus.count()

    validLines = corpus.filter(lambda x: 'id' in x and 'url=' in x)
    keyAndText = validLines.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))

    regex = re.compile('[^a-zA-Z]')
    keyAndListOfWords = keyAndText.map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()))

    allWords = keyAndListOfWords.flatMap(lambda item: item[1])
    allCounts = allWords.map(lambda word: (word, 1)).reduceByKey(lambda x, y: x + y)
    topWords = allCounts.top(20000, lambda x:x[1])
    twentyK = sc.parallelize(range(20000))
    dictionary = twentyK.map(lambda x: (topWords[x][0], x))
    allWords = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
    allDictionaryWords = allWords.join(dictionary)
    justDocAndPos = allDictionaryWords.map(lambda x: (x[1][0], x[1][1]))

    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
    allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))

	
    def binarizeArray(in_list):
        out_list = np.where(in_list > 0, 1, 0)
        return out_list

    zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0], binarizeArray(x[1])))
    dfArray = zeroOrOne.reduce(lambda x1, x2: (("", np.add(x1[1], x2[1]))))[1]
    multiplier = np.full(20000, numberOfDocs)
    idfArray = np.log(np.divide(multiplier, dfArray))
    allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))
	
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


    # EXAMPLE SEARCHES:
    a = sc.parallelize(getPrediction('President Lincoln Hat', 10), 1)
    print(a.collect())
    a.saveAsTextFile(sys.argv[2]+"_answer1")

    b = sc.parallelize(getPrediction('Pear Harbor Bombing', 30), 1)
    print(b.collect())
    b.saveAsTextFile(sys.argv[2]+"_answer2")

    c = sc.parallelize(getPrediction('Banksy Street Art', 1))
    print(c.collect())
    c.saveAsTextFile(sys.argv[2]+"_answer3")


    sc.stop()

	
    
