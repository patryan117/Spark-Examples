
from __future__ import print_function
from pyspark import SparkContext
import sys
from operator import add




if __name__ == "__main__":

    def isfloat(value):
        try:
            float(value)
            return True
        except:
            return False


    def isInt(value):
        try:
            int(value)
            return True
        except:
            return False

    sc = SparkContext(appName="assignment_1")


    raw_text = sc.textFile(sys.argv[1])

    parsed_text = raw_text.map(lambda x: (x.split(',')))

    # remove lines if they don't have 16 values
    def correctRows(p):
        if (len(p) == 17):
            if (len(p[3]) == 19):
                if (isfloat(p[4]) and isfloat(p[5]) and isfloat(p[12]) and isfloat(p[16])):
                    if (float(p[4]) != 0.0 and float(p[5]) != 0.0 and float(p[12]) != 0.0 and float(p[16]) != 0.0):
                        return p



    clean_data = parsed_text.filter(correctRows)



    # QUESTION 1:  TOP OF UNIQUE DRIVERS ID'S PER MEDALLION  (MEDALLION : COUNT OF UNIQUE DRIVER ID'S)

    answer_1 = clean_data\
        .map(lambda x: (x[0],x[1]))\
        .distinct()\
        .map(lambda x: (x[0], 1))\
        .reduceByKey(add) \
        .takeOrdered(10, key=lambda x: -x[1])



    #  QUESTION 2:  10 BEST DRIVERS IN TERMS OF AVG. PRICE PER MINUTE (DRIVER ID, AVERAGE MONEY PER MINUTE)

    answer_2 = clean_data.map(lambda x: (x[1], (float(x[16]) , float(x[4])/60))) \
        .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))  \
        .map(lambda (x, (y, z)): (x, y / z)) \
        .takeOrdered(10, key=lambda x: -x[1])




    #  QUESTION 3:  HOUR OF DAY WHICH YEILD'S DRIVERS THE HIGHEST SURCHARGE/MILE RATE

    answer_3 = clean_data.map(lambda x: (x[2][-8:-6], (float(x[12]) , float(x[5])))) \
        .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
        .map(lambda (x, (y, z)): (x, y / z)) \
        .takeOrdered(10, key=lambda x: -x[1])



    var1 = sc.parallelize(answer_1, 1)
    var1.saveAsTextFile(sys.argv[2])

    var1 = sc.parallelize(answer_2, 1)
    var1.saveAsTextFile(sys.argv[3])

    var1 = sc.parallelize(answer_3, 1)
    var1.saveAsTextFile(sys.argv[4])

    sc.stop()
