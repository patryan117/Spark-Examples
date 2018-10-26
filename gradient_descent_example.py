

# spark submit option
# cd /home/pryan/Desktop/spark/bin/
# spark-submit  /home/pryan/Desktop/assignment_3/gradient.py  /home/pryan/Desktop/assignment_3/taxi.csv


from __future__ import print_function
import sys
from numpy import *
from pyspark import SparkContext, SQLContext

sc = SparkContext(appName="assignment_3")





def main():

    

    ########### TASK 1 ###############################################################################################

    # CALCULATE M_HAT AND B_HAT USING SERIES BASED METHOD...

    ###########################################################################################################3######

    
    
    # cleaning up data
    lines = sc.textFile(sys.argv[1])

    taxilines = lines.map(lambda x: x.split(','))

    # Exception Handling  and removing wrong data lines
    def isfloat(value):
        try:
            float(value)
            return True
        except:
            return False

    # remove lines if they don't have 16 values
    def correctRows(p):
        if (len(p) == 17):
            if (isfloat(p[5]) and isfloat(p[11])):
                if (float(p[5]) != 0 and float(p[11]) != 0):
                    if (float(p[5]) > 0 and float(p[5]) < 100):
                         if (float(p[11]) > 5 and float(p[11]) < 100):
                            return p


    xyrdd = taxilines.filter(correctRows).map(lambda x: (float(x[5]), float(x[11])))



    def calc_m_hat(xyrdd):

        try:
            n = xyrdd.count()
        except ValueError:
            print("Target is not an rdd")

            try:
                n = len(xyrdd)
            except ValueError:
                print("Target is not a list")

        a = n * (xyrdd.map(lambda x: x[0] * x[1]).sum())
        b = xyrdd.map(lambda x: x[0]).sum() * xyrdd.map(lambda x: x[1]).sum()
        c = n * (xyrdd.map(lambda x: x[0] ** 2).sum())
        d = xyrdd.map( lambda x : x[0]).sum() ** 2
        m_hat = (a - b) / (c - d)
        return(m_hat)



    def calc_b_hat(xyrdd):

        try:
            n = xyrdd.count()
        except ValueError:
            print("Target is not an rdd")

            try:
                n = len(xyrdd)
            except ValueError:
                print("Target is not a list")


        # compute b_hat (the predicted y intercept)
        a = (xyrdd.map(lambda x: x[0]**2).sum()) * xyrdd.map(lambda x: x[1]).sum()
        b = xyrdd.map(lambda x: x[0]).sum() * (xyrdd.map(lambda x: x[0] * x[1]).sum())
        c = n * (xyrdd.map(lambda x: x[0] ** 2).sum())
        d = xyrdd.map( lambda x: x[0]).sum() ** 2
        b_hat =  (a - b) / (c - d)
        return(b_hat)






    ########## Task 2  ############################################################################################

    # ESTIMATION BASED ON (5) SAMPLES OF 1000

    ##############################################################################################################



    samp_m_hat_list = []
    samp_b_hat_list = []


    for i in range(5):

        samp_xyrdd = sc.parallelize(xyrdd.takeSample(False, 1000))
        samp_xrdd = samp_xyrdd.map(lambda x: (x[0]))
        samp_yrdd = samp_xyrdd.map(lambda x: (x[1]))
        samp_n = samp_xyrdd.count()

        samp_m_hat_list.append(calc_m_hat(samp_xyrdd))
        samp_b_hat_list.append(calc_b_hat(samp_xyrdd))









    ########## Task 3  ###############################################################################################

    # ESTIMATION BASED ON 5 SAMPLES OF 1000

    ###################################################################################################################


    def batch_gradient_descent(points, m_current, b_current, learning_rate, max_iterations, precision):


        m_current = float(m_current)
        b_current = float(b_current)
        learning_rate = float(learning_rate)
        max_iterations = float(max_iterations)
        precision = float(precision)

        n = float(1000)
        iters = 0

        previous_step_size = float(1)
        points = sc.parallelize((points))


        m_current = float(m_current)
        b_current = float(b_current)
        learning_rate = float(learning_rate)



        while (previous_step_size > precision and iters < max_iterations):

            xyrdd = sc.parallelize(points.takeSample(False, 1000))

            y_current = xyrdd.map(lambda x: (x[0] * m_current) + b_current)

            m_prev = m_current

            joined = xyrdd.zip(y_current).map(lambda x: (x[0][0], x[0][1], x[1]))   # zip as ((x, y), y_current)

            b = joined.map(lambda x:  x[0] * (x[1] - x[2])).sum()
            c = joined.map(lambda x:  x[1] - x[2]).sum()

            m_gradient = (-(float(2) / n)) * b
            b_gradient = (-(float(2) / n)) * c

            new_learning_rate = learning_rate

            m_current = m_current - (new_learning_rate * m_gradient)
            b_current = b_current - (new_learning_rate * b_gradient)
            previous_step_size = abs(m_current - m_prev)
            iters = iters + 1


            if (iters % 10 == 0):
                print("Iteration: ", iters,  " b_current: ", "{0:.10f}".format(b_current), "m_current: ", "{0:.10f}".format(m_current),
                "previous_step_size: ", "{0:.10f}".format(previous_step_size))

                # print("m_gradient :", m_gradient)
                # print("b_gradient :", b_gradient)
                # print("precision", precision)

        print("\n")
        print("Solution 3: ")
        print("Gradient_m_hat", m_current)
        print("Gradient_b_hat", b_current)
        print("\n")


        temp = list([m_current, b_current])
        answer = sc.parallelize(temp)
        answer.saveAsTextFile(sys.argv[2])

        return (m_current, b_current)



    batch_gradient_descent(points=xyrdd.takeSample(False, 100000), m_current=2, b_current=2, learning_rate=0.001, max_iterations = 10000,  precision = 0.000001)



    sc.stop()





#####################################################################################################################
#####################################################################################################################
#####################################################################################################################



def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False


def readData(fileName, lowFilter, highFilter):  
    x = []
    y = []
    with open(fileName) as f:
        for i in f:
            a = i.split(",")
            if (len(a) == 17):
                if (isfloat(a[5]) and isfloat(a[11])):
                    if (float(a[5]) != 0 and float(a[11]) != 0):
                        if (float(a[11]) > lowFilter and float(a[11]) < highFilter):
                            x.append(float(a[5]))  # trip distance
                            y.append(float(a[11]))  # fare amount


    return (x, y)





if __name__ == "__main__":
    main()


