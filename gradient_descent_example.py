


from __future__ import print_function
import sys
import numpy as np
from numpy import *
from scipy import stats
import spark
from pyspark import SparkContext, SQLContext
from pyspark.sql import *



def main():


    ########### TASK 1 ###############################################################################################

    # CALCULATE M_HAT AND B_HAT USING SERIES BASED METHOD...

    ###########################################################################################################3######



    sc = SparkContext(appName="assignment_3")

    points = readData(sys.argv[1], 5.0, 100.0)
    xrdd = sc.parallelize(points[0])
    yrdd = sc.parallelize(points[1])
    n = len(points[0])


    #TODO: fix so it doenst need to map twice
    # zips the two RDD's so that each item becomes a tuple of (trip distance(i), total revenue(i))
    # xyrdd = yrdd.zipWithIndex().map(lambda x: (x[1], x[0])).join(xrdd.zipWithIndex().map(lambda y: (y[1], y[0]))).map(lambda x: x[1])
    # xyrdd = xyrdd.map(lambda x: (x[1], x[0]))

    xyrdd = xrdd.zip(yrdd)
    # print(xyrdd.take(5))


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


    # def calc_b_hat(xyrdd, xrdd, yrdd, n):
    #
    #     # compute b_hat (the predicted y intercept)
    #     a = ((xrdd.map(lambda x: x**2).sum()) *  yrdd.sum())
    #     b = xrdd.sum() * (xyrdd.map(lambda x: x[0] * x[1]).sum())
    #     c = n * (xrdd.map(lambda x: x**2).sum())
    #     d = xrdd.sum() ** 2
    #     b_hat =  (a - b) / (c - d)
    #     return(b_hat)





    m_hat = (calc_m_hat(xyrdd))
    b_hat = (calc_b_hat(xyrdd))

    print("\n")
    print("Solution 1: ")
    print("m_hat", m_hat)
    print("b_hat", b_hat)
    print("\n")






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




    print("samp_m_hat_list: ", samp_m_hat_list)
    print("samp_b_hat_list: ", samp_b_hat_list)

    mean_m_hat = (sum(samp_m_hat_list) / len(samp_m_hat_list))
    mean_b_hat = (sum(samp_b_hat_list) / len(samp_b_hat_list))




    print("\n")
    print("Solution 2: ")
    print("mean m hat: ", mean_m_hat)
    print("meah b hat: ", mean_b_hat)
    print("\n")








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

        xyrdd = sc.parallelize(points.takeSample(False, 1000))

        xrdd = xyrdd.map(lambda x: (x[0]))
        yrdd = xyrdd.map(lambda x: (x[1]))

        m_current = float(m_current)
        b_current = float(b_current)
        learning_rate = float(learning_rate)



        while (previous_step_size > precision and iters < max_iterations):


            y_current = xrdd.map(lambda x: (x * m_current) + b_current)
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


            if (iters % 1 == 0):
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
        print("Solution 3 Check: ")
        print("Gradient_m_hat_check", calc_m_hat(xyrdd))
        print("Gradient_b_hat_check", calc_b_hat(xyrdd))
        print("\n")


        return(m_current, b_current)




    batch_gradient_descent(points=xyrdd, m_current=2, b_current=2, learning_rate=0.001, max_iterations = 10000,  precision = 0.000001 )



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


def readData(fileName, lowFilter, highFilter):   # why are uppper and lower limit filters applied?
    x = []
    y = []
    with open(fileName) as f:
        for i in f:
            a = i.split(",")
            if (len(a) == 17):
                if (isfloat(a[5]) and isfloat(a[11])):
                    if (float(a[5]) != 0 and float(a[11]) != 0):
                        if (float(a[11]) > lowFilter and float(a[11]) < highFilter):
                            x.append(float(a[5]))  # trip distance0
                            y.append(float(a[11]))  # fare amount

    ax = np.array(x)
    ay = np.array(y)

    return np.vstack((ax, ay))







if __name__ == "__main__":
    main()


