# linreg.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
# 
# This code is used to implements multiple linear regression using the closed form expression 
# for the ordinary least squares estimate of the linear regression coefficients computed 
# using summation.
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv
#
#

import sys
import numpy as np

from pyspark import SparkContext

np.set_printoptions(precision=13)


#This function is to calculate the value of X * X-transpose 
def getXintoXTranspose(yxvalue):
  #replacing y value to 1 to get X matrix. And as the first element in X matrix is 1.
  yxvalue[0]=1.0
  x_value = np.array(yxvalue).astype('float')
  #matrix of X will be np.asmatrix(x_value).T and X transpose will be np.asmatrix(x_value)
  X = np.asmatrix(x_value).T
  #X_transpose = np.asmatrix(x_value)
  X_into_Xt = np.dot(X,X.T)
  return X_into_Xt

#This function is to calculate the value of X * Y
def getXintoY(yxvalue):
  # extracting y value and replacing y value by 1 to get X matrix
  y_value = float(yxvalue[0])
  yxvalue[0] = 1.0
  x_value = np.array(yxvalue).astype('float')
  X = np.asmatrix(x_value).T
  X_into_y = np.multiply(X,y_value)
  return X_into_y
  

if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])

  yxlines = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = yxlines.first()
  yxlength = len(yxfirstline)
  #print "yxlength: ", yxlength

  # dummy floating point array for beta to illustrate desired output format
  beta = np.zeros(yxlength, dtype=float)

  #
  # Add your code here to compute the array of 
  # linear regression coefficients beta.
  # You may also modify the above code.

  # Calculating the value of X * X-transpose in map function and then adding all the values using reduceBYKey function.   
  X_Xt_sum = np.asmatrix(yxlines.map(lambda l: ("key_X_Xt",getXintoXTranspose(l))).reduceByKey(lambda x1,x2: np.add(x1,x2)).map(lambda l: l[1]).collect()[0])
  #print X_Xt_sum
  
  # Calculating the value of X * X-transpose in map function and then adding all the values using reduceBYKey function.
  X_y_sum = np.asmatrix(yxlines.map(lambda l: ("key_X_y",getXintoY(l))).reduceByKey(lambda x1,x2: np.add(x1,x2)).map(lambda l: l[1]).collect()[0])
  #print X_y_sum
 
  # Taking X_Xt_sum inverese and mutliplying it with X_y_sum for getting the beta the coefficients
  #print np.linalg.inv(X_Xt_sum)
  beta_values = np.dot(np.linalg.inv(X_Xt_sum),X_y_sum)
  # Converting the matrix to list
  beta = np.array(beta_values).tolist()
  #print beta


  # print the linear regression coefficients in desired output format
  print "beta coefficient: "
  for coeff in beta:
      print coeff

  sc.stop()
