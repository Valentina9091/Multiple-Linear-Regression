# linreg_gradient.py
#
# This code computes the gradient descent approach for
#linear regression
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile> <step-size> <alpha>
# Example usage: spark-submit linreg.py yxlin.csv 0.01 1000
#
# 


import sys
import numpy as np

from pyspark import SparkContext

np.set_printoptions(precision=13)


if __name__ == "__main__":
  if len(sys.argv) !=4:
    print >> sys.stderr, "Usage: linreg <datafile> <alpha/step-size> <iterations>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])

  yxlines = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = yxlines.first()
  yxlength = len(yxfirstline)

  # dummy floating point array for beta to illustrate desired output format
  beta = np.zeros(yxlength, dtype=float)
  # initializing step-size i.e alpha and iterations
  alpha = float(sys.argv[2])
  iteration = int(sys.argv[3])
  
  yxlines = np.array(yxlines.collect(),dtype='float64')
  Y = yxlines[:,[0]]
  X = yxlines[:,1: ]
  
  num_rows,num_cols=X.shape
  num_rows2,num_cols2=yxlines.shape
  b = np.zeros(shape=(num_rows,1))
  b.fill(1.0)

  beta=np.zeros(shape=(num_cols2,1))

  X=np.column_stack((b,X))
  Xt=X.transpose()
  
  orig_beta =beta
  restart_iteration = True
  while restart_iteration :
    restart_iteration  = False
    for i in range(0, iteration): 
      #step_size=i+1
      temp_beta=beta
      value = np.dot(X,beta)
      value = np.subtract(Y,value)
      value = np.multiply(alpha,np.dot(Xt,value))
      beta = np.add(beta,value)
      #print i
      if np.array_equal(beta,temp_beta):
        print "Converged at iteration number: %d when step size / step length = %f" % (i+1 , alpha)
        print "beta coefficient: "
        for coeff in beta:
          print coeff
        break
      if np.isnan(beta).any() or np.isinf(beta).any():
         restart_iteration = True
         alpha =alpha /10
         beta =orig_beta    
         break;
  sc.stop()
