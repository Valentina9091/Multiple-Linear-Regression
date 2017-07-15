
			Multiple Linear Regression
=========================================================================================
This code implements multiple linear regression using the closed form expression 
for the ordinary least squares estimate of the linear regression coefficients computed 
using summation.

It consists of 2 python files 
	- 	linreg.py
	- 	linreg_gradient.py

	
Note:
Instructions to create input directory:
-	Create folder in hdfs
	hadoop fs -mkdir /user/cloudera/input

-	Copy all the input file from local to hdfs input directory
	hadoop fs -put <source> <destination>
	hadoop fs -put /local/input/file/path /user/cloudera/input

Steps to execute linear regression:
	spark-submit linreg.py /hdfs/path/to/inputfile > output_file 
	
Note: output_file -> consist output as beta co-efficients.

=========================================================================================
				Gradient descent
=========================================================================================
Steps to execute linear regression:
	spark-submit linreg_gradient.py /hdfs/path/to/inputfile step-size iterations > output_file 
	
	spark-submit linreg_gradient.py /user/cloudera/input.csv 0.01 1000 > output_file 

Note: output_file -> consist output as beta co-efficients.
initial step-size is assumed to be 0.01

