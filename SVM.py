import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#Read the data from the csv file
df = pd.read_csv("SVM_data.csv")

#The learning rate
learning_rate = 0.01

#The weight of the classification term in the error function
cls_error = 1

#The weight of the distance term in the error function
dist_error = 1

#The number of iterations of gradient descent.
#TODO: Optimize number of rounds needed to train.
N = 30000

#The initial range of random numbers the coefficients, a,b,c can take on.
#The range is from 0 to the mean value of the x and y data points.
#TODO: Improve the range of the initial random numbers
a = random.uniform(0,df.mean()[0])
b = random.uniform(0, df.mean()[1])
c = random.uniform(0, df.mean()[1])

def error_function(x,y,label,a,b,c):
	value = a * x + b * y + c
	
	if label == 1:
		value = value + 1
		if value > 0:
			a = a - learning_rate * (x + 2 * a)
			b = b - learning_rate * (y + 2 * b)
			c = c - learning_rate
		elif value == 0:
			a = a - learning_rate * (x + 2 * a)
			b = b - learning_rate * (y + 2 * b)
			c = c - learning_rate
			
	if label == 0:
		value = value - 1
		if value < 0:
			a = a + learning_rate * (x + 2 * a)
			b = b + learning_rate * (y + 2 * b)
			c = c + learning_rate
		elif value == 0:
			a = a + learning_rate * (x + 2 * a)
			b = b + learning_rate * (y + 2 * b)
			c = c + learning_rate
	return (a,b,c)
	
#The learning process:  
for i in range(N):
	#Pick random point
	rand = random.randint(0,df.shape[0]-1)
	points = df.iloc[rand,:]
	x = points[0]
	y = points[1]
	label = points[2]
	
	updated_coefficients = error_function(x,y,label,a,b,c)
	a = updated_coefficients[0]
	b = updated_coefficients[1]
	c = updated_coefficients[2]

x = df.iloc[:,0]
y = df.iloc[:,1]
label = df.iloc[:,2]
color=['red' if l == 1 else 'blue' for l in label]
plt.scatter(x,y,color=color)
plt.plot(x,-(a/b)*x - (c/b),color='black')
plt.show()
