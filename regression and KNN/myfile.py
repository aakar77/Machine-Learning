import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



 # reading train data
dataset_train = pd.read_csv('auto_train.csv')

# reading the  train data 
displacement_train = dataset_train.iloc[:,0].values
mpg_train = dataset_train.iloc[:,2].values
    
# reading test data
dataset_test = pd.read_csv('auto_test.csv')
    
#reading the test data
displacement_test = dataset_test.iloc[:,0].values
mpg_test = dataset_test.iloc[:,2].values

#reading the test data
displacement_test = dataset_test.iloc[:,0].values
mpg_test = dataset_test.iloc[:,2].values

print(type(mpg_test))            

plt.plot(displacement_train, mpg_train, 'ro')
plt.xlabel("displacement")
plt.ylabel("mpg")
plt.show()