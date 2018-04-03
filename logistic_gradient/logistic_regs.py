import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
   
def sigmoid(x):
  return 1 / (1 + math.exp(-x))    

# method for logistc
def logistic_regression(df, steps, eta):
    # w0, w1, w2, w3, ..... w60 so total 61 weights 
    weights = [float(0.5) for i in range(len(df.columns)) ]


    print(len(weights))
    
    for i in range(len(df.index)):
    
        # adding the first weights to the list
        fx = weights[0]
        
        for y in range(len(weights) - 1):
            # w1 corresponds i+1
            fx += (weights[y+1] * df.iloc[i,y])
  
        y = sigmoid(fx)
        
        # updating new weights
        

def main():    
    
    # making pandas data frame
    df = pd.read_csv('sonar.csv')
    x = {'Mine' : 1, 'Rock' : 2}
    df.iloc[:,60] = df.iloc[:,60].map(x)
    
    # value for eta
    eta = [0.001,0.01,0.05,0.1,0.5,1.0,1.5]
    
    for i in range(len(eta)):
        # calling method for logistic regression
        logistic_regression(df, 50, eta[0])
    
if __name__ == "__main__":
    main()











