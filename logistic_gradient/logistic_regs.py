import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
   
def sigmoid(x):
    
    # changing if x goes less  than e^-16
    if( x < math.exp(-16)):
        x = math.exp(-16)
    
    y =  1 / (1 + math.exp(-x)) 
    return y  

def cross_entropy(y, r):
    
    z = 1-y
    
    if(z < math.exp(-16)):
        z = math.exp(-16)
        
    s = (r * math.log(y) + (1-r) * math.log(z))
    return s

def cross_entropy_2(y, r):
    
    s = float(0)
        
    for i in range(0, len(y)):
    
        z = 1 - y[i][0]
       
    
        if(z < math.exp(-16)):
            z = math.exp(-16)
        
        s += (r[i][0] * math.log(y[i][0]) + (1-r[i][0]) * math.log(z))

    return ((-1) * s)



def classification_error(Y, R):
    c=0
    for i in range(0,len(Y)):
        
        #print(Y[i]," : ",R[i][0])
        
        if(Y[i] != R[i]):
           c+=1
    #print(c)
    
    return (c / len(Y)) * 100

def plot_graph(X,Y):
    pass
    
    
    

# method for logistc
def logistic_regression(X, steps, eta, R):
    
    
    # w0, w1, w2, w3, ..... w60 so total 61 weights 
    w0 = 0.5
    W = np.full((60,1), 0.5)
    cross_entropy_list = []
    
    # making vectorize sigmoid_function
    sigmoid_vector = np.vectorize(sigmoid)
    cross_entropy_vector = np.vectorize(cross_entropy)
    
    
    #print(X.shape)
    #print(W.shape)
    #print(np.dot(X,W) + w0)
    
    # gradient descent 50 rounds
    for i in range(0, steps):
        
        entropy_list = []
        
        # taking product of [180,60] * [60,1] gives [60,1]
        # adding w0 to above
        Z = np.dot(X,W) + w0
    
        Y = sigmoid_vector(Z)    
        
        # taking difference between R - Y
        diff = R - Y
        
        # updating a new w0
        w0 = w0 + (diff * eta)
        
        # temp will be a matrix of 180
        sum_prod = np.dot(X.T, diff) * eta
        
        # updating the W, taking transpose bec sum_prod is of form [1, 60]
        W = W + sum_prod
        
        # calculating intermediate cross entropy for each iteration
        entropy_error = ((cross_entropy_vector(Y, R).sum(axis = 0))[0] * (-1))
        
        cross_entropy_list.append(entropy_error)
        
        
    print(len(cross_entropy_list))
        
    # with final values of w0 and w1,w2,w3 ... w60
    Z = np.dot(X,W) + w0    
    Y = sigmoid_vector(Z)


    Y_class =  map(lambda x: 1 if x > 0.5 else 0, Y)
    
    entropy_error = ((cross_entropy_vector(Y, R).sum(axis = 0))[0] * (-1))
    
    # Debugg the cross entropy error value
    # print(cross_entropy_2(Y.tolist(), R.tolist()))

    # W2 regularization
    
    W2 = np.square(W)
    W2 = W2.sum(axis = 0)
    W2 = math.sqrt(W2[0])

    # calculate and return cross_entropy and classification error as tuple
    return (entropy_error, classification_error(Y_class,R.tolist()), W2) 
                
def main():    
    
    # making pandas data frame
    data = pd.read_csv('sonar.csv', header=-1)
    
    # class data R(t) in our equation, maping to 1 - C1,0 - C2
    R = data[60]
    value = {'Mine' : 1, 'Rock' : 0}
    R = R.map(value).reshape((180,1))
    # print(R.shape)

    # drooping last column from data
    X = data.drop(data.columns[[-1]], axis=1)
    
    # value for eta
    eta = [0.001,0.01,0.05,0.1,0.5,1.0,1.5]
    
    for i in range(0,len(eta)):
        # calling method for logistic regression
        print(logistic_regression(X, 50,i, R))
   
   
if __name__ == "__main__":
    main()







