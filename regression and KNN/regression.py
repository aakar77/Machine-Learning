import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

    
# Function for calculating squarred error
def calculate_squarred_error(mpg_test, mpg_predict):
    error_list = [math.pow(float(mpg_test[i] - mpg_predict[i]), 2) for i in range(len(mpg_predict))]
    return float(sum(error_list)) / 2
        

# 1) plotting displacement versus mpg
def plott_graph_one(displacement, mpg):
    plt.plot(displacement, mpg, 'ro')
    plt.xlabel("displacement")
    plt.ylabel("mpg")
    plt.show()

# Graph plot function used for 2, 3 questions   
def plott_graph(test, mpg_test, mpg_predict):
    
    # creating a list of tuple [(x,y), (x,y)]
    mpg_tuple = zip(test, mpg_predict)
    
    #sorting the list of tuples based on displacement
    mpg_tuple = sorted(mpg_tuple, key=lambda x: x[0])
    
    plt.xlabel("displacement")
    plt.ylabel("mpg")
    
    # Plotting on the test data
    plt.plot(test, mpg_test,'ro')
    
    plt.plot(*zip(*mpg_tuple), color = "blue")
    plt.show()
    
# 2) Simple linear regression, train a linear function to predict mpg based on displacement only       
def linear_regression(displacement_train, mpg_train, displacement_test, mpg_test):
    
    displacement_test = displacement_test.reshape((101,1))
    
    # Creating Linear regression object
    linear_regression = LinearRegression()
    linear_regression.fit(displacement_train.reshape((291,1)),mpg_train.reshape((291,1)))

    # Testing  on the test data
    mpg_predict_test = linear_regression.predict(displacement_test)
    
    # Testing on the train data
    mpg_predict_train = linear_regression.predict(displacement_train.reshape((291,1)))
    
    plott_graph(displacement_test, mpg_test, mpg_predict_test)
    

    return (calculate_squarred_error(mpg_test, mpg_predict_test), \
             calculate_squarred_error(mpg_train, mpg_predict_train))

# 3) Polynomial Regression, train 
def poly_regression(p_train,mpg_train, p_test, mpg_test, degree):
    
    poly_regression = PolynomialFeatures(degree = degree)

    # transforming the array into poly of degree 2
    p_train_matrix = poly_regression.fit_transform(p_train.reshape((291,1)))
    
    # training the model
    poly_regression.fit(p_train_matrix, mpg_train.reshape((291,1)))
   
    # creating linear regression object
    linear_regression = LinearRegression()
    linear_regression.fit(p_train_matrix,  mpg_train.reshape(291,1))

    # creating model using training data
    p_test_matrix = poly_regression.fit_transform(p_test.reshape(101,1))
    
    # Getting code of mpg value and and converting array to list using ravel 
    mpg_predict_test = linear_regression.predict(p_test_matrix).ravel()
    
    # Testing on the train data
    mpg_predict_train = linear_regression.predict(p_train_matrix).ravel()
    
    #Plotting the graph
    plott_graph(p_test, mpg_test, mpg_predict_test)
    
    # caclulating the squarred error
    # print(mpg_predict)
    return (calculate_squarred_error(mpg_test, mpg_predict_test), \
             calculate_squarred_error(mpg_train, mpg_predict_train))
    
# 4) multiple_linear_regression(data, train, test, )
def multiple_linear_regression(mlr_train, mpg_train, mlr_test, mpg_test):
    
    linear_regression = LinearRegression()
    linear_regression.fit(mlr_train, mpg_train.reshape((291,1)))

    # Testing the train data
    mpg_predict_test = linear_regression.predict(mlr_test)
    mpg_predict_train = linear_regression.predict(mlr_train)
    
    #print(len(mpg_predict))
    #print(len(mpg_test))
    
    return (calculate_squarred_error(mpg_test, mpg_predict_test), \
             calculate_squarred_error(mpg_train, mpg_predict_train))

def main():    
    
    mydict = {}
    
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
    
    
    # 1 -- Plot Graph Test data displacement versus mpg
    plott_graph_one(displacement_train, mpg_train)

    # 2 -------- Linear Regression
    mydict[0] = linear_regression(displacement_train, mpg_train, displacement_test, mpg_test)
    print("Linear Regression Testing error %f" %(mydict[0][0]))
    print("Linear Regression Trainning error %f" %(mydict[0][1]))
    
    # 3 ------- Polynomial Regression
    mydict[1] = poly_regression(displacement_train, mpg_train, displacement_test, mpg_test, 2)
    print("Polynomial Regression of degree 2 Testing error %f" %(mydict[1][0]))
    print("Polynomial Regression of degree 2 Training error %f" %(mydict[1][1]))
    

    mydict[2] = poly_regression(displacement_train, mpg_train, displacement_test, mpg_test, 4)
    print("Polynomial Regression of degree 4 Testing error %f" %(mydict[2][0]))
    print("Polynomial Regression of degree 4 Training error %f" %(mydict[2][1]))
 
    
    mydict[3] = poly_regression(displacement_train, mpg_train, displacement_test, mpg_test, 6)
    print("Polynomial Regression of degree 6 Testing error %f" %(mydict[3][0]))
    print("Polynomial Regression of degree 6 Training error %f" %(mydict[3][1]))
 
    
    # 4 ------- Multiple Linear Regression 
    
    # taking displacement and horsepower for training and test
    mlr_train = dataset_train.iloc[:,0:2].values   
    mlr_test = dataset_test.iloc[:,0:2].values
 
    # calling multiple linear regression function
    mydict[4] = multiple_linear_regression(mlr_train, mpg_train, mlr_test, mpg_test)        
    print("Multiple Linear Regression Testing error %f" %(mydict[4][0]))
    print("Multiple Linear Regression Training error %f" %(mydict[4][1]))
    
    
    
if __name__ == "__main__":
    main()






#displacement =
#mpg = 