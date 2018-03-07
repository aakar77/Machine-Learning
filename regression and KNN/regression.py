import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures


    
# Function for calculating squarred error
def calculate_squarred_error(mpg_test, mpg_predict):
    error = mpg_predict - mpg_test
    error = np.square(error)
    
    sum = np.sum(error) / (error.size * 2)
    
    return (sum)


# 1) plotting displacement versus mpg
def plott_graph_one(displacement, mpg):
    
    plt.plot(displacement, mpg, 'ro')
    plt.xlabel("displacement")
    plt.ylabel("mpg")
    plt.show()
    
def plott_graph(test, mpg_test, mpg_predict):
    
    # Plotting on the test data
    plt.plot(test, mpg_test,'ro')
    plt.plot(test, mpg_predict, color = "blue")
    plt.show()
    
# 2) Simple linear regression, train a linear function to predict mpg based on displacement only       
def linear_regression(displacement_train, mpg_train, displacement_test, mpg_test):
    
    displacement_test = displacement_test.reshape((101,1))
    
    # Creating Linear regression object
    linear_regression = LinearRegression()
    linear_regression.fit(displacement_train.reshape((291,1)),mpg_train.reshape((291,1)))

    # Testing the train data
    mpg_predict = linear_regression.predict(displacement_test)
    
    
    # Plotting on the test data
    plt.scatter(displacement_test, mpg_test, color = "red")
    plt.plot(displacement_test, mpg_predict, color = "blue")
    plt.show()
    
    #print(mpg_predict)
    #print(mpg_test)
    
    print(calculate_squarred_error(mpg_test, mpg_predict))
    

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

    # Testing on training data
    p_test_matrix = poly_regression.fit_transform(p_test.reshape(101,1))
    mpg_predict = linear_regression.predict(p_test_matrix)
    
    #Plotting the graph
    plott_graph(p_test, mpg_test, mpg_predict)
    
    # caclulating the squarred error
    #print(mpg_predict)
    print(calculate_squarred_error(mpg_test, mpg_predict))
    
    return 
    
    
# 4) multiple_linear_regression(data, train, test, )
def multiple_linear_regression(mlr_train, mpg_train, mlr_test, mpg_test):
    
    linear_regression = LinearRegression()
    linear_regression.fit(mlr_train, mpg_train.reshape((291,1)))

    # Testing the train data
    mpg_predict = linear_regression.predict(mlr_test)
    
    #print(len(mpg_predict))
    #print(len(mpg_test))
    
    
    print(calculate_squarred_error(mpg_test, mpg_predict))
    return 
    

def main():    

    mydict = {}
    
    # reading train data
    dataset_train = pd.read_csv('auto_train.csv')

    # reading the  train data 
    displacement_train = dataset_train.iloc[:,0].values
    horsepower_train = dataset_train.iloc[:, 1].values
    mpg_train = dataset_train.iloc[:,2].values
    
    
    # reading test data
    dataset_test = pd.read_csv('auto_test.csv')
    
    #reading the test data
    displacement_test = dataset_test.iloc[:,0].values
    horsepower_test = dataset_test.iloc[:,1].values
    mpg_test = dataset_test.iloc[:,2].values
    
    plott_graph_one(displacement_test, mpg_test)

    #  2 -------- Linear Regression
    
    # calling linear regression function
    linear_regression(displacement_train, mpg_train, displacement_test, mpg_test)
    
    # 3 ------- Polynomial Regression
    poly_regression(displacement_train, mpg_train, displacement_test, mpg_test, 2)
    poly_regression(displacement_train, mpg_train, displacement_test, mpg_test, 4)
    poly_regression(displacement_train, mpg_train, displacement_test, mpg_test, 6)
    
    
    # 4 ------- Multiple Linear Regression 
    
    # taking displacement and horsepower for training and test
    mlr_train = dataset_train.iloc[:,0:2].values   
    mlr_test = dataset_test.iloc[:,0:2].values
 
    # calling multiple linear regression function
    multiple_linear_regression(mlr_train, mpg_train, mlr_test, mpg_test)
    
    
if __name__ == "__main__":
    main()










#displacement =
#mpg = 