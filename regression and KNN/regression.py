import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


    
# Function for calculating squarred error
def calculate_squarred_error(mpg_test, mpg_predict):
    error = mpg_predict - mpg_test
    error = np.square(error)
    
    sum = np.sum(error) / (error.size * 2)
    
    return (sum)


# 1) plotting displacement versus mpg
def plott_graph(displacement, mpg):
    
    plt.plot(displacement, mpg, 'ro')
    plt.xlabel("displacement")
    plt.ylabel("mpg")
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

def main():    

    # reading data and dataset
    dataset_train = pd.read_csv('auto_train.csv')
    dataset_test = pd.read_csv('auto_test.csv')
    
    # reading the  train data 
    displacement_train = dataset_train.iloc[:,0].values
    horsepower_train = dataset_train.iloc[:, 1].values
    mpg_train = dataset_train.iloc[:,2].values
    
    #reading the test data
    displacement_test = dataset_test.iloc[:,0].values
    horsepower_test = dataset_test.iloc[:,1].values
    mpg_test = dataset_test.iloc[:,2].values
    
    print(len(displacement_test))
    
    # calling linear regression function
    linear_regression(displacement_train, mpg_train, displacement_test, mpg_test)
    
    
    #plott_graph(displacement, mpg)
    
    
    
if __name__ == "__main__":
    main()










#displacement =
#mpg = 