import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# ---------------------------------------------------------------
# K NN algorithm
# ---------------------------------------------------------------
    
# Only have to consider displacement and horsepower and to predict the MPG value
def calculate_ecludiean_distance(point1, point2):
    
    distance = math.sqrt(math.pow((point1[0] - point2[0]),2) \
                         + math.pow((point1[1] - point2[1]),2)) 
    return distance

    
# Function for calculating squarred error
def calculate_squarred_error(mpg_test, mpg_predict):
    
    error_list = [math.pow(float(mpg_test[i] - mpg_predict[i]), 2) for i in range(len(mpg_predict))]
    
    return float(sum(error_list)) / 2
        
# ----------------------------------------------------------------
def find_neighbours(train_data, test_data, k):
    distance = []
    k_near_list = []
    
    # calculating the distance from the given point
    for i in range(0, len(train_data)):
        distance.append((train_data[i][0], train_data[i][1], train_data[i][2], calculate_ecludiean_distance(train_data[i],test_data)))
    
    distance.sort(key = lambda tup : tup[3]) 

    '''
    for i in range(len(distance)):
        print(distance[i])    
    '''    
    mpg_predict = float(0)
    
    # for breaking ties, as of now I am just considering index value 
    # adding the k nearest neighbour to the list
    for i in range(k):
        k_near_list.append((distance[i][0], distance[i][1], distance[i][2]))
        mpg_predict += distance[i][2]
        
    mpg_predict = mpg_predict / k
    
    print("\n ---- For the Test value --- %f %f %f"%(test_data[0], test_data[1], test_data[2]))
    
    for i in range(len(k_near_list)):
        print("%s : %s : %s" %(k_near_list[i][0], k_near_list[i][1],k_near_list[i][2]))
    
    print("\n Predicted Value : %s" %(mpg_predict))
              
    print("\n")
    
    return mpg_predict    

def knn_algorithm(train_data, test_data, k):
    
    test_mpg = []
    predict_mpg = []
    
    for i in range(0, len(test_data)):
        test_mpg.append(test_data[i][2])
        predict_mpg.append(find_neighbours(train_data, test_data[i], k))
      
    
    for i in range(len(test_mpg)):
        print(" %f : %f" %(test_mpg[i], predict_mpg[i]))
    
    test_error = calculate_squarred_error(test_mpg, predict_mpg)  
    
    print("\n Test error %s" %(test_error))
    
    
def main():
    
    # reading test data
    dataset_test = pd.read_csv('auto_test.csv')
    
    # reading train data
    dataset_train = pd.read_csv('auto_train.csv')

    knn_algorithm(dataset_train.iloc[:,0:3].values, dataset_test.iloc[:,0:3].values, 20)

    knn_algorithm(dataset_train.iloc[:,0:3].values, dataset_test.iloc[:,0:3].values, 5)

    # K-Nearest-Neighbor
    knn_algorithm(dataset_train.iloc[:,0:3].values, dataset_test.iloc[:,0:3].values, 1)
    
    knn_algorithm(dataset_train.iloc[:,0:3].values, dataset_test.iloc[:,0:3].values, 5)
    
    knn_algorithm(dataset_train.iloc[:,0:3].values, dataset_test.iloc[:,0:3].values, 20)
    
    
if __name__ == "__main__":
    main()

   
    