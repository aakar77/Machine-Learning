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

# Only have to consider displacement and horsepower and to predict the MPG value
def calculate_manhattan_distance(point1, point2):
    
    distance = math.fabs(point1[0] - point2[0]) + math.fabs(point1[1] - point2[1]) 
    #print(distance)
    return distance
    
# Function for calculating squarred error
def calculate_squarred_error(mpg_test, mpg_predict):
    
    error_list = [math.pow(float(mpg_test[i] - mpg_predict[i]), 2) for i in range(len(mpg_predict))]
    
    return float(sum(error_list)) / 2
        
# ----------------------------------------------------------------
def find_neighbours(train_data, test_data, k):
    distance = []
    k_near_list = []
    
    # calculating the distance from the given point, forms list of tuples
    for i in range(0, len(train_data)):
        distance.append((train_data[i][0], train_data[i][1], train_data[i][2], calculate_ecludiean_distance(train_data[i],test_data)))
    
    # tup 3 is the euclidean distance, sorting list by tuple 3
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
    
    '''
    #--------- Debugging
    
    print("\n ---- For the Test value --- %f %f %f"%(test_data[0], test_data[1], test_data[2]))
    
    for i in range(len(k_near_list)):
        print("%s : %s : %s" %(k_near_list[i][0], k_near_list[i][1],k_near_list[i][2]))
    
    print("\n Predicted Value : %s" %(mpg_predict))          
    print("\n")

    '''
    return mpg_predict    

def knn_algorithm(train_data, test_data, k):
    
    test_mpg = []
    predict_mpg = []

    # calculating the distance from the given point, forms list of tuples    
    for i in range(0, len(test_data)):
        test_mpg.append(test_data[i][2])
        predict_mpg.append(find_neighbours(train_data, test_data[i], k))
      
    '''
    #--------- Debugging
    
    for i in range(len(test_mpg)):
        print(" %f : %f" %(test_mpg[i], predict_mpg[i]))
    '''
    test_error = calculate_squarred_error(test_mpg, predict_mpg)  
    
    return test_error

#----------- For question 7

def find_neighbours_weighted(train_data, test_data, k):
    
    distance = []
    k_near_list = []
    weight_sum = float(0)
    mpg_predict = float(0)
    
    # calculating the distance from the given point
    for i in range(0, len(train_data)):
        distance.append((train_data[i][0], train_data[i][1], train_data[i][2], calculate_manhattan_distance(train_data[i],test_data)))
    
    # tup is the manhattan distance, sorting list by tuple 3
    distance.sort(key = lambda tup : tup[3]) 

    # for breaking ties, as of now I am just considering index value 
    # adding the k nearest neighbour to the list
    # taking Weighted average
    for i in range(k):
        k_near_list.append((distance[i][0], distance[i][1], distance[i][2]))
        
        # if the distance = 0, returning the point's mpg value to be most nearest so return it
        if(distance[i][3] == 0):
            mpg_predict = distance[i][2]
            return(mpg_predict)
        else:
            inverse_distance = (1 / distance[1][3])
            mpg_predict += (distance[i][2]) * inverse_distance
            weight_sum += inverse_distance
    
    mpg_predict = mpg_predict / weight_sum
    
    '''
    #--------- Debugging
    print("\n ---- For the Test value --- %f %f %f"%(test_data[0], test_data[1], test_data[2]))
    
    for i in range(len(k_near_list)):
        print("%s : %s : %s" %(k_near_list[i][0], k_near_list[i][1],k_near_list[i][2]))
    
    print("\n Predicted Value : %s" %(mpg_predict))          
    print("\n")    
    '''
    return mpg_predict    

def knn_algorithm_weighted(train_data, test_data, k):
    
    test_mpg = []
    predict_mpg = []
    
    for i in range(0, len(test_data)):
        test_mpg.append(test_data[i][2])
        
        # calling function for finding weighted neighbours
        predict_mpg.append(find_neighbours_weighted(train_data, test_data[i], k))
      
    '''
    #--------- Debugging
    
    for i in range(len(test_mpg)):
        print(" %f : %f" %(test_mpg[i], predict_mpg[i]))    
    '''
    test_error = calculate_squarred_error(test_mpg, predict_mpg)  
    
    return test_error
    
def main():
    
    # reading test data
    dataset_test = pd.read_csv('auto_test.csv')
    
    # reading train data
    dataset_train = pd.read_csv('auto_train.csv')


    # K-Nearest-Neighbor
    test_error1 = knn_algorithm(dataset_train.iloc[:,0:3].values, dataset_test.iloc[:,0:3].values, 1)  
    test_error3 = knn_algorithm(dataset_train.iloc[:,0:3].values, dataset_test.iloc[:,0:3].values, 3) 
    test_error20 = knn_algorithm(dataset_train.iloc[:,0:3].values, dataset_test.iloc[:,0:3].values, 20)
    
    
    print("Test error for k = 1 %f" %(test_error1))
    print("Test error for k = 3 %f" %(test_error3))
    print("Test error for k = 20 %f" %(test_error20))
    
    print("\n\nQues 6 ANSWER: \nOn Observing the test error, performance of KNN with 20 neighbours \nis better as compared to that with 3 neighbours")
    
    print("\n")
    
    # Weighted K Nearest Neighbor
    test_error_w1 = knn_algorithm_weighted(dataset_train.iloc[:,0:3].values, dataset_test.iloc[:,0:3].values, 1) 
    print("Weighted KNN - Test error for k = 1 %f" %(test_error_w1))
    
    test_error_w3 = knn_algorithm_weighted(dataset_train.iloc[:,0:3].values, dataset_test.iloc[:,0:3].values, 3)
    print("Weighted KNN - Test error for k = 3 %f" %(test_error_w3))
    
    test_error_w20 = knn_algorithm_weighted(dataset_train.iloc[:,0:3].values, dataset_test.iloc[:,0:3].values, 20)
    print("Weighted KNN - Test error for k = 20 %f" %(test_error_w20))
    
    print("\n\nQues 7 ANSWER: \n1) We have used Manhattan distance instead of euclidean distance")
    print("2) Taken weighted average, considering 1 / distance as weightes")
    print("3) For those points whose distance is 0, We are directly taking that point's mpg as predicted value instead of taking weighted average")
    
if __name__ == "__main__":
    main()

   
    