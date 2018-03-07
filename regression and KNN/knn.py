import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# ---------------------------------------------------------------
# K NN algorithm
# ---------------------------------------------------------------
    
def calculate_ecludiean_distance(point1, point2):

    #print(point1)
    #print(point2)
    
    distance = math.sqrt(math.pow((point1[0] - point2[0]),2) \
                         + math.pow((point1[1] - point2[1]),2) \
                         + math.pow((point1[2] - point2[2]),2))
        
    return distance

def calculate_manhattan_distance(point1, point2):
    pass
    
    
# ----------------------------------------------------------------
def find_neighbours(train_data, test_data, k):
    
    distance = []
    
    # calculating the distance from the given point
    for i in range(0, len(train_data)):
        distance.append((train_data[i][0], train_data[i][1], train_data[i][2],calculate_ecludiean_distance(train_data[i],test_data)))
    
    print(distance)    
         
        
# ----------------------------------------------------------------
def break_ties():
    pass


def knn_algorithm(train_data, test_data, k):
    
    for i in range(0, len(test_data)):
        find_neighbours(train_data, test_data[i], k)
                
      
def main():
    
    # reading test data
    dataset_test = pd.read_csv('auto_test.csv')
    
    # reading train data
    dataset_train = pd.read_csv('auto_train.csv')

    # K-Nearest-Neighbor
    knn_algorithm(dataset_train.iloc[:,0:3].values, dataset_test.iloc[:,0:3].values, 1)
    
    knn_algorithm(dataset_train.iloc[:,0:3].values, dataset_test.iloc[:,0:3].values, 3)
    
    knn_algorithm(dataset_train.iloc[:,0:3].values, dataset_test.iloc[:,0:3].values, 5)
    

if __name__ == "__main__":
    main()

   
    