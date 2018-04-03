from IPython import get_ipython
get_ipython().magic('reset -sf')


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
   
# Function for calculating squarred error
def calculate_squarred_error(mpg_test, mpg_predict):
    error_list = [math.pow(float(mpg_test[i] - mpg_predict[i]), 2) for i in range(len(mpg_predict))]
    return float(sum(error_list)) / 2
        

def main():    
    
 
    
    df = pd.read_csv('sonar.csv')
    x = {'Mine' : 1, 'Rock' : 2}
    
    df.iloc[:,60] = df.iloc[:,60].map(x)
    
    print(df.iloc[:,60])
   
if __name__ == "__main__":
    main()











