import matplotlib.pyplot as plt   
import numpy as np

def fx(x):
    return 16*pow(x,4) - 32*pow(x,3) - 8*pow(x,2) + 10*pow(x,1) + 9

def d_fx(x):
    return 16*4*pow(x,3) - 32*3*pow(x,2) - 8*2*pow(x,1) + 10

# Question _ 1 for plotting the graph 
def question_1(inter_1, inter_2):
    
    t = np.linspace(-2, 3, 1000)    # 51 points between 0 and 3
    y = np.zeros(len(t))         # allocate y with float elements
    for i in xrange(len(t)):
        y[i] = fx(t[i])
    
    plt.plot(t, y)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

    print("As seen from the above graph the local min is at ")

# Gradient descent procedure    
def gradient_descent(steps, initial, rate):
    
    ans = {}
    fx_list = []
    x_list = []
    
    x = initial
    
    for i in range(1,steps+1):
        
        val = d_fx(x) 
        x = x - rate * val
        
        x_list.append(x)
        fx_list.append(val)
    
    ans['x'] = x_list
    ans['fx'] = fx_list
    
    return ans
        
def print_series(mydict, flag):
    
    if flag == 0 :
        # Print First 5
        for i in range(0,5):
            print(i,'  x ',mydict['x'][i],' f(x)',mydict['fx'][i])    
    elif flag == 1:
        # Print Last 5
        for i in range(len(mydict['x'])-5,len(mydict['x'])):
            print(i,' x ',mydict['x'][i],' f(x)',mydict['fx'][i])
    elif flag == 2:
        # Print Top 10
        for i in range(5,10):
            print(i,' x ',mydict['x'][i],' f(x)',mydict['fx'][i])
    else:
        # Print the entire list
        for i in range(0, len(mydict['x'])):
            print(i,' x ',mydict['x'][i],' f(x)',mydict['fx'][i])
    

def main():    
    
    # for finding the local minimum and plotting the graph both
    question_1(-2,3)
    
    print("\nx = (-2) eta = 0.001, steps = 5")
    # gradient descent 1
    ans1 = gradient_descent(5, -1, 0.001)    
    print_series(ans1, 0)

        
    #B  gradient descent
    print("\nB) x = (-1) eta = 0.001, steps = 1000")
    ans2 = gradient_descent(1000, -1, 0.001)
    
    # First 5 values and Last 5 values
    print_series(ans2, 0)
    print("")
    print_series(ans2, 1)
    
    #C  gradient descent 3
    print("\nC) x = 2 eta = 0.001, steps = 1000")
    ans3 = gradient_descent(1000, -2, 0.001)

    # First 5 values and Last 5 values
    print_series(ans3, 0)
    print("")
    print_series(ans3, 1)        

    #D  gradient descent 4
    print("\nD) x = -1 eta = 0.01, steps = 1000")
    ans3 = gradient_descent(1000, -2, 0.001)

    # First 5 values and Last 5 values
    print_series(ans3, 0)
    print("")
    print_series(ans3, 1)        
    print_series(ans3, 2)

    #E  gradient descent 4
    print("\nE) x = -1 eta = 0.01, steps = 100")
    ans3 = gradient_descent(100, -2, 0.001)

    # First 5 values and Last 5 values
    print_series(ans3, 4)
    
if __name__ == "__main__":
    main()