import math

# This method reads file
def read_file(filePath):
    with open(filePath) as f:
        data = [line.split() for line in f]
    data = [i[0].split(',') for i in data]

    return data


# method for mean for entire column
def calculate_mean(data):
    mysum = sum(map(float, data))

    return mysum / len(data)


#  method for variance calculation
def calculate_var(data, mean):
    sumData = 0
    for x in data:
        sumData += (float(x) - mean) ** 2

    return float(sumData) / (len(data) - 1)

# method returns std and mean for a dataset
def calculate_mean_std_of_dataset(data):

    h = dict()

    for i in range(1, 10):
        # Inserting entire row values in the list
        attrbValue = list()
        attrbValue.append([float(k[i]) for k in data])
        attrbValue = attrbValue[0]

        print attrbValue


        # Calculating Mean and Standard of each attribute
        attribMean = calculate_mean(attrbValue)
        attribStd = calculate_var(attrbValue, attribMean)

        print attribMean
        print attribStd

        h[i] = [attribMean, attribStd]
    return h


# log of guassian probability density function for x
def calculate_log_prob_xi(val, mean, var):


    x = (-0.5) * math.log(2 * math.pi * var)
    y = math.pow((val - mean), 2) / ( -2 * var)

    return x + y

    '''
    x = float(math.pow(val-mean, 2)) / (-2 * var)
    exp_part = float(math.exp(x))
    y = 1 / math.sqrt(2 * math.pi * var)

    return math.log(exp_part * y)
    '''


'''
1) Method calculates Sum of P(Xi | C) for all Xi
2) Finds log(P(C1)) + Sum P(Xi | C) and log(P(C2)) + Sum P(Xi | C)
3) Finds compares log(P(C1) * P(
4) returns a dict having three lists having above values

'''
def calculate_p_of_class(data, attrib_stat, un_pC1, un_pC2):

    prob_list = dict()
    prob_list[0] = []
    prob_list[1] = []
    prob_list[2] = []
    prob_list[3] = []
    prob_list[4] = []

    # for each row calculate the formula log( P(C) ) + sum log(p(xi | C))
    for row in data:
        prod_pXi = float(1)
        for i in range(1, 10):
            prod_pXi *= calculate_log_prob_xi(float(row[i]), attrib_stat[i][0], attrib_stat[i][1])

        prob_list[0].append(prod_pXi)

        log_pC1_sum_pXi = math.log(un_pC1) + prod_pXi
        log_pC2_sum_pXi = math.log(un_pC2) + prod_pXi

        prob_list[1].append(log_pC1_sum_pXi)
        prob_list[2].append(log_pC2_sum_pXi)

        if(log_pC1_sum_pXi > log_pC2_sum_pXi):
            prob_list[3].append(1)
        else:
            prob_list[3].append(2)

        prob_list[4].append(int(row[10]))

    return prob_list

def calculate_un_prob(data):
    pass


def calculate_train_error(compute_class, actual_class):
    pass

def gaussian_nb(data):

    p = calculate_un_prob(data)
    resClass = list()
    resClass.append([i[10] for i in data])
    resClass = resClass[0]

    p = dict()  # dict of probabilities of resultant classes
    N = 200

    # unconditional probability for class 1
    p['1'] = float(resClass.count('1')) / N

    # unconditional probability for class 2
    p['2'] = float(resClass.count('2')) / N

    '''
        # class1_data = list()
        # class2_data = list()
        for i in data:
            if i[10] == "1":
                class1_data.append(i)
            elif i[10] == "2":
                class2_data.append(i)
        # Calculate the mean and std for all the attributes
        class1_stat = calculate_mean_std_of_dataset(class1_data)
        class2_stat = calculate_mean_std_of_dataset(class2_data)
        # Calculate the Probability of Class, will get a List:
        map_1_1 = calculate_p_of_class(data, class1_stat, p['1'])
        map_1_2 = calculate_p_of_class(data, class1_stat, p['2'])
    '''

    data_stat = calculate_mean_std_of_dataset(data)
    prob_list = calculate_p_of_class(data, data_stat, float(p['1']), float(p['2']))

    print prob_list[0]
    print prob_list[1]
    print prob_list[2]
    print prob_list[3]
    print prob_list[4]

    calculate_train_error(prob_list[3], prob_list[4])

def main():
    filePath = 'glasshw1.csv'
    data = read_file(filePath)
    gaussian_nb(data)

if __name__ == "__main__":
    main()
