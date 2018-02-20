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

        # Calculating Mean and Standard of each attribute
        attribMean = calculate_mean(attrbValue)
        attribStd = calculate_var(attrbValue, attribMean)

        h[i] = [attribMean, attribStd]
    return h


# log of guassian probability density function for x
def calculate_log_prob_xi(val, mean, var):

    x = (-0.5) * math.log(2 * math.pi * var)
    y = math.pow((val - mean), 2) / (-2 * var)

    return x + y
    '''
    x = float(math.pow(val-mean, 2)) / (-2 * var)
    exp_part = float(math.exp(x))
    y = 1 / math.sqrt(2 * math.pi * var)
    return math.log(exp_part * y)
    '''

def calculate_p_of_class(data, class1_stat, class2_stat, un_pC1, un_pC2):
    prob_list = dict()
    prob_list['pC1'] = []
    prob_list['pC2'] = []
    prob_list['Xi1'] = []
    prob_list['Xi2'] = []
    prob_list['pre'] = []
    prob_list['post'] = []

    # for each row calculate the formula log( P(C) ) + sum log(p(xi | C))
    for row in data:
        sum_pXi_C1 = float(0)
        sum_pXi_C2 = float(0)

        #print class1_stat
        #print class2_stat

        for i in range(1, 10):
            #print(i)
            sum_pXi_C1 += calculate_log_prob_xi(float(row[i]), class1_stat[i][0], class1_stat[i][1])
            sum_pXi_C2 += calculate_log_prob_xi(float(row[i]), class2_stat[i][0], class2_stat[i][1])

        prob_list['Xi1'].append(sum_pXi_C1)
        prob_list['Xi2'].append(sum_pXi_C2)

        log_pC1_sum_pXi = math.log(un_pC1) + sum_pXi_C1
        log_pC2_sum_pXi = math.log(un_pC2) + sum_pXi_C2

        prob_list['pC1'].append(log_pC1_sum_pXi)
        prob_list['pC2'].append(log_pC2_sum_pXi)

        if (log_pC1_sum_pXi > log_pC2_sum_pXi):
            prob_list['post'].append(1)
        else:
            prob_list['post'].append(2)

        prob_list['pre'].append(int(row[10]))

    return prob_list

def calculate_train_error(pre, post):
    k = 0
    error_index = []

    for i in range(len(pre)):

        if pre[i] != post[i]:
            k += 1
            error_index.append(post)
    return (k / float(len(pre))) * 100

def gaussian_nb(train_data, test_data, prior):

    # Divide data into two classes
    class1_data = list()
    class2_data = list()

    for row in train_data:
        if row[10] == "1":
            class1_data.append(row)
        elif row[10] == "2":
            class2_data.append(row)

    # Calculate the mean and std for all the attributes
    class1_stat = calculate_mean_std_of_dataset(class1_data)
    class2_stat = calculate_mean_std_of_dataset(class2_data)

    # Calculate the Probability of Class, will get a List:
    prob_list = calculate_p_of_class(test_data, class1_stat, class2_stat, prior['1'], prior['2'])

    print prob_list['pC1']
    print prob_list['pC2']
    print prob_list['Xi1']
    print prob_list['Xi2']
    print prob_list['pre']
    print prob_list['post']

    prob_list['train_error'] = calculate_train_error(prob_list['pre'], prob_list['post'])

    return prob_list

def calculate_prior(data):
    resClass = list()
    resClass.append([i[10] for i in data])
    resClass = resClass[0]

    p = dict()
    N = len(data)

    # unconditional probability for class 1
    p['1'] = float(resClass.count('1')) / N

    # unconditional probability for class 2
    p['2'] = float(resClass.count('2')) / N

    return p

def k_fold_validation(data, prior, k):

    error = float(0)
    final_prediction = {}

    # Generating data for k fold and calling Gaussian calculation
    for i in range(k):
        x = len(data) / k
        y = x * i

        #print(" %s : %s " % (y,y+x))
        train_data = data[:y] + data[y+x:]
        test_data = data[y:y+x]
        train_error_list = []

        computed_data = gaussian_nb(train_data,test_data, prior)

        j = 0
        for i in range(y, y+x):
            try:
                final_prediction[i] = computed_data['post'][j]
                print("i = %s : data = %s : j = %s" %(i,computed_data['post'][j], j))
            except:
                print("List Error at i = %s j = %s" %(i, j))
            j = j + 1

        train_error_list.append(computed_data['train_error'])
        error += computed_data['train_error']

def main():
    filePath = 'glasshw1.csv'
    data = read_file(filePath)

    # calculate unconditional probabililty for both the classes
    prior = calculate_prior(data)

    # sending the same data for test and train for whole 200 entries
    all_data = gaussian_nb(data, data, prior)

    #five fold validation
    k_fold_validation(data, prior, 5)

if __name__ == "__main__":
    main()
