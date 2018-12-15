
"""

Direction of Movements in extended Lift PageRank Method

Authors: Arash N. Kia
         Saman Haratizadeh
         Saeed Bagheri Shouraki

Date: 2018 June

"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
import time
import sys

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score

###############################################################################

# For print debugging... Make it False and the print debugging won't show!
debug = True

# Start and End date of the dataset
start_date = '2009-01-05'
end_date   = '2018-02-09'

# Path of the data file (Must be changed according to where you put your file!)
path       = '/home/freeze/Documents/DiMex/newdata.xls'

# proportion of train and validation in the whole dataset
# proportion_of_test => 1 - prop_train - prop_val
prop_train, prop_val = 0.7, 0.2


# Known nodes in the network = Known time series that reflect other time series futures
known_nodes = 9
# 31 days before to predict next day for supervised prediction methods
delay = 32

#1: America, 2: Europe and Africa, 3: Russia and Central Asia, 4: Far East and Australia
# With the same order of the time series in the dataset file
time_zones = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
              2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]

rs = 37 #Random State

###############################################################################
# For prompting and print debugging

def log(message):
    
    if debug:
        print(message)
        
###############################################################################
# Data Visualization (plots for time series)
# Input: dataset numpy matrix
# Output: plots of log return time series plus box-whisker 
    
def plot_all_series(dataset, headers):
    
    number_of_series = dataset.shape[1]
    fig, pltarray = plt.subplots(int(number_of_series/4), 4)
    
    for i in range(0, number_of_series):
        pltarray[int(i / 4), i % 4].plot(dataset[:, i])
        pltarray[int(i / 4), i % 4].set_title(headers[i])

    barfig, bararray = plt.subplots(int(number_of_series/4), 4)
    
    for i in range(0, number_of_series):
        bararray[int(i / 4), i % 4].boxplot(dataset[:, i])
        bararray[int(i / 4), i % 4].set_title(headers[i])

###############################################################################        
# Data Preparation Block
# Input: an excel file of markets time series, start and end date as string with format of YYYY-MM-DD
# Output: matrix of markets log-return time series (rows as days and columns as markets) in numpy array and names of markets in headers list
        
def data_reader(path, start_date, end_date):
    
    log("data_reader()...")
    
    xl = pd.read_excel(path)
    xl['Date'] = pd.to_datetime(xl['Date'])
    xl = xl.set_index('Date')
    xl = xl.loc[start_date:end_date]
    xl = xl.dropna()
    
    #Making the log return series out of pure values of indices and commodity prices
    xl = np.log(xl/xl.shift(1))
    
    xl = xl.dropna()
    
    dataset = xl.values
    headers = list(xl.columns)
    
    return [dataset, headers]
        
###############################################################################
# Splits the dataset into train, validation and test datasets
# Input: dataset!, proportion for train, proportion for validation
# Output: train, validation, and test datasets in numpy array format

def train_validation_test_maker(dataset, prop_train, prop_val):
    
    log('train_validation_test_maker()...')
    
    rows = dataset.shape[0]
    rows_train = int(rows * prop_train)
    rows_val   = int(rows * prop_val)
    
    train = dataset[:rows_train, :].copy()
    val   = dataset[rows_train:rows_train + rows_val, :].copy()
    test  = dataset[rows_train + rows_val:, :].copy()
    
    return [train, val, test]

###############################################################################
# Making the DiMex adjacency matrix for the DiMex network
# Input: Train dataset
# Output: Some kind of extend support, confidence, and lift (The lift is used)
 
def DiMex_matrix(train):
        
    log("DiMex_Rank_matrix()...")
    
    days, series = train.shape
    

    #Extended Support Matrices    
    UUS, UDS, DUS, DDS = np.zeros((series, series)), np.zeros((series, series)), np.zeros((series, series)), np.zeros((series, series))
    #Extended Confidence Matrices
    UUC, UDC, DUC, DDC = np.zeros((series, series)), np.zeros((series, series)), np.zeros((series, series)), np.zeros((series, series))
    #Extended Lift Matrices
    UUL, UDL, DUL, DDL = np.zeros((series, series)), np.zeros((series, series)), np.zeros((series, series)), np.zeros((series, series))
    
    
    for s1 in range(0, series):
        for s2 in range(0, series):
            
            uu, ud, du, dd = 0, 0, 0, 0
            
            ups_s1 = 1 * (train[:, s1] > 0) 
            ups_s1 = np.sum(ups_s1 * train[:, s1]) 
            downs_s1 = 1 * (train[:, s1] < 0)
            downs_s1 = - np.sum(downs_s1 * train[:, s1])
            
            ups_s2 = 1 * (train[:, s2] > 0)
            ups_s2 = np.sum(ups_s2 * train[:, s2])
            downs_s2 = 1 * (train[:, s2] < 0)
            downs_s2 = - np.sum(downs_s2 * train[:, s2])
            
            for d in range(0, days - 1):
                
                # Different time zones means different tomorrow in calender
                
                if time_zones[s1] > time_zones[s2]:
                    
                    if (train[d, s1] > 0) and (train[d, s2] > 0):
                        uu = uu + train[d, s2]
                    if (train[d, s1] > 0) and (train[d, s2] < 0):
                        ud = ud - train[d, s2]
                    if (train[d, s1] < 0) and (train[d, s2] > 0):
                        du = du + train[d, s2]
                    if (train[d, s1] < 0) and (train[d, s2] < 0):
                        dd = dd - train[d, s2]
               
                else:
                    
                    if (train[d, s1] > 0) and (train[d + 1, s2] > 0):
                        uu = uu + train[d + 1, s2]
                    if (train[d, s1] > 0) and (train[d + 1, s2] < 0):
                        ud = ud - train[d + 1, s2]
                    if (train[d, s1] < 0) and (train[d + 1, s2] > 0):
                        du = du + train[d + 1, s2]
                    if (train[d, s1] < 0) and (train[d + 1, s2] < 0):
                        dd = dd - train[d + 1, s2]
            
            # Final computation
            UUS[s1, s2], UDS[s1, s2], DUS[s1, s2], DDS[s1, s2] = uu / (uu + ud + du + dd), ud / (uu + ud + du + dd), du / (uu + ud + du + dd), dd/ (uu + ud + du + dd)            
            UUC[s1, s2], UDC[s1, s2], DUC[s1, s2], DDC[s1, s2] = uu / ups_s1, ud / ups_s1, du / downs_s1, dd/ downs_s1
            UUL[s1, s2], UDL[s1, s2], DUL[s1, s2], DDL[s1, s2] = uu / (ups_s1 * ups_s2), ud /(ups_s1 * downs_s2), du /(downs_s1 * ups_s2), dd /(downs_s1 * downs_s2)

    # Joining all uu, ud, du, and dd parts of the matrix
    extended_support = np.vstack((np.hstack((UUS, UDS)), np.hstack((DUS, DDS))))
    extended_confidence = np.vstack((np.hstack((UUC, UDC)), np.hstack((DUC, DDC))))
    extended_lift = np.vstack((np.hstack((UUL, UDL)), np.hstack((DUL, DDL))))

    return extended_support, extended_confidence, extended_lift

###############################################################################
# Prediction for a day (With page rank and process matrix (Lift Matrix))
# Input: sample: zeros and ones for downs and ups of a day and other zeros for unknown nodes
# Output: zeros and ones for downs and ups of unknown nodes
    
def predict(sample, process, known_nodes = known_nodes):
    
    G = nx.from_numpy_array(process, create_using = nx.DiGraph())

    series = len(sample)
    
    labels = 1 * (sample > 0)
    
    up_known_labels = labels[:known_nodes]    
    down_known_labels = np.logical_not(up_known_labels) * 1    
    unknown_labels = np.zeros((series - known_nodes,))
    
    personal_vector = np.concatenate((up_known_labels, unknown_labels, down_known_labels, unknown_labels))
  
    keys = list(range(0, series * 2))
    personalization = dict(zip(keys, personal_vector))
    
    ranks = nx.pagerank_numpy(G, personalization = personalization)

    ranks = list(ranks.values())
        
    up_unknown_ranks = np.array(ranks[known_nodes:series])
    down_unknown_ranks = np.array(ranks[series + known_nodes:])
    
    prediction = (up_unknown_ranks > down_unknown_ranks) * 1
    
    return prediction

###############################################################################
# Rank prediction for a test set
# Input: test set, network matrix
# Output: result of prediction in a test set( n days )    
    
def set_predict(test, process, known_nodes = known_nodes):
    
    days, series = test.shape
    
    result = np.zeros((days - 1, series - known_nodes))
    
    for day in range(0, days - 1):

        sys.stdout.write("\r day %s of %s" % ( str(day), str(days - 1)))
        sample = test[day, :]

        prediction = predict(sample, process, known_nodes)
        result[day, :] = prediction
    
    print()
    
    return result

###############################################################################
# Evaluate the resul of prediction
# Input: test and result set (output of set_predict)
# Output: accuracy of the whole set and each series, standar deviation of the whole set, set of errors
    
def evaluate(test, result, known_nodes = known_nodes):
    
    known_labels = (test[1:, known_nodes:] > 0) * 1
    
    errors = np.abs(known_labels - result)
    accuracy = 1 - errors.mean()
    series_accs = 1 - np.mean(errors, axis = 0)
    s = errors.std()
    
    return accuracy, s, series_accs, errors
     
###############################################################################
# Changes a time series into a dataset of n days before (delay) and next day
# Input: A time series and how many days before is important
# Output: delayed dataset
    
def delay_maker(series, delay = delay):
    
    l = len(series)
    new_data = []
    for i in range(0, l - delay):
        temp = series[i: i + delay + 1]
        new_data.append(temp)
        
    new_data = np.array(new_data)
    
    new_data[:, -1] = 1 * (new_data[:, -1] > 0)
    
    return new_data

###############################################################################
# Supervised prediction of all time series in dataset and their results in a test set
# Input: train, test, delay: (how many days before to predict the next day)
# Output: accuracy for each time series and results for all time series in all days
    
def sup_predict(train, test, delay = delay, known_nodes = known_nodes):
    
    days, series = train.shape
    valdays = test.shape[0]
    
    series_sup_accs = []
    results = np.zeros((series - known_nodes, valdays - delay))
    
    
    for s in range(known_nodes, series):
        
        sys.stdout.write("\r Supervised prediction for series %s of %s" % ( str(s), str(series)))
        timeseries_train = train[:, s]
        timeseries_test = test[:, s]
        delay_trainset = delay_maker(timeseries_train, delay)
        delay_testset   = delay_maker(timeseries_test, delay)
        
        model = RF(n_estimators = 100, random_state = rs)
        model = model.fit(delay_trainset[:, :-1], delay_trainset[:, -1])
        y_pred = model.predict(delay_testset[:, :-1])
        results[s - known_nodes, :] = y_pred
        series_sup_accs.append(accuracy_score(delay_testset[:, -1], y_pred))
    
    print()
    
    return np.array(series_sup_accs), results.T

###############################################################################
# Fast print of the accuracy of rank prediction method 
# For debugging purpose
    
def accprint(test, P):
    r = set_predict(test, P)
    a = evaluate(test, r)
    print(a[0])

###############################################################################
#
#         MAIN PART
#
###############################################################################
    
    
[dataset, headers] = data_reader(path, start_date, end_date)

[train, validation, test] = train_validation_test_maker(dataset, prop_train, prop_val)

# Train and validation together for the test phase (After validation phase)
trval = np.vstack((train, validation))

###
# Comparing phase in validation
###

print()

print("1) Computing extended lifts for network of movements in train set...")

start = time.time()

PS, PC, PL = DiMex_matrix(train)

print("2) Rank Prediction in validation set...")
    
result = set_predict(validation, PL)
acc, std, rank_series_accs, e = evaluate(validation, result)

print("DiMex rank accuracy of complete graph in validation set: ", acc)

print("3) Supervised predictionin validation set...")
    
sup_accs, sup_results = sup_predict(train, validation)

# Finding candidate markets to be predicted by supervised method
comparison = sup_accs > rank_series_accs
candidates = np.where(comparison)

###
# Results in Test set
###

print("4) Computing Lifts network of movements in train+validation set...")

PS_trval, PC_trval, PL_trval = DiMex_matrix(trval)

print("5) Rank prediction in test set...")
    
result_test = set_predict(test, PL_trval)
acc_test, std_test, rank_series_accs_test, e_test = evaluate(test, result_test)

print("DiMex rank accuracy in test set: ", acc_test)

print("6) Supervised prediction in test set...")
    
sup_accs_test, sup_results_test = sup_predict(trval, test)

# Finding mixture model accuracy without eliminating the candidates from the network

print("7) Mixture model...")
    
mixture_accs = rank_series_accs_test.copy()

for i in candidates:
    mixture_accs[i] = sup_accs_test[i]

# Delay maker eliminates some rows. s is the number of rows after elimination    
s = sup_results_test.shape[0]    
rank_results_reduced = result_test[-s:, :].copy()

mixed_results_without_elimination = rank_results_reduced.copy()

for i in candidates:
    mixed_results_without_elimination[:, i] = sup_results_test[:, i]


print("Accuracy of mixture method with graph with candidates:", mixture_accs.mean())


# Final mixture model with elimiation of candidate market who are better predicted with supervised models

dataset_without_candids = np.delete(dataset, candidates, 1)
[train_w, validation_w, test_w] = train_validation_test_maker(dataset_without_candids, prop_train, prop_val)
PS_w, PC_w, PL_w = DiMex_matrix(np.vstack((train_w, validation_w)))
result_w = set_predict(test_w, PL_w)
acc_w, std_w, rank_series_accs_test_w, e_test_w = evaluate(test_w, result_w)

print("Accuracy without candidates with DiMex network: ", acc_w)


e_test_final = np.zeros((test.shape[0] - delay, test.shape[1] - known_nodes))
final_accs_list = np.zeros((test.shape[1] - known_nodes,))

c = 0
for i in range(0, test.shape[1] - known_nodes):
    if i in candidates[0]:
        e_test_final[:, i] = sup_results_test[:, i]
        final_accs_list[i] = sup_accs_test[i]
        c = c + 1
    else:
        e_test_final[:, i] = e_test_w[:s, i - c]
        final_accs_list[i] = rank_series_accs_test_w[i - c]
            
print("Final mixture model accuracy: ", final_accs_list.mean())

end = time.time()

print("8) T-Tests...")
        
# T-Test calculation between best rival model (HyS3) and our model
# The results of HyS3 are the output of HyS3.py code which is available in my github
HyS3_err = np.array(pd.read_csv('/home/freeze/Documents/DiMex/HyS3_error.csv', header = None))
HyS3_err = HyS3_err[:-2, :]
#Accuracy list for HyS3 model (from HyS3.py accesssible in my github)
HyS3_accs_list = 1 - HyS3_err.mean(axis = 0)

# ConKruG network model accuracies (from ConKruG.py accessible in my github)
ConKruG_err = np.array(pd.read_csv('/home/freeze/Documents/DiMex/ConKruG_error.csv', header = None))
ConKruG_err = ConKruG_err[:-1, :]
ConKruG_accs_list = 1 - ConKruG_err.mean(axis = 0)

#target = 1 * (test[delay:, known_nodes:] > 0)

e_test_final = e_test_final[:-2, :]
HyS3_err = HyS3_err[delay:, :]
r = rank_results_reduced.flatten()

pvalue_between_final_models = stats.ttest_rel(e_test_final.flatten(), HyS3_err.flatten())
r2 = r[:-2*(test.shape[1] - known_nodes)]
pvalue_between_mixed_ranked = stats.ttest_rel(e_test_final.flatten(), r2)


print("p-Value between mixture method and rank method:", stats.ttest_rel(e_test_final.flatten(), r2))
print("P-Value between final mixed model and HyS3:", pvalue_between_final_models)

print()

print("Time of the whole process:", (end - start) / 60, "mins!")
