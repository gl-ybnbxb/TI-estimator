''' Functions for simulations'''
import numpy as np
import sys
import pandas as pd
from collections import Counter, defaultdict
import re
from scipy.stats import zscore
from scipy.special import logit
import math
import itertools
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import random


# split data into training and testing part
def train_test_split(df,prop_train=2/3):
    n = len(df)
    idx = list(range(n))
    random.shuffle(idx) # shuffle the index
    n_train = math.ceil(prop_train*n) 

    # split the data into training and test sets
    idx_train = idx[0:n_train]
    idx_test = idx[n_train:]
    training_df = df.loc[idx_train]
    test_df = df.loc[idx_test]

    return training_df, test_df


def ATE_unadjusted(T, Y):
  x = defaultdict(list)
  for t, y in zip(T, Y):
      x[t].append(y)
  T0 = np.mean(x[0])
  T1 = np.mean(x[1])
  return T1 - T0


def ATE_adjusted(C, T, Y):
  x = defaultdict(list)
  for c, t, y in zip(C, T, Y):
      x[c, t].append(y)

  C0_ATE = np.mean(x[0,1]) - np.mean(x[0,0])
  # print(C0_ATE)
  C1_ATE = np.mean(x[1,1]) - np.mean(x[1,0])
  # print(C1_ATE)
  return np.mean([C0_ATE, C1_ATE])


def sigmoid(x):
    out = 1/(1 + math.exp(-x))

    return out


def propensity_estimator(T,C):
    ''' Estimate propensity scores for the confounder C'''
    ''' Use proportion of each strata of the confound to estimate the probability'''
    data = pd.DataFrame({'C':C, 'T':T})
    T_levels = set(T)
    propensities = []
    for c_level in set(C):
        subset = data.loc[data.C == c_level]
        
        pi_C = [
            float(len(subset.loc[subset['T'] == t])) / len(subset) 
            for t in T_levels
        ]
        print(pi_C)
        propensities.append(pi_C[1])

    return propensities 



# beta_t:  makes treatment sepearte more (i.e. give more 1's)
# beta_c: 1, 10, 100, makes confound (buzzy/not) seperate more (drives means apart)
# gamma: 0 , 1, 4, noise level
# beta_o: moves propensities towards the middle so sigmoid can split them into some noise
def sample_Y(C, T, beta_c=0.5, beta_t = 10, beta_o = 0.75, gamma = 0.0):
    propensities = propensity_estimator(T,C)
    ''' Sample Y from known model'''

    Ys = []

    for C_k, T_k in zip(C,T):
        noise = np.random.normal(0, 1)
        y0 = beta_c * (propensities[C_k] - beta_o)
        y1 = beta_t*T_k + y0
        y = y1 + gamma * noise # gamma
        simulated_prob = sigmoid(y)
        threshold = np.random.uniform(0, 1)
        Y = int(simulated_prob > threshold)
        Ys.append(Y)

    return Ys

def sample_cts_Y(C, T, beta_c=0.5, beta_t = 10, beta_o = 0.75, gamma = 0.5):
    propensities = propensity_estimator(T,C)
    ''' Sample Y from known model'''

    Ys = []

    for C_k, T_k in zip(C,T):
        noise = np.random.normal(0, 1)
        y0 = beta_c * (propensities[C_k] - beta_o)
        y1 = beta_t*T_k + y0
        Y = y1 + gamma * noise # gamma
        Ys.append(Y)

    return Ys


def adjust_propensity(df, target_propensities):
    # subset to to desired propensities (must be smaller than true)

    for k, pi_target in enumerate(target_propensities):
        # drop enough samples so that we get the desired propensity
        # inverse of y = x / (x + out sample size) gives you target number for proportion
        Ck_subset = df.loc[df.C == k]
        Ck_T0_subset = Ck_subset.loc[Ck_subset['T'] == 0]
        Ck_T1_subset = Ck_subset.loc[Ck_subset['T'] == 1]
        target_num = len(Ck_T0_subset) * pi_target / (1-pi_target)
        drop_prop = (len(Ck_T1_subset) - target_num) / len(Ck_T1_subset)
        df = df.drop(Ck_T1_subset.sample(frac=drop_prop).index)
    return df




# preprocess and get simulated data
def run_simulation(data,
    propensities=[0.9, 0.7],
    beta_t=0.5,
    beta_c=10,
    gamma=0.0,
    size=-1,
    cts=False):
    """ T: 5 stars review (1) or 1 or 2 stars (0)
        C: CD/MP3 (1) or not (0)
        Y: simulate from f(T,C)
    """

    ''' adjust data to owning target propensities '''
    data = adjust_propensity(data, propensities)

    ''' compute offset '''
    n = len(data)
    my_offset = propensities[1]*len(data.loc[data.C==1])/n + propensities[0]*len(data.loc[data.C==0])/n
    print('Offset:',my_offset)

    ''' get simulated outcome Y '''
    if cts:
      data['Y'] = sample_cts_Y(data.C, data['T'], beta_c=beta_c, beta_t = beta_t, beta_o = my_offset, gamma = gamma)
    else:
      data['Y'] = sample_Y(data.C, data['T'], beta_c=beta_c, beta_t=beta_t, beta_o=my_offset, gamma=gamma)

    if size > 0:
        sample_size = float(size) / len(data)
        data = data.sample(frac=sample_size)

    ''' reindex to rm phantom rows '''
    data = data.reset_index()

    return data, my_offset


def oracle_ate(C, propensities=[0.9, 0.6],
               beta_t=0.8,
               beta_c=4.0,
               gamma=1.0,
               beta_o=0.9,
               cts = True):
  Y0s = []
  Y1s = []

  for C_k in C:
    noise = np.random.normal(0, 1)
    # draw from T=0
    y0 = beta_c * (propensities[C_k] - beta_o) + gamma * noise
    simulated_prob_0 = sigmoid(y0)
    threshold = np.random.uniform(0, 1)
    Y0 = int(simulated_prob_0 > threshold)

    # draw from T=1
    y1 = y0 + beta_t
    simulated_prob_1 = sigmoid(y1)
    threshold = np.random.uniform(0, 1)
    Y1 = int(simulated_prob_1 > threshold)

    if cts:
        Y0s.append(y0)
        Y1s.append(y1)
    else:
        Y0s.append(Y0)
        Y1s.append(Y1)
  
  ate = sum(Y1s)/len(Y1s)-sum(Y0s)/len(Y0s)

  return ate







