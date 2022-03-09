''' Test for the baseline algorithm'''
from tqdm import tqdm
import sys
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import random
import os.path as osp
import os

import torch

from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn import feature_extraction

sys.path.append('/home/glin6/scratch-midway2/causal_text/src/')
from Qmod import *
from simulation import *


''' Target:
generate simulated data'''


def get_data(args):
    '''Generate simulated data and split it into two parts'''
    random.seed(args.dataseed)
    np.random.seed(args.dataseed)

    raw_df = pd.read_csv(args.data)
    df, offset =run_simulation(raw_df, propensities=[args.p0, args.p1], 
                            beta_t=args.beta_t,  
                            beta_c=args.beta_c,  # 1, 10, 100
                            gamma=args.gamma, # 1.0, 4.0
                            cts=True)  
    args.beta_o = offset
    dat = df[['text','T','C', 'Y']]
    data_dir = osp.join(args.datadir,'beta_t_%.1f_beta_c_%.1f_gamma_%.1f'%(args.beta_t, args.beta_c, args.gamma))
    data_dir = data_dir+'processed.csv'
    dat.to_csv(data_dir, index = False)

    # training_df, test_df = train_test_split(dat,prop_train=0.7)
    # training_df = training_df.reset_index()
    # train_dir = osp.join(args.datadir,'beta_t_%.1f_beta_c_%.1f_gamma_%.1f'%(args.beta_t, args.beta_c, args.gamma))
    # train_dir = train_dir+'train.csv'
    # training_df.to_csv(train_dir, index = False)

    # test_df = test_df.reset_index()
    # test_dir = osp.join(args.datadir,'beta_t_%.1f_beta_c_%.1f_gamma_%.1f'%(args.beta_t, args.beta_c, args.gamma))
    # test_dir = test_dir+'test.csv'
    # test_df.to_csv(test_dir, index = False)

    return 


if __name__ == '__main__':
    from argparse import ArgumentParser
    import pandas as pd
    import json

    parser = ArgumentParser()
    parser.add_argument('--dataseed', type=str, default='420', help='Choose random seed')
    parser.add_argument('--datadir', type=str, default='/home/glin6/scratch-midway2/causal_text/binary_causal/data/', help='Path to saving the simulated dataset')
    parser.add_argument('--data', type=str, default='/home/glin6/scratch-midway2/causal_text/binary_causal/data/', help='Path to the raw dataset')

    # for data generation
    parser.add_argument('--p0', type=float, default=0.8, help='P(T = 0 | C) in simulation (-1 to ignore).')
    parser.add_argument('--p1', type=float, default=0.6, help='P(T = 1 | C) in simulation (-1 to ignore).')
    parser.add_argument('--beta_t', type=float, default=1.0, help='Simulated treatment strength.')
    parser.add_argument('--beta_c', type=float, default=0.1, help='Simulated confound strength.')
    parser.add_argument('--beta_o', type=float, default=1.0, help='Simulated offset for T/C pre-threshold means.')
    parser.add_argument('--gamma', type=float, default=1.0, help='Noise level in simulation')  # 0, 1
    
    args = parser.parse_args()

    args.dataseed = int(args.dataseed)

    get_data(args)


    quit()
