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
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.append('/home/glin6/scratch-midway2/causal_text/src/')
from Qmod import *
from util import *


''' Target:
 hyperpaameters are chosen by cross validation
 compared to baselines & see the probabilities' change'''


def k_fold_fit_and_predict(args, n_splits:int):
    """
    Implements K fold cross-fitting for the model predicting the outcome Y. 
    That is, 
    1. Split data into K folds
    2. For each fold j, the model is fit on the other K-1 folds
    3. The fitted model is used to make predictions for each data point in fold j
    Returns two arrays containing the predictions for all units untreated, all units treated  
    """

    # get data
    data_dir = osp.join(args.datadir,'beta_t_%.1f_beta_c_%.1f_gamma_%.1f'%(args.beta_t, args.beta_c, args.gamma))
    data_dir = data_dir+'processed.csv'
    df = pd.read_csv(data_dir)

    # path to save the model
    moddir = osp.join(args.resultsdir,'T_%d_C_%d_gamma_%d_lr_%.0e'%(args.beta_t, args.beta_c, args.gamma, args.lr))
    moddir = moddir.replace('.', '_')
    # moddir = moddir+'qmod.pt'
    
    n_df = len(df)
    # initialize summary statistics
    gs = np.array([np.nan]*n_df, dtype = float)
    Q0s = np.array([np.nan]*n_df, dtype = float)
    Q1s = np.array([np.nan]*n_df, dtype = float)
    As = np.array([np.nan]*n_df, dtype = float)
    Ys = np.array([np.nan]*n_df, dtype = float)
    Cs = np.array([np.nan]*n_df, dtype = float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    idx_array = np.array(range(n_df))
    
    for train_index, test_index in kf.split(idx_array):
        # training df
        train_df = df.loc[train_index]
        train_df = train_df.reset_index()

        # test df
        test_df = df.loc[test_index]
        test_df = test_df.reset_index()

        # train the model with training data and train the propensitiy model with the testing data
        mod = QNet(batch_size=32, a_weight = args.a_weight, y_weight = args.y_weight, mlm_weight=args.mlm_weight, modeldir=moddir)
        mod.train(train_df['text'], train_df['T'], train_df['C'], train_df['Y'], epochs=args.epoch, learning_rate = args.lr)

        # g, Q, A, Y, C for the this test part (best model)
        Q0, Q1, A, Y, C = mod.get_Q(test_df['text'], test_df['T'], test_df['C'], test_df['Y'], model_dir=moddir+'_bestmod.pt')
        Q0s[test_index] = Q0
        Q1s[test_index] = Q1
        As[test_index] = A
        Ys[test_index] = Y
        Cs[test_index] = C

        # delete models for this part
        os.remove(moddir + '_bestmod.pt')

    gs = get_propensities(As, Q0s, Q1s, model_type='GaussianProcessRegression')

    # if there's nan in Q0/Q1/g, raise error
    assert np.isnan(Q0s).sum() == 0
    assert np.isnan(Q1s).sum() == 0
    assert np.isnan(gs).sum() == 0


    # save g, Q0, Q1, A, Y, C from the best model into a file
    stats_info = np.array(list(zip(gs, Q0s, Q1s, As, Ys, Cs)))
    stats_info = pd.DataFrame(stats_info, columns=['g','Q0','Q1','A','Y','C'])
    filename = osp.join(args.resultsdir, 'stats_T_%d_C_%d_gamma_%d_lr_%.0e_seed_%d'%(args.beta_t, args.beta_c, args.gamma, args.lr, args.seed))
    filename = filename.replace('.', '_')+'.csv'
    stats_info.to_csv(filename)

    return gs, Q0s, Q1s, As, Ys, Cs


def run_experiment(args):
    """ Run an experiment with the given args and seed.
        Returns {causal estimator: ATE estimate}
    """

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 

    ATE_estimates = []

    # get data
    data_dir = osp.join(args.datadir,'beta_t_%.1f_beta_c_%.1f_gamma_%.1f'%(args.beta_t, args.beta_c, args.gamma))
    data_dir = data_dir+'processed.csv'
    df = pd.read_csv(data_dir)
  
    # baseline ATE
    ATE_estimates.append(('unadj_T', ATE_unadjusted(df['T'], df['Y'])))
    ATE_estimates.append(('adj_T', ATE_adjusted(df['C'], df['T'], df['Y'])))

    true_propensity = [args.p0, args.p1]

    # Apply cross-fit
    gs, Q0s, Q1s, As, Ys, Cs = k_fold_fit_and_predict(args, n_splits=5)

    # compute the ATE estimates
    ATE_BD = ate_aiptw(Q0s, Q1s, As, Ys, g=None)
    ATE_AIPTW = ate_aiptw(Q0s, Q1s, As, Ys, gs)
    ATE_IPTW = ate_iptw(As, Ys, gs)
    ATE_BMM = bmm(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=gs)

    # organize the ATE dictionary
    ATE_estimates.append(('ate_BD', ATE_BD))
    ATE_estimates.append(('ate_AIPTW', ATE_AIPTW))
    ATE_estimates.append(('ate_IPTW', ATE_IPTW))
    ATE_estimands.append(('ate_BMM', ATE_BMM))
    ATE_estimates = pd.DataFrame(ATE_estimates)

    # scatter plot of (Q0, Q1)
    # filename = osp.join(args.resultsdir, 'QT_%.1f_C_%.1f_gamma_%.0f_lr_%.0e_seed_%d'%(args.beta_t, args.beta_c, args.gamma, args.lr, args.seed))
    # filename = filename.replace('.', '_')+'.png'
    # fig = plt.figure(1)
    # plt.xlabel('Q0')
    # plt.ylabel('Q1')
    # ax = fig.add_subplot(111)
    # ax.scatter(Q0, Q1, marker = 'o', c = A, alpha = 0.8)  
    # plt.savefig(filename)
    # plt.clf()
  

    # save histograms of propensities
    # fig2 = plt.hist(g)
    # plt.title('Propensity Scores (Gaussian Process)')
    # plt.xlabel("value")
    # plt.ylabel("Frequency")
    # filename2 = osp.join(args.resultsdir, 'g_gp_T_%.1f_C_%.1f_gamma_%.0f_lr_%.0e_seed_%d'%(args.beta_t, args.beta_c, args.gamma, args.lr, args.seed))
    # filename2 = filename2.replace('.', '_')+'.png'
    # plt.savefig(filename2)
    # plt.clf()

    return ATE_estimates

def aggregate_sd(sd_vec):
    k = len(sd_vec)
    agg_sd = math.sqrt(sum(np.array(sd_vec)**2)/k)

    return agg_sd


if __name__ == '__main__':
    from argparse import ArgumentParser
    import pandas as pd
    import json
    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser.add_argument('--seed', type=str, default='420', help='Choose random seed')
    parser.add_argument('--resultsdir', type=str, default='/home/glin6/scratch-midway2/causal_text/binary_causal/', help='Path to the results')
    parser.add_argument('--datadir', type=str, default='/home/glin6/scratch-midway2/causal_text/binary_causal/data/', help='Path to saving the simulated dataset')

    # for data generation
    parser.add_argument('--p0', type=float, default=0.8, help='P(T = 0 | C) in simulation (-1 to ignore).')
    parser.add_argument('--p1', type=float, default=0.6, help='P(T = 1 | C) in simulation (-1 to ignore).')
    parser.add_argument('--beta_t', type=float, default=1.0, help='Simulated treatment strength.')
    parser.add_argument('--beta_c', type=float, default=0.1, help='Simulated confound strength.')
    parser.add_argument('--beta_o', type=float, default=1.0, help='Simulated offset for T/C pre-threshold means.')
    parser.add_argument('--gamma', type=float, default=1.0, help='Noise level in simulation')  # 0, 1

    # for building the model
    parser.add_argument('--a_weight', type=float, default=0.1, help='A Loss weight for the Y~text+A function in the baseline model.')
    parser.add_argument('--y_weight', type=float, default=0.1, help='Y Loss weight for the Y~text+A function in the baseline model.')
    parser.add_argument('--mlm_weight', type=float, default=1.0, help='Loss weight for the mlm head in the baseline model.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=20, help='Largest number of epochs for training')
    # parser.add_argument('--num_neighbors', type=int, default=500, help='k in the k-nearest neighbor')
    

    args = parser.parse_args()
    args.seed = int(args.seed)

    # run the experiment
    result = run_experiment(args)

    # save ATE
    filename = osp.join(args.resultsdir, 'ATE_T_%d_C_%d_gamma_%d_lr_%.0e_seed_%d'%(args.beta_t, args.beta_c, args.gamma, args.lr, args.seed))
    filename = filename +'.csv'
    result.to_csv(filename, index=False)

    quit()
