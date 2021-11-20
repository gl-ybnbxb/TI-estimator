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

sys.path.append('/home/glin6/causal_text/src/cts/')
sys.path.append('/home/glin6/causal_text/src/')
from Qmod import *
from simulation import *


''' Target:
 hyperpaameters are chosen by cross validation
 compared to baselines & see the probabilities' change'''

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
  C1_ATE = np.mean(x[1,1]) - np.mean(x[1,0])
  return np.mean([C0_ATE, C1_ATE])
  

def oracle_AIPTW(C, T, Y, true_propensity):
    C, T, Y = np.array(C), np.array(T), np.array(Y)
    g = np.array([true_propensity[v] for v in C])
    oracle_AIPTW = np.mean(Y/g*T - Y/(1-g)*(1-T))

    return oracle_AIPTW


def run_experiment(args):
    """ Run an experiment with the given args and seed.
        Returns {causal estimator: ATE estimate}
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 

    train_dir = osp.join(args.datadir,'beta_t_%.1f_beta_c_%.1f_gamma_%.1f'%(args.beta_t, args.beta_c, args.gamma))
    train_dir = train_dir+'train.csv'
    training_df = pd.read_csv(train_dir)

    test_dir = osp.join(args.datadir,'beta_t_%.1f_beta_c_%.1f_gamma_%.1f'%(args.beta_t, args.beta_c, args.gamma))
    test_dir = test_dir+'test.csv'
    test_df = pd.read_csv(test_dir)

    ATE_estimates = []
    ATE_sd = []

    
    # baseline ATE
    ATE_estimates.append(('unadj_T', ATE_unadjusted(test_df['T'], test_df['Y'])))
    ATE_estimates.append(('adj_T', ATE_adjusted(test_df['C'], test_df['T'], test_df['Y'])))

    true_propensity = [args.p0, args.p1]
    ATE_estimates.append(('oracle_ATE', oracle_AIPTW(test_df['C'], test_df['T'], test_df['Y'], true_propensity)))


    # my ATE after training the model
    moddir = osp.join(args.resultsdir,'T_%.1f_C_%.1f_gamma_%.0f_lr_%.0e_seed_%d'%(args.beta_t, args.beta_c, args.gamma, args.lr, args.seed))
    moddir = moddir+'qmod.pt'
    mod = QBaselineWrapper(batch_size=32, a_weight = args.a_weight, y_weight = args.y_weight, mlm_weight=args.mlm_weight, 
        num_neighbors=args.num_neighbors,modeldir=moddir)
    mod.train(training_df['text'], training_df['T'], training_df['C'], training_df['Y'], epochs=args.epoch, learning_rate = args.lr)
    mod.train_propensities(test_df['text'], test_df['T'], test_df['C'], test_df['Y'])
    gs_knn, gs_gp, Q_mat, ate_BD, ate_AIPTW_knn, ate_AIPTW_gp, As, Ys, Cs = mod.statistics_computing(texts=test_df['text'], treatments=test_df['T'], confounders=test_df['C'], outcomes=test_df['Y'])
    
    # Add cross fit?


    ATE_estimates.append(('ate_BD', ate_BD[0]))
    ATE_estimates.append(('ate_AIPTW_knn', ate_AIPTW_knn[0]))
    ATE_estimates.append(('ate_AIPTW_gp', ate_AIPTW_gp[0]))
    ATE_sd.append(('ate_BD_sd', ate_BD[1]))
    ATE_sd.append(('ate_AIPTW_knn_sd', ate_AIPTW_knn[1]))
    ATE_sd.append(('ate_AIPTW_gp_sd', ate_AIPTW_gp[1]))

    # scatter plot of (Q0, Q1)
    filename = osp.join(args.resultsdir, 'QT_%.1f_C_%.1f_gamma_%.0f_lr_%.0e_seed_%d'%(args.beta_t, args.beta_c, args.gamma, args.lr, args.seed))
    filename = filename.replace('.', '_')+'.png'
    fig = plt.figure(1)
    plt.xlabel('Q0')
    plt.ylabel('Q1')
    ax = fig.add_subplot(111)
    ax.scatter(Q_mat[:,0], Q_mat[:,1], marker = 'o', c = As, alpha = 0.8)  
    plt.savefig(filename)
    plt.clf()
  

    # save histograms of propensities
    fig2 = plt.hist(gs_knn)
    plt.title('Propensity Scores KNN')
    plt.xlabel("value")
    plt.ylabel("Frequency")
    filename2 = osp.join(args.resultsdir, 'g_knn_T_%.1f_C_%.1f_gamma_%.0f_lr_%.0e_seed_%d'%(args.beta_t, args.beta_c, args.gamma, args.lr, args.seed))
    filename2 = filename2.replace('.', '_')+'.png'
    plt.savefig(filename2)
    plt.clf()

    fig3 = plt.hist(gs_gp)
    plt.title('Propensity Scores Gaussian Process')
    plt.xlabel("value")
    plt.ylabel("Frequency")
    filename3 = osp.join(args.resultsdir, 'g_gp_T_%.1f_C_%.1f_gamma_%.0f_lr_%.0e_seed_%d'%(args.beta_t, args.beta_c, args.gamma, args.lr, args.seed))
    filename3 = filename3.replace('.', '_')+'.png'
    plt.savefig(filename3)
    plt.clf()

    stats_info = np.array(list(zip(gs_knn, gs_gp, Q_mat[:,0], Q_mat[:,1], As, Ys, Cs)))
    stats_info = pd.DataFrame(stats_info, columns=['g_knn','g_gp','Q0','Q1','A','Y','C'])
    filename4 = osp.join(args.resultsdir, 'stats_T_%.1f_C_%.1f_gamma_%.0f_lr_%.0e_seed_%d'%(args.beta_t, args.beta_c, args.gamma, args.lr, args.seed))
    filename4 = filename4.replace('.', '_')+'.csv'
    stats_info.to_csv(filename4)

    # delete the model
    os.remove(moddir)

    return dict(ATE_estimates), dict(ATE_sd)

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
    parser.add_argument('--num_neighbors', type=int, default=500, help='k in the k-nearest neighbor')
    

    args = parser.parse_args()

    if ',' in args.seed:
        seeds = args.seed.split(',')
    else:
        seeds = [args.seed]

    results = defaultdict(list)
    results_sd = defaultdict(list)

    for seed in seeds:
        args.seed = int(seed)
        result, result_sd = run_experiment(args)
        for k, v in result.items():
            results[k] += [v]
        for k,v in result_sd.items():
            results_sd[k] += [v]

    out = {**vars(args), **{k: np.mean(v) for k, v in results.items()}, **{k: aggregate_sd(v) for k, v in results_sd.items()}}

    print('Adjusted:\t%.4f' % out['adj_T'])
    print('Unadjusted:\t%.4f' % out['unadj_T'])
    print('Oracle:\t%.4f' % out['oracle_ATE'])
    print('Q Baseline BD:\t%.4f (%.4f)' % (out['ate_BD'], out['ate_BD_sd']))
    print('Q Baseline AIPTW with knn:\t%.4f (%.4f)' % (out['ate_AIPTW_knn'], out['ate_AIPTW_knn_sd']))
    print('Q Baseline AIPTW with gp:\t%.4f (%.4f)' % (out['ate_AIPTW_gp'], out['ate_AIPTW_gp_sd']))

    # save ATE
    filename = osp.join(args.resultsdir, 'ATE_T_%.1f_C_%.1f_gamma_%.0f_lr_%.0e'%(args.beta_t, args.beta_c, args.gamma, args.lr))
    filename = filename.replace('.', '_')+'.json'
    with open(filename, 'w') as outfile:
        json.dump(out, outfile)


    quit()
