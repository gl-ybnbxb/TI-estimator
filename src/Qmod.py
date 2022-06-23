'''
Implementation of the TI estimator from
"Causal Estimation for Text Data with Apparent Overlap Violations"
'''
import math
import random
import os.path as osp
from collections import defaultdict

import torch
import torch.nn as nn

from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import DistilBertTokenizer
from transformers import DistilBertModel
from transformers import DistilBertPreTrainedModel


import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, RBF
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from tqdm import tqdm

from util import *


CUDA = (torch.cuda.device_count() > 0)
device = ("cuda" if torch.cuda.is_available() else "cpu")
MASK_IDX = 103


''' The first stage QNet'''
class CausalQNet(DistilBertPreTrainedModel):
    """ QNet model to estimate the conditional outcomes for the first stage
        Note the outcome Y is continuous """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vocab_size = config.vocab_size
        self.distilbert = DistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.g_hat = nn.Linear(config.hidden_size, self.num_labels)
        self.Q_cls = nn.ModuleDict()
        for A in range(2):
          self.Q_cls['%d' % A] = nn.Sequential(
          nn.Linear(config.hidden_size, 200),
          nn.ReLU(),
          nn.Linear(200, 1))

        self.init_weights()

    def forward(self, text_ids, text_len, text_mask, A, Y, use_mlm=True):
        text_len = text_len.unsqueeze(1) - 2  # -2 because of the +1 below
        attention_mask_class = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
        mask = (attention_mask_class(text_len.shape).uniform_() * text_len.float()).long() + 1  # + 1 to avoid CLS
        target_words = torch.gather(text_ids, 1, mask)
        mlm_labels = torch.ones(text_ids.shape).long() * -100
        if CUDA:
            mlm_labels = mlm_labels.cuda()
        mlm_labels.scatter_(1, mask, target_words)
        text_ids.scatter_(1, mask, MASK_IDX)

        # distilbert output
        bert_outputs = self.distilbert(input_ids=text_ids, attention_mask=text_mask)
        seq_output = bert_outputs[0]
        pooled_output = seq_output[:, 0]  # CLS token

        # masked language modeling objective
        if use_mlm:
            prediction_logits = self.vocab_transform(seq_output)  # (bs, seq_length, dim)
            prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
            mlm_loss = CrossEntropyLoss(reduction='sum')(prediction_logits.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            mlm_loss = None

        # sm = nn.Softmax(dim=1)

        # A ~ text
        a_text = self.g_hat(pooled_output)
        a_loss = CrossEntropyLoss(reduction='sum')(a_text.view(-1, 2), A.view(-1))
        # accuracy
        a_pred = a_text.argmax(dim=1)
        a_acc = (a_pred == A).sum().float()/len(A) 
        
        # Y ~ text+A
        # conditional expected outcomes
        Q0 = self.Q_cls['0'](pooled_output)
        Q1 = self.Q_cls['1'](pooled_output)

        if Y is not None:
            A0_indices = (A == 0).nonzero().squeeze()
            A1_indices = (A == 1).nonzero().squeeze()
            # Y loss
            y_loss_A1 = (((Q1.view(-1)-Y)[A1_indices])**2).sum()
            y_loss_A0 = (((Q0.view(-1)-Y)[A0_indices])**2).sum()
            y_loss = y_loss_A0 + y_loss_A1
        else:
            y_loss = 0.0

        return Q0, Q1, mlm_loss, y_loss, a_loss, a_acc

class QNet:
    """Model wrapper for training Qnet and get Q's for new data"""
    def __init__(self, a_weight = 1.0, y_weight=1.0, mlm_weight=1.0,
                 batch_size = 64,
                 modeldir = None):

        self.model = CausalQNet.from_pretrained(
            'distilbert-base-uncased',
            num_labels = 2,
            output_attentions=False,
            output_hidden_states=False)

        if CUDA:
            self.model = self.model.cuda()

        self.loss_weights = {
            'a': a_weight,
            'y': y_weight,
            'mlm': mlm_weight}

        self.batch_size = batch_size
        self.modeldir = modeldir


    def build_dataloader(self, texts, treatments, confounders=None, outcomes=None, tokenizer=None, sampler='random'):

        # fill with dummy values
        if outcomes is None:
            outcomes = [-1 for _ in range(len(treatments))]

        if tokenizer is None:
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

        out = defaultdict(list)
        for i, (W, A, C, Y) in enumerate(zip(texts, treatments, confounders, outcomes)):
            encoded_sent = tokenizer.encode_plus(W, add_special_tokens=True,
                                                max_length=128,
                                                truncation=True,
                                                pad_to_max_length=True)

            out['text_id'].append(encoded_sent['input_ids'])
            out['text_mask'].append(encoded_sent['attention_mask'])
            out['text_len'].append(sum(encoded_sent['attention_mask']))
            out['A'].append(A)
            out['C'].append(C)
            out['Y'].append(Y)

        data = (torch.tensor(out[x]) for x in ['text_id', 'text_len', 'text_mask', 'A', 'C','Y'])
        data = TensorDataset(*data)
        sampler = RandomSampler(data) if sampler == 'random' else SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader
    
    def train(self, texts, treatments, confounders, outcomes, learning_rate=2e-5, epochs=1, patience=5):
        ''' Train the model'''

        # split data into two parts: one for training and the other for validation
        idx = list(range(len(texts)))
        random.shuffle(idx) # shuffle the index
        n_train = int(len(texts)*0.8) 
        n_val = len(texts)-n_train
        idx_train = idx[0:n_train]
        idx_val = idx[n_train:]

        # list of data
        train_dataloader = self.build_dataloader(texts[idx_train], 
            treatments[idx_train], confounders[idx_train], outcomes[idx_train])
        val_dataloader = self.build_dataloader(texts[idx_val], 
            treatments[idx_val], confounders[idx_val], outcomes[idx_val], sampler='sequential')
        

        self.model.train() 
        optimizer = AdamW(self.model.parameters(), lr = learning_rate, eps=1e-8)

        best_loss = 1e6
        epochs_no_improve = 0

        for epoch in range(epochs):
            losses = []
            self.model.train()
        
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),desc='Training')
            for step, batch in pbar:
                if CUDA:
                    batch = (x.cuda() for x in batch)
                text_id, text_len, text_mask, A, _, Y = batch
            
                self.model.zero_grad()
                _, _, mlm_loss, y_loss, a_loss, a_acc = self.model(text_id, text_len, text_mask, A, Y)

                # compute loss
                loss = self.loss_weights['a'] * a_loss + self.loss_weights['y'] * y_loss + self.loss_weights['mlm'] * mlm_loss
                
                       
                pbar.set_postfix({'Y loss': y_loss.item(),
                  'A loss': a_loss.item(), 'A accuracy': a_acc.item(), 
                  'mlm loss': mlm_loss.item()})

                # optimizaion for the baseline
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            # evaluate validation set
            self.model.eval()
            pbar = tqdm(val_dataloader, total=len(val_dataloader), desc='Validating')
            a_val_losses, y_val_losses, a_val_accs = [], [], []
        
            for batch in pbar:
                if CUDA:
                    batch = (x.cuda() for x in batch)
                text_id, text_len, text_mask, A, _, Y = batch
                _, _, _, y_loss, a_loss, a_acc = self.model(text_id, text_len, text_mask, A, Y, use_mlm=False)
                
                a_val_losses.append(a_loss.item())
                y_val_losses.append(y_loss.item())

                # A accuracy
                a_acc = torch.round(a_acc*len(A))
                a_val_accs.append(a_acc.item())


            a_val_loss = sum(a_val_losses)/n_val
            print('A Validation loss:',a_val_loss)

            y_val_loss = sum(y_val_losses)/n_val
            print('Y Validation loss:',y_val_loss)

            val_loss = self.loss_weights['a'] * a_val_loss + self.loss_weights['y'] * y_val_loss
            print('Validation loss:',val_loss)

            a_val_acc = sum(a_val_accs)/n_val
            print('A accuracy:',a_val_acc)


            # early stop
            if val_loss < best_loss:
                torch.save(self.model, self.modeldir+'_bestmod.pt') # save the best model
                best_loss = val_loss
                epochs_no_improve = 0              
            else:
                epochs_no_improve += 1
           
            if epoch >= 5 and epochs_no_improve >= patience:              
                print('Early stopping!' )
                print('The number of epochs is:', epoch)
                break

        # load the best model as the model after training
        self.model = torch.load(self.modeldir+'_bestmod.pt')

        return self.model


    def get_Q(self, texts, treatments, confounders=None, outcomes=None, model_dir=None):
        '''Get conditional expected outcomes Q0 and Q1 based on the training model'''
        dataloader = self.build_dataloader(texts, treatments, confounders, outcomes, sampler='sequential')
        As, Cs, Ys = [], [], []
        Q0s = []  # E[Y|A=0, text]
        Q1s = []  # E[Y|A=1, text]
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Statistics computing")

        if not model_dir:
            self.model.eval()
            for step, batch in pbar:
                if CUDA:
                    batch = (x.cuda() for x in batch)
                text_id, text_len, text_mask, A, C, Y = batch
                Q0, Q1, _, _, _, _ = self.model(text_id, text_len, text_mask, A, Y, use_mlm = False)
                As += A.detach().cpu().numpy().tolist()
                Cs += C.detach().cpu().numpy().tolist()
                Ys += Y.detach().cpu().numpy().tolist()
                Q0s += Q0.detach().cpu().numpy().tolist()
                Q1s += Q1.detach().cpu().numpy().tolist()
        else:
            Qmodel = torch.load(model_dir)
            Qmodel.eval()
            for step, batch in pbar:
                if CUDA:
                    batch = (x.cuda() for x in batch)
                text_id, text_len, text_mask, A, C, Y = batch
                Q0, Q1, _, _, _, _ = Qmodel(text_id, text_len, text_mask, A, Y, use_mlm = False)
                As += A.detach().cpu().numpy().tolist()
                Cs += C.detach().cpu().numpy().tolist()
                Ys += Y.detach().cpu().numpy().tolist()
                Q0s += Q0.detach().cpu().numpy().tolist()
                Q1s += Q1.detach().cpu().numpy().tolist()

        Q0s = [item for sublist in Q0s for item in sublist]
        Q1s = [item for sublist in Q1s for item in sublist]
        As = np.array(As)
        Ys = np.array(Ys)
        Cs = np.array(Cs)

        return Q0s, Q1s, As, Ys, Cs

    
def k_fold_fit_and_predict(read_data_dir, save_data_dir,
                        a_weight, y_weight, mlm_weight, model_dir, 
                        n_splits:int, lr=2e-5, batch_size=64):
    """
    Implements K fold cross-fitting for the model predicting the outcome Y. 
    That is, 
    1. Split data into K folds
    2. For each fold j, the model is fit on the other K-1 folds
    3. The fitted model is used to make predictions for each data point in fold j
    Returns two arrays containing the predictions for all units untreated, all units treated 
    """

    # get data
    df = pd.read_csv(read_data_dir)
    n_df = len(df)

    # initialize summary statistics
    Q0s = np.array([np.nan]*n_df, dtype = float)
    Q1s = np.array([np.nan]*n_df, dtype = float)
    As = np.array([np.nan]*n_df, dtype = float)
    Ys = np.array([np.nan]*n_df, dtype = float)
    Cs = np.array([np.nan]*n_df, dtype = float)


    # get k folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    idx_array = np.array(range(n_df))


    # train in k-folding fashion
    for train_index, test_index in kf.split(idx_array):
        # training df
        train_df = df.loc[train_index]
        train_df = train_df.reset_index()

        # test df
        test_df = df.loc[test_index]
        test_df = test_df.reset_index()

        # train the model with training data and train the propensitiy model with the testing data
        mod = QNet(batch_size=batch_size, a_weight=a_weight, y_weight=y_weight, mlm_weight=mlm_weight, modeldir=model_dir)
        mod.train(train_df['text'], train_df['T'], train_df['C'], train_df['Y'], epochs=20, learning_rate = lr)

        # g, Q, A, Y, C for the this test part (best model)
        Q0, Q1, A, Y, C = mod.get_Q(test_df['text'], test_df['T'], test_df['C'], test_df['Y'], model_dir=model_dir+'_bestmod.pt')
        Q0s[test_index] = Q0
        Q1s[test_index] = Q1
        As[test_index] = A
        Ys[test_index] = Y
        Cs[test_index] = C


        # delete models for this part
        os.remove(moddir + '_bestmod.pt')


    # if there's nan in Q0/Q1, raise error
    assert np.isnan(Q0s).sum() == 0
    assert np.isnan(Q1s).sum() == 0


    # save Q0, Q1, A, Y, C from the best model into a file
    stats_info = np.array(list(zip(Q0s, Q1s, As, Ys, Cs)))
    stats_info = pd.DataFrame(stats_info, columns=['Q0','Q1','A','Y','C'])
    stats_info.to_csv(save_data_dir, index = False)

    return


def get_agg_q(data_dir_dict, save_data_dir):
    '''Get aggregated conditional expected outcomes
        data_dir_dict is a dictionary that each seed has a corresponding data directory'''

    
    k = len(data_dir_dict)
    Q0s, Q1s, As, Ys, Cs = [], [], [], [], []

    for seed in data_dir_dict.keys():
        df = pd.read_csv(data_dir_dict[seed])
        Q0, Q1, A, Y, C = list(df['Q0']), list(df['Q1']), list(df['A']), list(df['Y']), list(df['C'])
        Q0s += [Q0]
        Q1s += [Q1]
        As += [A]
        Ys += [Y]
        Cs += [C]

    # check the data with the same index is the same one for all seeds
    for i in range(k-1):
        assert Ys[i]==Ys[i+1]
        assert As[i]==As[i+1]
        assert Cs[i]==Cs[i+1]

    Q0s, Q1s = np.array(Q0s), np.array(Q1s)
    A, Y, C = np.array(A), np.array(Y), np.array(C)
    Q0_agg, Q1_agg = np.sum(Q0s, axis=0)/k, np.sum(Q1s, axis=0)/k
        
    # save the aggregated data
    df_agg = df[['A', 'Y', 'C']].copy()
    df_agg['Q0'] = Q0_agg.copy()
    df_agg['Q1'] = Q1_agg.copy()
    df_agg.to_csv(save_data_dir, index=False)

    return
        


''' The second stage: propensity scores estimation '''
def get_propensities(As, Q0s, Q1s, model_type='GaussianProcessRegression', kernel=None, random_state=0, n_neighbors=100, base_estimator=None):
    """Train the propensity model directly on the data 
    and compute propensities of the data"""

    Q_mat = np.array(list(zip(Q0s, Q1s)))

    if model_type == 'GaussianProcessRegression':
        if kernel == None:
            kernel = DotProduct() + WhiteKernel()
        propensities_mod = GaussianProcessClassifier(kernel=kernel, random_state=random_state, warm_start=True)
        propensities_mod.fit(Q_mat, As)

        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'KNearestNeighbors':
        propensities_mod = KNeighborsClassifier(n_neighbors=n_neighbors)
        propensities_mod.fit(Q_mat, As)
        
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'DecisionTree':
        propensities_mod = DecisionTreeClassifier(random_state=random_state)
        propensities_mod.fit(Q_mat, As)
        
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'AdaBoost':
        propensities_mod = AdaBoostClassifier(base_estimator = base_estimator, random_state=random_state)
        propensities_mod.fit(Q_mat, As)
        
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'Bagging':
        propensities_mod = BaggingClassifier(base_estimator = base_estimator, random_state=random_state)
        propensities_mod.fit(Q_mat, As)
        
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'Logistic':
        propensities_mod = LogisticRegression(random_state=random_state)
        propensities_mod.fit(Q_mat, As)
        
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    return gs


''' The third stage: get TI estimator '''
def get_TI_estimator(gs, Q0s, Q1s, As, Ys, error=0.05):
    '''Get TI estimator '''
    try:
        try_est = att_aiptw(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=gs, error_bound=error)    
    except:
        print('There is 0/1 in propensity scores!')
    else:
        ti_estimate = pd.DataFrame(try_est, index = ['point estimate', 'standard error', 'confidence interval lower bound', 'confidence interval upper bound'])
        return ti_estimate


def get_estimands(gs, Q0s, Q1s, As, Ys, Cs=None, alpha=1, error=0.05, g_true=[0.8,0.6]):
    """ Get different estimands based on propensity scores, conditional expected outcomes, treatments and outcomes """
    estimands = []

    estimands.append(('unadj_T', [ATE_unadjusted(As, Ys)] + [np.nan] * 3))
    estimands.append(('adj_T', [ATE_adjusted(Cs, As, Ys)] + [np.nan] * 3))
    idx = (0.1 < gs) * (gs < 0.90)

    # Q only ATE
    ATE_Q = ate_aiptw(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=None, weight=False, error_bound=error)
    estimands.append(('ate_Q', ATE_Q))


    # ATE AIPTW
    try:
        try_est = ate_aiptw(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=gs, weight=False,error_bound=error)    
    except:
        estimands.append(('ate_AIPTW', [np.nan]*4))
    else:
        estimands.append(('ate_AIPTW', try_est))

  
    # trimmed ATE AIPTW
    try:
        try_est = ate_aiptw(Q0=Q0s[idx], Q1=Q1s[idx], A=As[idx], Y=Ys[idx], g=gs[idx], weight=False,error_bound=error)    
    except:
        estimands.append(('trimmed ate_AIPTW', [np.nan]*4))
    else:
        estimands.append(('trimmed ate_AIPTW', try_est))


    # BMM
    try:
        bmm_ate = bmm(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=gs, alpha=alpha,error_bound=error)    
    except:
        estimands.append(('ate_BMM', [np.nan]*4))
    else:
        estimands.append(('ate_BMM', bmm_ate))


    # trimmed BMM
    try:
        bmm_ate = bmm(Q0=Q0s[idx], Q1=Q1s[idx], A=As[idx], Y=Ys[idx], g=gs[idx], alpha=alpha,error_bound=error)    
    except:
        estimands.append(('trimmed ate_BMM', [np.nan]*4))
    else:
        estimands.append(('trimmed ate_BMM', bmm_ate))


    # ATE IPTW
    try:
        try_est = ate_iptw(As, Ys, gs, error_bound=error)  
    except:
        estimands.append(('ate_IPTW', [np.nan]*4))
    else:
        estimands.append(('ate_IPTW', try_est))


    # trimmed ATE IPTW
    try:
        try_est = ate_iptw(As[idx], Ys[idx], gs[idx], error_bound=error)  
    except:
        estimands.append(('trimmed ate_IPTW', [np.nan]*4))
    else:
        estimands.append(('trimmed ate_IPTW', try_est))


    # ATT Q only
    try:
        try_est = att_q(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, error_bound=error)    
    except:
        estimands.append(('att_Q', [np.nan]*4))
    else:
        estimands.append(('att_Q', try_est))


    # ATT AIPTW
    try:
        try_est = att_aiptw(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=gs, error_bound=error)    
    except:
        estimands.append(('att_AIPTW', [np.nan]*4))
    else:
        estimands.append(('att_AIPTW', try_est))


    # trimmed ATT AIPTW
    try:
        try_est = att_aiptw(Q0=Q0s[idx], Q1=Q1s[idx], A=As[idx], Y=Ys[idx], g=gs[idx], error_bound=error)    
    except:
        estimands.append(('trimmed att_AIPTW', [np.nan]*4))
    else:
        estimands.append(('trimmed att_AIPTW', try_est))


    # ATT BMM
    try:
        try_est = att_bmm(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=gs, error_bound=error)   
    except:
        estimands.append(('att_BMM', [np.nan]*4))
    else:
        estimands.append(('att_BMM', try_est))


    # trimmed ATT BMM
    try:
        try_est = att_bmm(Q0=Q0s[idx], Q1=Q1s[idx], A=As[idx], Y=Ys[idx], g=gs[idx], error_bound=error)   
    except:
        estimands.append(('trimmed att_BMM', [np.nan]*4))
    else:
        estimands.append(('trimmed att_BMM', try_est))


    estimands = pd.DataFrame(data=dict(estimands))
    return estimands       




        
