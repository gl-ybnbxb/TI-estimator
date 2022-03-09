from collections import defaultdict

import torch
import torch.nn as nn

from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import SGD

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import DistilBertTokenizer
from transformers import DistilBertModel
from transformers import DistilBertPreTrainedModel

from torch.nn import CrossEntropyLoss

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.tree import DecisionTreeClassifier

from tqdm import tqdm
import math
import random

import os.path as osp
import sys
sys.path.append('/home/glin6/scratch-midway2/causal_text/src/')
from util import *

# print(torch.cuda.device_count())
CUDA = (torch.cuda.device_count() > 0)
device = ("cuda" if torch.cuda.is_available() else "cpu")
MASK_IDX = 103

#   error function
def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class CausalQNet(DistilBertPreTrainedModel):
    """Build the model"""
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

        # bert output
        bert_outputs = self.distilbert(input_ids=text_ids, attention_mask=text_mask)
        seq_output = bert_outputs[0]
        pooled_output = seq_output[:, 0]  # CLS token
        #bert_rep = pooled_output.detach()

        # masked language modeling objective
        if use_mlm:
            prediction_logits = self.vocab_transform(seq_output)  # (bs, seq_length, dim)
            prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
            mlm_loss = CrossEntropyLoss(reduction='sum')(prediction_logits.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            mlm_loss = None

        sm = nn.Softmax(dim=1)
        CE_loss = nn.CrossEntropyLoss(reduction='sum')

        # A ~ text
        a_text = self.g_hat(pooled_output)
        a_loss = CrossEntropyLoss(reduction='sum')(a_text.view(-1, 2), A.view(-1))
        # accuracy
        a_pred = a_text.argmax(dim=1)
        a_acc = (a_pred == A).sum().float()/len(A) 
        
        # Y ~ text+A
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
    """Train the prognostic model"""
    def __init__(self, a_weight = 1.0, y_weight=1.0,
                 mlm_weight=1.0,
                 batch_size = 32,
                 modeldir = None):

        self.model = CausalQNet.from_pretrained(
            '/home/glin6/causal_text/pretrained_model/model/',
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
            tokenizer = DistilBertTokenizer.from_pretrained(
               '/home/glin6/causal_text/pretrained_model/tokenizer/', do_lower_case=True)

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

            # if i > 100: break

        data = (torch.tensor(out[x]) for x in ['text_id', 'text_len', 'text_mask', 'A', 'C','Y'])
        data = TensorDataset(*data)
        sampler = RandomSampler(data) if sampler == 'random' else SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader
    
    def train(self, texts, treatments, confounders, outcomes, learning_rate=2e-5, epochs=1, patience=5):
        ''' Train the model'''

        # split data into two parts
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
        # optimizer = SGD(self.model.parameters(), lr = learning_rate)

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

        # save the overfiting model       
        # torch.save(self.model, self.modeldir+'_overfitmod.pt')

        # load the best model as the model after training
        self.model = torch.load(self.modeldir+'_bestmod.pt')

        return self.model


    def get_Q(self, texts, treatments, confounders=None, outcomes=None, model_dir=None):
        '''Get prognostic scores Q0 and Q1 based on training model or saved model'''
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
        

def get_propensities(As, Q0s, Q1s, model_type='GaussianProcessRegression', kernel=None, random_state=0, n_neighbors=100):
    """Train the propensity model directly on the data 
    and compute propensities of the data"""

    Q_mat = np.array(list(zip(Q0s, Q1s)))

    if model_type == 'GaussianProcessRegression':
        if kernel == None:
            kernel = DotProduct() + WhiteKernel()
        propensities_mod = GaussianProcessClassifier(kernel=kernel, random_state=random_state)
        propensities_mod.fit(Q_mat, As)
        print(propensities_mod.score(Q_mat, As)) # check the prediction score of the propensity model

        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'KNN':
        propensities_mod = KNeighborsClassifier(n_neighbors=n_neighbors)
        propensities_mod.fit(Q_mat, As)
        print(propensities_mod.score(Q_mat, As)) # check the prediction score of the propensity model
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'KernelRegression':
        ksrmv = KernelReg(endog=Q_mat, exog=As, var_type='u')
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'DecisionTree':
        propensities_mod = DecisionTreeClassifier(random_state=random_state)
        propensities_mod.fit(Q_mat, As)
        print(propensities_mod.score(Q_mat, As)) # check the prediction score of the propensity model
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    if model_type == 'Logistic':
        propensities_mod = LogisticRegression(random_state=random_state)
        propensities_mod.fit(Q_mat, As)
        print(propensities_mod.score(Q_mat, As)) # check the prediction score of the propensity model
        # get propensities
        gs = propensities_mod.predict_proba(Q_mat)[:,1]

    return gs



        




        