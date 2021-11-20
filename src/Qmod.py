from collections import defaultdict

import torch
import torch.nn as nn

from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

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

from tqdm import tqdm
import math
import random

import os.path as osp


CUDA = (torch.cuda.device_count() > 0)
MASK_IDX = 103

#   error function
def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class CausalQBaseline(DistilBertPreTrainedModel):
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



class QBaselineWrapper:
    """Train the model and do inference"""
    def __init__(self, a_weight = 1.0, y_weight=1.0,
                 mlm_weight=1.0,
                 num_neighbors = 100,
                 batch_size = 32,
                 modeldir = None):

        self.model = CausalQBaseline.from_pretrained(
            '/home/glin6/causal_text/pretrained_model/model/',
            num_labels = 2,
            output_attentions=False,
            output_hidden_states=False)
        # self.model = CausalQBaseline.from_pretrained(
        #     '/content/drive/MyDrive/causal_text/model/',
        #     num_labels = 2,
        #     output_attentions=False,
        #     output_hidden_states=False)

        if CUDA:
            self.model = self.model.cuda()

        self.loss_weights = {
            'a': a_weight,
            'y': y_weight,
            'mlm': mlm_weight}

        self.batch_size = batch_size

        self.propensities_knn = KNeighborsClassifier(n_neighbors=num_neighbors)

        kernel = DotProduct() + WhiteKernel()
        self.propensities_gp = GaussianProcessClassifier(kernel=kernel, random_state=0)


        self.modeldir = modeldir


    def build_dataloader(self, texts, treatments, confounders=None, outcomes=None, tokenizer=None, sampler='random'):

        # fill with dummy values
        if outcomes is None:
            outcomes = [-1 for _ in range(len(treatments))]

        if tokenizer is None:
            tokenizer = DistilBertTokenizer.from_pretrained(
               '/home/glin6/causal_text/pretrained_model/tokenizer/', do_lower_case=True)
            # tokenizer = DistilBertTokenizer.from_pretrained(
            #    '/content/drive/MyDrive/causal_text/tokenizer/', do_lower_case=True)

        out = defaultdict(list)
        for i, (W, A, C, Y) in enumerate(zip(texts, treatments, confounders, outcomes)):
            # out['W_raw'].append(W)
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
    
    def train(self, texts, treatments, confounders, outcomes, learning_rate=2e-5, epochs=1, patience=4):
        ''' Train the baseline model'''

        # split data into two parts
        idx = list(range(len(texts)))
        random.shuffle(idx) # shuffle the index
        n_train = int(len(texts)*0.8) 
        n_val = len(texts)-n_train
        idx_train = idx[0:n_train]
        idx_val = idx[n_train:]

        # list of data
        # train_dataloader = self.build_dataloader(texts, treatments, outcomes)
        train_dataloader = self.build_dataloader(texts[idx_train], 
            treatments[idx_train], confounders[idx_train], outcomes[idx_train])
        val_dataloader = self.build_dataloader(texts[idx_val], 
            treatments[idx_val], confounders[idx_val], outcomes[idx_val], sampler='sequential')
        

        self.model.train() # the baseline model
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
                  'mlm loss': mlm_loss.item(), })

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
                _, _, mlm_loss, y_loss, a_loss, a_acc = self.model(text_id, text_len, text_mask, A, Y)
                
                a_val_losses.append(a_loss.item())
                y_val_losses.append(y_loss.item())

                # A accuracy
                a_acc = torch.round(a_acc*len(A))
                a_val_accs.append(a_acc.item())


            # print(val_losses)
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
                torch.save(self.model, self.modeldir)
                # torch.save(self.model,'/content/drive/MyDrive/causal_text/q_basemodel.pt')
                best_loss = val_loss
                epochs_no_improve = 0              
            else:
                epochs_no_improve += 1
           
            if epoch >= 5 and epochs_no_improve >= patience:
                print('Early stopping!' )
                print('The number of epochs is:', epoch)
                break

        # load the saved best model
        self.model = torch.load(self.modeldir)
        # self.model = torch.load('/content/drive/MyDrive/causal_text/q_basemodel.pt')

        return self.model


    def get_Q(self, texts, treatments, confounders=None, outcomes=None):
        self.model.eval()
        dataloader = self.build_dataloader(texts, treatments, confounders, outcomes, sampler='sequential')
        As, Cs, Ys = [], [], []
        Q0s = [] # E[Y|A=0, text]
        Q1s = [] # E[Y|A=1, text]

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Baseline Statistics computing")
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
            
        Q0s = [item for sublist in Q0s for item in sublist]
        Q1s = [item for sublist in Q1s for item in sublist]

        Q_probs = np.array(list(zip(Q0s, Q1s)))
        As = np.array(As)
        Ys = np.array(Ys)

        return Q_probs, As, Ys, Cs
        
    def train_propensities(self, texts, treatments, confounders=None, outcomes=None):
      '''Train the g model directly on testing data'''
      Q_mat, As , Ys, _ = self.get_Q(texts, treatments, confounders, outcomes)

      # knn to build propensity model
      self.propensities_knn.fit(Q_mat, As)
      print(self.propensities_knn.score(Q_mat, As))

      # Gaussian Process
      self.propensities_gp.fit(Q_mat, As)
      print(self.propensities_gp.score(Q_mat, As))

      return self.propensities_knn, self.propensities_gp


    def refit_Q(self, Q_mat, treatments, outcomes=None):
        '''Use knn to fit and get Q0 Q1
        Q_mat, tretaments and outcomes are all numpy array'''
        A0_indices = np.where(treatments == 0)[0]
        A1_indices = np.where(treatments == 1)[0]

        # Q1 model
        Y1 = outcomes[A1_indices]
        knn_q1 = KNeighborsClassifier(n_neighbors=50).fit(Q_mat[A1_indices,:], Y1)
        Q1 = knn_q1.predict_proba(Q_mat)

        # Q0 model
        Y0 = outcomes[A0_indices]
        knn_q0 = KNeighborsClassifier(n_neighbors=50).fit(Q_mat[A0_indices,:], Y0)
        Q0 = knn_q0.predict_proba(Q_mat)

        return Q1, Q0


    def statistics_computing(self, texts, treatments, confounders, outcomes):
        Q_mat, As , Ys, Cs = self.get_Q(texts, treatments, confounders, outcomes)
        Q0, Q1 = Q_mat[:,0], Q_mat[:,1]

        # get propensities
        gs_knn = self.propensities_knn.predict_proba(Q_mat)[:,1]
        gs_gp = self.propensities_gp.predict_proba(Q_mat)[:,1]

        # matching
        # info_mat = np.array(list(zip(As, Ys, gs)))
        # phi_match= find_match_pair(info_mat)
        # ate_match = phi_match.mean()
        # sd_match = phi_match.std()/math.sqrt(len(phi_match))

        # refit (Q0,Q1)


        # get ATE
        n = len(Ys)
        phi_BD = Q1 - Q0
        ate_BD = phi_BD.mean()
        sd_BD = phi_BD.std()/math.sqrt(n)

        phi_AIPTW_knn = phi_BD + (Ys - Q1)/gs_knn*As - (Ys - Q0)/(1-gs_knn)*(1-As)
        ate_AIPTW_knn = phi_AIPTW_knn.mean()
        sd_AIPTW_knn = phi_AIPTW_knn.std()/math.sqrt(n)

        phi_AIPTW_gp = phi_BD + (Ys - Q1)/gs_gp*As - (Ys - Q0)/(1-gs_gp)*(1-As)
        ate_AIPTW_gp = phi_AIPTW_gp.mean()
        sd_AIPTW_gp = phi_AIPTW_gp.std()/math.sqrt(n)


        ate_BD = [ate_BD, sd_BD]
        ate_AIPTW_knn = [ate_AIPTW_knn, sd_AIPTW_knn]
        ate_AIPTW_gp = [ate_AIPTW_gp, sd_AIPTW_gp]
        # ate_match = [ate_match, sd_match]

        return gs_knn, gs_gp, Q_mat, ate_BD, ate_AIPTW_knn, ate_AIPTW_gp, As, Ys, Cs




        
