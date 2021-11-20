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

# import torch
# import torch.nn as nn

# from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

# from torch.nn import CrossEntropyLoss

# from tqdm import tqdm
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
    accuracy=-1,
    size=-1,
    cts=False):
    """ T: 5 stars review (1) or 1 or 2 stars (0)
        T*: sentiment/bert predictions/random sampling for T*
        C: CD/MP3 (1) or not (0)
        Y: simulate from f(T,C)
    """

    # Get treatment T from the ground truth rating
    # def treatment_from_rating(rating):
    #     return int(rating == 5.0)

    # data['T'] = data['rating'].apply(treatment_from_rating)


    # # get confound C: 1 cd; 0 mp3 + vinyl
    # C_from_product = lambda p: 1 if p == 'audio cd' else 0
    # data['C'] = data['product'].apply(C_from_product)
    # data['C_from_text'] = sample_C_from_text(data['C_true'], data['text'])

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
    # data['Y_sim'] = sample_Y(data.C_from_text, data.T_true, beta_c=beta_c, beta_t=beta_t, beta_o=beta_o, gamma=gamma)

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



# class CRegressor(DistilBertPreTrainedModel):
#   """Build the model"""
#   def __init__(self, config):
#       super().__init__(config)
#       self.num_labels = config.num_labels
#       self.vocab_size = config.vocab_size
#       self.distilbert = DistilBertModel(config)
#       self.vocab_transform = nn.Linear(config.dim, config.dim)
#       self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
#       self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

#       self.c_linear = nn.Linear(config.hidden_size, self.num_labels)
#       self.init_weights()

#   def forward(self, text_ids, text_len, text_mask, use_mlm=True, loss_computing=True):
#       text_len = text_len.unsqueeze(1) - 2  # -2 because of the +1 below
#       attention_mask_class = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
#       mask = (attention_mask_class(text_len.shape).uniform_() * text_len.float()).long() + 1  # + 1 to avoid CLS
#       target_words = torch.gather(text_ids, 1, mask)
#       mlm_labels = torch.ones(text_ids.shape).long() * -100
#       if torch.cuda.is_available():
#           mlm_labels = mlm_labels.cuda()
#       mlm_labels.scatter_(1, mask, target_words)
#       text_ids.scatter_(1, mask, 103)

#       # bert output
#       bert_outputs = self.distilbert(input_ids=text_ids, attention_mask=text_mask)
#       seq_output = bert_outputs[0]
#       pooled_output = seq_output[:, 0]  # CLS token

#       # C ~ text
#       c_text = self.c_linear(pooled_output)

#       return c_text


# def sample_C_from_text(c_true, text):

#   # split data into two parts
#   idx = list(range(len(c_true)))
#   random.shuffle(idx) # shuffle the index
#   n_train = int(len(c_true)*0.8) 
#   n_val = len(c_true)-n_train
#   idx_train = idx[0:n_train]
#   idx_val = idx[n_train:]

#   # list of data
#   train_dataloader = build_dataloader(text[idx_train], c_true[idx_train])
#   val_dataloader = build_dataloader(text[idx_val], c_true[idx_val])

#   # train model
#   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#   model = CRegressor.from_pretrained(
#             '/home/glin6/causal_text/pretrained_model/model/',
#             num_labels = 2,
#             output_attentions=False,
#             output_hidden_states=False)
#   model.to(device)

#   optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
#   CE_loss = nn.CrossEntropyLoss(reduction='sum')

#   best_loss = 1e6
#   n_epochs_stop = 3
#   epochs_no_improve = 0
  
#   for epoch in range(10):
#     model.train()
#     for batch in tqdm(train_dataloader, desc='Train epoch {}'.format(epoch)):
#       model.zero_grad()
#       batch = (x.to(device) for x in batch)
#       text_id, text_len, text_mask, C = batch
#       c_logits = model(text_id, text_len, text_mask)
#       # print(c_logits.shape)
#       c_loss = CE_loss(c_logits.view(-1, 2), C.view(-1))
#       c_loss.backward()
#       optimizer.step()
    
#     # evaluate validation set
#     model.eval()
#     pbar = tqdm(val_dataloader, total=len(val_dataloader), desc='Validating')
#     val_losses = []
    
#     for batch in pbar:
#       batch = (x.cuda() for x in batch)
#       text_id, text_len, text_mask, C = batch
#       c_logits = model(text_id, text_len, text_mask)
#       val_loss = CE_loss(c_logits.view(-1, 2), C.view(-1))
#       val_losses.append(val_loss.item())
#       pbar.set_postfix({'C loss':val_loss.item()})

#     val_loss = sum(val_losses)/n_val
#     print('C validation loss:',val_loss)
    
#     # add early stop
#     if val_loss < best_loss:
#         torch.save(model, '/home/glin6/causal_text/my_model.pt')
#         # torch.save(model, '/content/drive/MyDrive/causal_text/C_simulation_model.pt')
#         best_loss = val_loss
#         epochs_no_improve = 0
#     else:
#         epochs_no_improve += 1
    
#     if epoch >= 3 and epochs_no_improve >= n_epochs_stop:
#         print('Early stopping!' )
#         print('The number of epochs is:', epoch)
#         break

#     # load the saved best model
#     model = torch.load('/home/glin6/causal_text/my_model.pt')
#   # model = torch.load('/content/drive/MyDrive/causal_text/C_simulation_model.pt')

#   # inference c_hat
#   c_prob_pred = []
#   model.eval()
#   loader = build_dataloader(text, c_true, sampler='sequential')
#   for batch in tqdm(loader, desc='Inference'):
#     batch = (x.to(device) for x in batch)
#     text_id, text_len, text_mask, C = batch
#     c_logits = model(text_id, text_len, text_mask)
#     c_pred = c_logits.argmax(dim=1)
#     c_prob_pred+=c_pred.detach().cpu().numpy().tolist()

#   return c_prob_pred


# def build_dataloader(texts, c_true, tokenizer=None, sampler='random'):

#     if tokenizer is None:
#         tokenizer = DistilBertTokenizer.from_pretrained(
#             '/home/glin6/causal_text/pretrained_model/tokenizer/', do_lower_case=True)

#     out = defaultdict(list)
#     for i, (W, C) in enumerate(zip(texts, c_true)):
#         encoded_sent = tokenizer.encode_plus(W, add_special_tokens=True,
#                                             max_length=128,
#                                             truncation=True,
#                                             pad_to_max_length=True)

#         out['text_id'].append(encoded_sent['input_ids'])
#         out['text_mask'].append(encoded_sent['attention_mask'])
#         out['text_len'].append(sum(encoded_sent['attention_mask']))
#         out['C'].append(C)

#     data = (torch.tensor(out[x]) for x in ['text_id', 'text_len', 'text_mask', 'C'])
#     data = TensorDataset(*data)
#     sampler = RandomSampler(data) if sampler == 'random' else SequentialSampler(data)
#     dataloader = DataLoader(data, sampler=sampler, batch_size=64)

#     return dataloader