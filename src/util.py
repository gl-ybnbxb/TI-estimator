import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
from scipy.stats import norm


def ATE_unadjusted(T, Y):
  T0 = np.mean(Y[T==0])
  T1 = np.mean(Y[T==1])
  return T1 - T0


def ATE_adjusted(C, T, Y):
  n = len(Y)
  pi_0 = sum(C==0)/n
  pi_1 = 1-pi_0
  idx01 = (C==0)*(T==1)
  idx00 = (C==0)*(T==0)
  idx11 = (C==1)*(T==1)
  idx10 = (C==1)*(T==0)

  C0_ATE = np.mean(Y[idx01]) - np.mean(Y[idx00])
  C1_ATE = np.mean(Y[idx11]) - np.mean(Y[idx10])
  # print(C0_ATE)
  # print(C1_ATE)
  
  return C0_ATE*pi_0+C1_ATE*pi_1
  

def oracle_AIPTW(C, T, Y, true_propensity):
  n = len(Y)
  C, T, Y = np.array(C), np.array(T), np.array(Y)
  g = np.array([true_propensity[int(v)] for v in C])
  phi= Y/g*T - Y/(1-g)*(1-T)
  tau_hat = phi.mean()
  std_hat = np.std(phi)/math.sqrt(n)

  ci_lower = tau_hat - 1.96*std_hat
  ci_upper = tau_hat + 1.96*std_hat

  return [tau_hat, std_hat, ci_lower, ci_upper]


def ate_aiptw(Q0, Q1, A, Y, g=None, weight=False, error_bound=0.01):
  """
  # Double ML/BD estimator for the ATE
  if g is None, then return BD estimator
  """
  n = len(Y) # number of observations

  if g is None:
    phi = Q1 - Q0
    w = np.ones(n)
  else:
    phi = Q1 - Q0 + A*(Y-Q1)/g - (1-A)*(Y-Q0)/(1-g)
    if weight:
      w = g*(1-g)
    else:
      w = np.ones(n)

  tau_hat = sum(phi*w)/sum(w)
  w_cst = math.sqrt(sum((w)**2))
  std_hat = w_cst*np.std(phi)/sum(w)

  ci_lower = tau_hat - norm.ppf(1-error_bound/2)*std_hat
  ci_upper = tau_hat + norm.ppf(1-error_bound/2)*std_hat

  return [tau_hat, std_hat, ci_lower, ci_upper]


def ate_iptw(A, Y, g):
  """
  # IPTW for the ATE;
  g is the propensity score
  """
  n = len(Y) # number of observations

  phi = A*Y/g - (1-A)*Y/(1-g)
  tau_hat = phi.mean()
  
  std_hat = np.std(phi) / math.sqrt(n)

  ci_lower = tau_hat - 1.96*std_hat
  ci_upper = tau_hat + 1.96*std_hat

  return [tau_hat, std_hat, ci_lower, ci_upper]


def att_q(Q0, Q1, g, A, Y, prob_t=None, error_bound=0.01):
  '''
  Double ML estimator for the ATT
  '''
  if prob_t is None:
      prob_t = A.mean() # estimate marginal probability of treatment

  tau_hat = ((Q1-Q0)*A).mean()/prob_t
  scores = ((Q1-Q0)*A - tau_hat*A) / prob_t
  n = Y.shape[0] # number of observations
  std_hat = np.std(scores) / math.sqrt(n)
  ci_lower = tau_hat - norm.ppf(1-error_bound/2)*std_hat
  ci_upper = tau_hat + norm.ppf(1-error_bound/2)*std_hat

  return [tau_hat, std_hat, ci_lower, ci_upper]


def att_aiptw(Q0, Q1, g, A, Y, prob_t=None, error_bound=0.01):
  '''
  Double ML estimator for the ATT
  '''
  if prob_t is None:
      prob_t = A.mean() # estimate marginal probability of treatment

  tau_hat = (A*(Y-Q0) - (1-A)*(g/(1-g))*(Y-Q0)).mean()/ prob_t
  scores = (A*(Y-Q0) - (1-A)*(g/(1-g))*(Y-Q0) - tau_hat*A) / prob_t
  n = Y.shape[0] # number of observations
  std_hat = np.std(scores) / math.sqrt(n)
  ci_lower = tau_hat - norm.ppf(1-error_bound/2)*std_hat
  ci_upper = tau_hat + norm.ppf(1-error_bound/2)*std_hat

  return [tau_hat, std_hat, ci_lower, ci_upper]


def overlap_weight_est(Q0, Q1, g, A, Y, original_g=False):
  n = len(Y)
    
  trans_g = np.array([logit(v) for v in g])
  X = np.concatenate((Q0.reshape(-1,1),Q1.reshape(-1,1),trans_g.reshape(-1,1)),axis=1)
    
  # logistic regression
  if original_g == False:
    logistic = LogisticRegression(random_state=0).fit(X,A)
    g_tilde = logistic.predict_proba(X)[:,1]
    phi_1 = Y*A*(1-g_tilde)
    phi_0 = Y*(1-A)*g_tilde
    tau_hat = sum(phi_1)/sum(A*(1-g_tilde)) - sum(phi_0)/sum((1-A)*g_tilde)
    std_hat_1 = np.std(phi_1/(A*(1-g_tilde)).mean())/math.sqrt(n)
    std_hat_0 = np.std(phi_0/((1-A)*g_tilde).mean())/math.sqrt(n)
    std_hat = math.sqrt((std_hat_1**2+std_hat_0**2))
  else:
    phi_1 = Y*A*(1-g)
    phi_0 = Y*(1-A)*g
    tau_hat = sum(phi_1)/sum(A*(1-g)) - sum(phi_0)/sum((1-A)*g)
    std_hat_1 = np.std(phi_1/(A*(1-g)).mean())/math.sqrt(n)
    std_hat_0 = np.std(phi_0/((1-A)*g).mean())/math.sqrt(n)
    std_hat = math.sqrt((std_hat_1**2+std_hat_0**2))

  ci_lower = tau_hat - 1.96*std_hat
  ci_upper = tau_hat + 1.96*std_hat

  return [tau_hat, std_hat, ci_lower, ci_upper]



''' Bayesian Medain of the Means'''
def sample_Y(theta: np.array, alpha: float):
  ''' Samples p from Dir(alpha, ..., alpha) and, given theta, returns the
    weighted sum of theta by p.
  '''
  n = len(theta)
  p = np.random.dirichlet(alpha*np.ones(n))
  return np.dot(theta, p)


def bayesian_median_of_means(theta_hat: np.array, alpha: float, J=1000):
  ''' Compute Bayesian median of means using J replications from a
  Dirichlet(alpha, ..., alpha) to weight theta_hat.'''
  if alpha==0:
    Y_vec = theta_hat
  else:
    Y_vec = np.fromiter([sample_Y(theta_hat, alpha) for j in range(J)], dtype=float)
  return np.median(Y_vec)


def MSE(predictions: np.array, truth: np.array):
  ''' Calculate the mean squared error between predictions and
  ground truth.
  '''
  return np.mean((predictions - truth)**2)

def ci_bmm(theta_hat: np.array, alpha: float, error_bound = 0.05, J=1000, B=1000):
  '''bootstrap ci for the bmm'''
  theta_bmm = bayesian_median_of_means(theta_hat, alpha, J=J)
  n = len(theta_hat)
  if alpha!=0:
    p_mat = np.random.dirichlet(alpha*np.ones(n),size=J)
  theta_vec = [theta_bmm]

  for b in range(B):
    theta_b = np.random.choice(theta_hat, size=n)
    if alpha==0:
      Y_vec = theta_b
    else:
      Y_vec = p_mat@theta_b
    theta_vec += [np.median(Y_vec)]

  theta_vec = np.array(theta_vec)
  ci_lower, ci_upper = np.quantile(theta_vec, q=error_bound/2), np.quantile(theta_vec, q=1-error_bound/2)
  return theta_bmm, np.nan, ci_lower, ci_upper


def bmm(Q0, Q1, A, Y, g, alpha=1, type='AIPTW'):
  if type == 'AIPTW':
    phi = Q1 - Q0 + A*(Y-Q1)/g - (1-A)*(Y-Q0)/(1-g)
  elif type == 'IPTW':
    phi = A*Y/g - (1-A)*Y/(1-g)
  elif type == 'Q only':
    phi = Q1-Q0

  return ci_bmm(phi, alpha=alpha, error_bound = 0.01, J=1000, B=1000) 


''' Adaptive IPW '''
def aipw(I, p, Y):
  n = len(Y)
  phi = Y/p*I
  
  S_hat = sum(phi)
  n_hat = sum(I/p)

  mu_HT = S_hat/n
  mu_Hajek = S_hat/n_hat
  
  T_seq = (1-p)/p*Y*I/p
  T_hat = T_seq.mean()
  pi_seq = (1-p)/p*I/p
  pi_hat = pi_seq.mean()
  
  # AIPW
  mu_lambda = S_hat/n+T_hat/pi_hat*(1-n_hat/n)
  var_lambda = ((1-p)/p*((Y-T_hat/pi_hat)**2)).mean()
  std_lambda = math.sqrt(var_lambda)
  
  return mu_lambda, std_lambda

def ate_aipw(A, Y, g, error_bound=0.05):
  mu1 = aipw(A, g, Y)
  mu0 = aipw(1-A, 1-g, Y)

  tau_hat = mu1[0]-mu0[0]
  std_hat = math.sqrt(mu1[1]**2+mu0[1]**2)/math.sqrt(2)

  ci_lower = tau_hat - norm.ppf(1-error_bound/2)*std_hat
  ci_upper = tau_hat + norm.ppf(1-error_bound/2)*std_hat

  return tau_hat, std_hat, ci_lower, ci_upper

    
 
def get_estimands(gs, Q0s, Q1s, As, Ys, Cs=None):
    """Get estimands based on propensity scores, prognostic scores, treatments and outcomes"""

    estimands = []

    estimands.append(('unadj_T', [ATE_unadjusted(As, Ys)]+[np.nan]*3))
    estimands.append(('adj_T', [ATE_adjusted(Cs, As, Ys)]+[np.nan]*3))
    idx = (0.10<gs)*(gs<0.90)

    # Q only ATE
    ATE_Q = ate_aiptw(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=None, weight=False)
    estimands.append(('ate_Q', ATE_Q))

    # ATE AIPTW
    estimands.append(('ate_AIPTW', ate_aiptw(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=gs, weight=False)))
    estimands.append(('trimmed ate_AIPTW', ate_aiptw(Q0=Q0s[idx], Q1=Q1s[idx], A=As[idx], Y=Ys[idx], g=gs[idx], weight=False)))

    # weighted ATE AIPTW
    estimands.append(('weighted_ate_AIPTW', ate_aiptw(Q0s, Q1s, As, Ys, gs, weight=True)))
    estimands.append(('trimmed weighted_ate_AIPTW', ate_aiptw(Q0=Q0s[idx], Q1=Q1s[idx], A=As[idx], Y=Ys[idx], g=gs[idx], weight=True)))
    
    # AIPTW with oracle g
    estimands.append(('oracle_ate_AIPTW', oracle_AIPTW(Cs, As, Ys, [0.8, 0.6])))
    
    # BMM
    bmm_ate = bmm(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=gs)
    estimands.append(('ate_BMM', bmm_ate))
    estimands.append(('trimmed ate_BMM', bmm(Q0=Q0s[idx], Q1=Q1s[idx], A=As[idx], Y=Ys[idx], g=gs[idx])))
    
    # ATE IPTW
    estimands.append(('ate_IPTW', ate_iptw(As, Ys, gs)))
    estimands.append(('trimmed ate_IPTW', ate_iptw(As[idx], Ys[idx], gs[idx])))
    
    # ATT AIPTW
    estimands.append(('att_AIPTW', att_aiptw(Q0=Q0s, Q1=Q1s, A=As, Y=Ys, g=gs)))
    estimands.append(('trimmed att_AIPTW', att_aiptw(Q0=Q0s[idx], Q1=Q1s[idx], A=As[idx], Y=Ys[idx], g=gs[idx])))

    # ATT AIPTW
    estimands.append(('att_AIPW', att_aipw(A=As, Y=Ys, g=gs)))
    estimands.append(('trimmed att_AIPW', att_aipw(A=As[idx], Y=Ys[idx], g=gs[idx])))

    # overlap weighted with adjusted g
    estimands.append(('overlap_with_adjusted_g', overlap_weight_est(Q0s, Q1s, gs, As, Ys)))
    estimands.append(('trimmed overlap_with_ori_g', overlap_weight_est(Q0s[idx], Q1s[idx], gs[idx], As[idx], Ys[idx], original_g=True)))

    # overlap weighted with gpr g
    estimands.append(('overlap_with_ori_g', overlap_weight_est(Q0s, Q1s, gs, As, Ys, original_g=True)))
    estimands.append(('trimmed overlap_with_ori_g', overlap_weight_est(Q0s[idx], Q1s[idx], gs[idx], As[idx], Ys[idx], original_g=True)))

    estimands = pd.DataFrame(data=dict(estimands))

    return estimands