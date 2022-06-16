# Treatment Ignorant Effect Estimation
Pytorch implementation of "Causal Estimation for Text Data with Apparent Overlap Violations".

# Environment Setup
```
pip install -r requirements.txt
```
Then install the stable version (1.11.0) of PyTorch following the [official guideline](https://pytorch.org/get-started/locally/).

# Description
* Q-Net:
  * Inputs of the system include
    * **Text** data $X$
    * A binary **treatment** variable $A$
    * A continuous **outcome** variable $Y$
    * (Optional) A binary confound $C$
  * Outputs
    * Estimated conditional outcome $Q_0=E(Y|A=0,X)$, $Q_1=E(Y|A=1,X)$
  * Other related functions
    * ***k_fold_fit_and_predict***: train Q-Net in K-fold fashion
    * ***get_agg_q***: train Q-Net with different seeds and get aggregated (average) conditional outcomes
* Propensity estimation
  * Inputs
    * A binary **treatment** variable
    * Conditional outcomes: $Q_0=E(Y|A=0,X)$, $Q_1=E(Y|A=1,X)$
    * Choice of the nonparametric model
  * Output
    * Estimated propensity scors $g$
* TI estimator
  * Inputs
    * A binary **treatment** variable $A$
    * A continuous **outcome** variable $Y$
    * Conditional outcomes: $Q_0=E(Y|A=0,X)$, $Q_1=E(Y|A=1,X)$
    * Estimated propensity scors $g$
    * Error bound for the confidence interval
  * Output
    * The TI estimator together with its uncertainty quantification and confidence interval
  * Other related functions: 
    * ***get_estimands***: get a list of causal estimators


# Usage

**Initialize** the Q-Net model wrapper and **train** the model.
```
mod = QNet(batch_size = 64, # batch size for training
           a_weight = 0.1,  # loss weight for A ~ text
           y_weight = 0.1,  # loss weight for Y ~ A + text
           mlm_weight=1.0,  # loss weight for DistlBert
           modeldir='model/train') # directory for saving the best model
           
mod.train(df['text'],  # texts in training data
          df['T'],     # treatments in training data
          df['C'],     # confounds in training data, binary
          df['Y'],     # outcomes in training data
          epochs=20,   # the maximum number of training epochs
          learning_rate = 2e-5)  # learning rate for the training
```

Then, obtain the **conditional oucomes** for the test data.
```
Q0, Q1, A, Y, _ = mod.get_Q(test_df['text'], test_df['T'], test_df['C'], test_df['Y'])
```

Compute the estimated **propensity scores**.

```
g = get_propensities(A, Q0, Q1, 
                     model_type='GaussianProcessRegression', # choose the nonparametric model
                     kernel=None,    # kernel function for GPR
                     random_state=0) # random seed for GPR 
```

Get the **TI estimator** and its confidence interval.
```
get_TI_estimator(g, Q0, Q1, A, Y, 
                  error=0.05)  # error bound for confidence interval
```

