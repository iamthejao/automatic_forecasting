#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:59:30 2020
Trains the SM kernel via Bayesian optimization as in
"""

import GPy
import numpy as np
import pandas as pd
import scipy.stats as stat
import properscoring as ps


# Trains the kernel
def buildSM (X, Y, Q,pars_array, priors, yearly):

    print("received pars: " + str(pars_array))

    # Priors inherited by the hierarchial model
    logGauss_var = GPy.priors.LogGaussian (-1.6, 1.0)
    logGauss_lscal_rbf = GPy.priors.LogGaussian (1.04, 0.75)     
    logGauss_lscal_long = GPy.priors.LogGaussian (0.97, 0.73)
    logGauss_lscal_std_periodic = GPy.priors.LogGaussian(0.35, 1.44) 
    logGauss_lscal_short = GPy.priors.LogGaussian(-0.71,  0.84)
    logGauss_lscal_sparse = GPy.priors.LogGaussian(1, 3)

        
    # The first component is the linear kernel
    lin = GPy.kern.Linear(input_dim=1)
    if (priors == "prior"):
        lin.variances.set_prior(logGauss_var)
    K = lin
    offset = 2

    # The second component  is the stdPeriodic
    per0 = GPy.kern.StdPeriodic(input_dim=1, lengthscale = pars_array[0], variance=pars_array[1])
    per0.period.fix(1) # period is set to 1 year 
        
    if (priors == "prior"):
        per0.lengthscale.set_prior(logGauss_lscal_std_periodic)
        per0.variance.set_prior(logGauss_var)
    
    K = K + per0
    
    # Add the rbf kernel
    rbf = GPy.kern.RBF(input_dim=1)
    if (priors == "prior"):
        rbf.lengthscale.set_prior(logGauss_lscal_rbf)
        rbf.variance.set_prior(logGauss_var)
    K = K + rbf
    
    # Now initiliazes the  Q-1 SM components. Each component is rfb*cos, where
    # The variance of the cos is set to 1.
    for ii in range (1,Q):

        cos =  GPy.kern.Cosine(input_dim=1, lengthscale=pars_array[offset])
        cos.variance.fix(1)
        rbf =  GPy.kern.RBF(input_dim=1, lengthscale = pars_array[offset+1], variance=pars_array[offset+2]) #input dim, variance, lenghtscale

        if (ii==1):
            if (priors == "prior"):
                rbf.variance.set_prior(logGauss_var)
                rbf.lengthscale.set_prior(logGauss_lscal_long)               
                cos.lengthscale.set_prior(logGauss_lscal_long)
        elif (ii==2):
            if (priors == "prior"):
                rbf.variance.set_prior(logGauss_var)
                rbf.lengthscale.set_prior(logGauss_lscal_short)
                cos.lengthscale.set_prior(logGauss_lscal_short)
        
        K = K + cos * rbf
        offset = offset + 3

    SMmodel = GPy.models.GPRegression(X, Y, K)
    SMmodel.likelihood.variance.set_prior(logGauss_var)

    try:
        SMmodel.optimize()
        logLik =  SMmodel.log_likelihood()
        waic = compute_waic(SMmodel,X,Y)
    except:
        logLik = float('nan')
        waic = float('nan')
        
    return(SMmodel, waic)

# Train the GP
# Stores the forecast
# Performance indicator and saves pars (parameters)
def train_test_gp(X, Y, Xtest, Ytest, restarts, currentTs, Q, scoreType, priors, yearly):

    current_model, waic = trainSM_bo(Q, restarts, X, Y, currentTs,
                                     scoreType=scoreType,
                                     Xtest=Xtest,
                                     Ytest=Ytest,
                                     priors=priors,
                                     yearly=yearly)

    # Prediction
    m, v = current_model.predict(Xtest)
    s = np.sqrt(v)

    # Setting up name, gp0 means without priors
    model_str = 'gp' if priors == 'prior' else 'gp0'

    # Compute the LL and CRPS on the test set
    # Check if it is the same number of the standardized and unstandardized values
    ll = np.zeros(len(Xtest))
    crps = np.zeros(len(Xtest))

    for ii in range(len(Xtest)):
        ll[ii] = stat.norm.logpdf(x=Ytest[ii], loc=m[ii], scale=s[ii])
        crps[ii] = ps.crps_gaussian(Ytest[ii], mu=m[ii], sig=s[ii])

    ll = np.mean(ll)
    crps = np.mean(crps)
    mae = np.mean(np.abs(Ytest - m))
    stats = (mae, crps, ll)

    df = pd.DataFrame(current_model.param_array, current_model.parameter_names())
    df.to_csv('results/pars/' + currentTs + '_' + model_str + '.csv')
    return stats, current_model, m, s

def trainSM(pars,X,Y,Q):   
    pars_array = np.array(list(pars.values()))#from dict to nparray
    SMmodel, marglik = buildSM (X, Y, Q, pars_array)
    return marglik 

def norm_lpdf(x,mu,sigma):
    return ( -0.5 * ((x-mu)/sigma)**2 -np.log(sigma) - np.log(2*np.pi)/2 )

# Computes the WAIC of a GP model
def compute_waic(gp,x_tr,y_tr): 
    #first we get all the predictions (mean and var of the function)
    #pay attention: predict_noiseless returns as variance only the variance of the function
    m,fvar = gp.predict_noiseless(x_tr)

    
    predLogLik = gp.log_predictive_density(x_tr, y_tr)
    #the following is implemented in GPstuff and yields the same results
    #predLogLik = norm_lpdf(y_tr, m, np.sqrt(gp.Gaussian_noise.variance+ fvar))
        
    varLik = np.ones(len(predLogLik)) 
    waic = np.ones(len(predLogLik))
    mcSamples = 5000

    #variance of the log-lik on each data point, across the sampled values of the function
    for i,x_i in enumerate(x_tr,0):
        fsamples = np.random.normal(m[i], np.sqrt(fvar[i]), mcSamples)
        sampled_logp = np.ones(len(fsamples))#vector init
        yi_vec = y_tr[i] * np.ones(mcSamples)
        sigma_noise_vec = np.sqrt(gp.Gaussian_noise.variance) * np.ones(mcSamples)
        sampled_logp = norm_lpdf(yi_vec, fsamples, sigma_noise_vec)
        #log_predictive = -np.log(mcSamples) + logsumexp(sampled_logp) 
        varLik[i] = np.var(sampled_logp)

    #waic is a vector, whose optimal value has to be maximized    
    waic = predLogLik - varLik
    return np.mean(waic)


# In general it is possible to train the GP different times, and to choose
# The supposedly best parameters according to a model selection criterion (marginal lik, waic, etc.)
# Yet in the experiments of the paper we use a single restart and thus the model selection criterion is irrelevant.
    
def trainSM_score(pars,X,Y,Q,ts_id,number,scoreType, Xtest, Ytest, priors, yearly):   

    score = 0
    pars = np.array(list(pars.values()))#from dict to nparray
    SMmodel, waic = buildSM (X, Y, Q, pars, priors,  yearly)
    
    #in this case we try the model on last 18 months of the training data
    if (scoreType=='validation'):
        print("using validation:")
        global_kern = SMmodel.kern
        #keep the parameter of the global model, but condition leaving out the last 18
        #observations
        local_model = GPy.models.GPRegression(X[:-18], Y[:-18], global_kern, noise_var=SMmodel.likelihood.variance)
        m,v = local_model.predict(X[-17:])
        
        #loglik
        z=np.zeros(17)
        ll=np.zeros(17)
        for ii in range(17):
            z[ii]=  (m[ii] - Y[ii])/np.sqrt(v[ii])
            ll[ii] = stat.norm.logpdf(z[ii])
        score = np.sum(ll)        
    
    elif (scoreType=='loo'):
        loo = SMmodel.inference_method.LOO(SMmodel.kern, X,Y,SMmodel.likelihood, SMmodel.posterior)
        score = -1* np.mean(np.abs(loo))
   
    #optimizing the score on the test for debugging purpose
    elif (scoreType=='test'):
        m,v = SMmodel.predict(Xtest)
        score = -1 * np.mean ( np.abs (Ytest  - m) )
    
    #marginal log-lik has to be maximized, we do not change the sign
    elif (scoreType=='marg_lik'):
        score = SMmodel.log_likelihood()
        
    elif (scoreType=='penalized_lik'):
        score = SMmodel.log_likelihood() + SMmodel.kern.log_prior()
        
    elif (scoreType=='waic'):
        #waic has to be maximized, hence we do not change the sign
        score = compute_waic(SMmodel, X, Y)
        print("waic"+str(score))
    return score


# Early code which trains the kernel with Bayesian optimization.
# Not necessary in the time series paper where we use a single restart.
# Requires the number of components and restart (3 and 15 could be sensible default)
# By setting m-loo=True you make two years of out-of-sample predictions
def trainSM_bo(Q,restarts,X,Y,ts_id,scoreType='validation',Xtest=0, Ytest=0, priors=True, yearly=False):

    import optuna

    #reduce output to screen
    #optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):

        #We avoid too short lenghtscales on the first components which model periodicities
        params = {
                'std_per_lscale' : trial.suggest_uniform('std_per_lscale', 0.5, 2),
                'std_per_var' : trial.suggest_uniform('std_per_var', 0.1, 1)
                  }
        for q in range(1,Q):
            params.update([ ('rbf_lscale'+str(q), trial.suggest_uniform('rbf_lscale'+str(q), 0.5, 2)),
                            ('cos_lscale'+str(q), trial.suggest_uniform('cos_lscale'+str(q), 0.5, 2)),
                            ('rbf_var'+str(q), trial.suggest_uniform('rbf_var'+str(q), 0.1, 1))
                            ])#recall that 1 year -> cos_l = 1/12 = 0.15
        return trainSM_score(params,X,Y,Q,ts_id,trial.number,scoreType,Xtest,Ytest,priors,yearly)
    
    # Make the sampler behave in a deterministic way
    # sampler = optuna.samplers.TPESampler(seed=0) 
    
    # Make the sampler behave differently every time
    import datetime
    micros = datetime.datetime.now().microsecond
    sampler = optuna.samplers.TPESampler(seed=micros) 
    print("seed: " + str(micros) )
    study = optuna.create_study(direction='maximize',sampler=sampler)
    study.optimize(objective, n_trials=restarts)
    pars_array = np.array(list(study.best_params.values()))
    model, waic  = buildSM(X,Y,Q,pars_array, priors, yearly)
    return model, waic

