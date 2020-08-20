import numpy as np
import properscoring as ps
import pandas as pd
import scipy.stats as sps

t = pd.DataFrame()

# https://stats.stackexchange.com/questions/404191/what-is-the-log-of-the-pdf-for-a-normal-distribution
def norm_lpdf(x,mu,sigma):
    return (-0.5 * ((x-mu)/sigma)**2 -np.log(sigma) - np.log(2*np.pi)/2)

def waic(model, xtrain, ytrain, samples=10000):
    import GPy
    # first we get all the predictions (mean and var of the function)
    # pay attention: predict_noiseless returns as variance only the variance of the function
    m, fvar = model.predict_noiseless(xtrain)

    # log P(y|x,theta)
    # predLogLik = model.log_predictive_density(xtrain, ytrain)

    # the following is implemented in GPstuff and yields the same results
    predLogLik = norm_lpdf(ytrain, m, np.sqrt(model.Gaussian_noise.variance + fvar))

    varLik = np.ones(len(predLogLik))
    waic = np.ones(len(predLogLik))

    # variance of the log-lik on each data point, across the sampled values of the function
    for i, x_i in enumerate(xtrain, 0):
        fsamples = np.random.normal(m[i], np.sqrt(fvar[i]), samples)
        sampled_logp = np.ones(len(fsamples))  # vector init
        yi_vec = ytrain[i] * np.ones(samples)
        sigma_noise_vec = np.sqrt(model.Gaussian_noise.variance) * np.ones(samples)
        sampled_logp = norm_lpdf(yi_vec, fsamples, sigma_noise_vec)
        # log_predictive = -np.log(mcSamples) + logsumexp(sampled_logp)
        varLik[i] = np.var(sampled_logp)

    # http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/waicwbic_e.html
    # This has to be minimized
    # The signal will be flipped later
    # The data has to be likely from the model (term1)
    # adjusted by increase likelihood caused by higher variances (term2)
    waic = -predLogLik + varLik
    return np.mean(waic)

def waic_gpflow(model, xtrain, ytrain, samples=10000):
    import gpflow
    # first we get all the predictions (mean and var of the function)
    # pay attention: predict_noiseless returns as variance only the variance of the function
    m, fvar = model.predict_f(xtrain)

    # log P(y|x,theta)
    # predLogLik = model.log_predictive_density(xtrain, ytrain)

    # the following is implemented in GPstuff and yields the same results
    noise_var = model.likelihood.variance.numpy()
    predLogLik = norm_lpdf(ytrain, m, np.sqrt(noise_var + fvar))

    varLik = np.ones(len(predLogLik))
    waic = np.ones(len(predLogLik))

    # variance of the log-lik on each data point, across the sampled values of the function
    for i, x_i in enumerate(xtrain, 0):
        fsamples = np.random.normal(m[i], np.sqrt(fvar[i]), samples)
        sampled_logp = np.ones(len(fsamples))  # vector init
        yi_vec = ytrain[i] * np.ones(samples)
        sigma_noise_vec = np.sqrt(noise_var) * np.ones(samples)
        sampled_logp = norm_lpdf(yi_vec, fsamples, sigma_noise_vec)
        # log_predictive = -np.log(mcSamples) + logsumexp(sampled_logp)
        varLik[i] = np.var(sampled_logp)

    # http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/waicwbic_e.html
    # This has to be minimized
    # The signal will be flipped later
    # The data has to be likely from the model (term1)
    # adjusted by increase likelihood caused by higher variances (term2)
    waic = -predLogLik + varLik
    return np.mean(waic)

def crpsGaussian(pred, sigma, truth):
    crps = ps.crps_gaussian(truth, mu=pred, sig=sigma)
    return crps.mean()

def loglikelihood(pred, sigma, truth):
    return sps.norm(loc=pred, scale=sigma).logpdf(truth).sum()

