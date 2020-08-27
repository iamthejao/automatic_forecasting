import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import properscoring as ps
from smKernel import train_test_gp
import sys
import time
import pandas as pd
import numpy as np
import scipy.stats as stat
import properscoring as ps

# Global variables and seeds
RESTARTS = 1
random.seed(0)

if __name__ == '__main__':

    # Simulation parameters
    # Frequency can be  'monthly' or "quarterly"
    # Batch can be 0 or 1 on the quarterly time series; between 0 and  4 on the monthly time series.
    frequency = sys.argv[1]
    batch = int(sys.argv[2])
    print("frequency: " + frequency)
    print("batch: " + str(batch))

    # Q is the number of spectral kernel mixtures + 1.
    # Hence two spectral mixtures are obtained by setting Q=3
    blackList = []
    if frequency == 'monthly':
        samplingFreq, testLen, Q, totBatches = 12, 18, 3, 5
        # In the monthly experiment we remove the ts used for fitting the hierarchical model
        blackList = pd.read_csv("data/" + str(frequency) + "/used_by_alessio.txt", header=None)[0]
    elif frequency == 'quarterly':
        samplingFreq, testLen, Q, totBatches = 4, 8, 3, 2
    elif frequency == 'weekly':
        samplingFreq, testLen, Q, totBatches = 365.25/7, 112, 3, 1
    else:
        sys.exit(f"Frequency {frequency} does not exist")

    if batch > (totBatches - 1):
        sys.exit("wrong batch number for " + frequency)

    # Reading time series
    data_dir = "data/" + frequency + "/"
    substring = 'train'
    file_list = os.listdir(data_dir)
    trainList = [i for i in file_list if substring in i]

    # Removing blacklist (if any)
    print('[On Black List]: ', len(blackList))
    print('[N ts before filter]: ', len(trainList))
    trainList = list(set(trainList) - set(blackList))
    print('[N ts after filter]: ', len(trainList))
    trainList.sort()

    ## Selecting batch to run

    # Select only the current batch of ts
    # First we split the length of the training list
    split = np.array_split(range(len(trainList)), totBatches)

    # The first index of the new shape (-1) is automatically inferred
    trainIdx = split[batch].reshape(-1, 1)  # list of ts in this batch
    trainIdx = trainIdx.astype(int)

    # We cast as array to slice, and then we cast back to list
    trainList = np.array(trainList)
    trainList = trainList[trainIdx].tolist()

    howManyTs = len(trainList)

    # Dataframe to storage the results
    results = pd.DataFrame(index=range(howManyTs), columns=['tsIdx', 'tsCategory', 'gpMae', 'etsMae', 'arimaMae', 'gp0Mae', 'prophetMae', 'thetaMae',
               'gpCRPS', 'etsCRPS', 'arimaCRPS', 'gp0CRPS', 'prophetCRPS', 'thetaCRPS',
               'gpLL', 'etsLL', 'arimaLL', 'gp0LL', 'prophetLL', 'trainingTime'])

    # Dicts to storage fcasts
    fcast = {}
    fcast_sigma = {}

    for idx in list(range(howManyTs))[:3]:

        print(str(idx) + "/" + str(howManyTs))
        print(trainList[idx][0])

        # Ts File Idx
        currentTs = trainList[idx][0].replace("train", "")

        # Train, Test and Prediction files
        testFile = "test" + currentTs
        etsFile = "etsFcast" + currentTs
        gpFile = "gpFcast" + currentTs
        arimaFile = "arimaFcast" + currentTs
        prophetFile = "prophetFcast" + currentTs
        trainFile = "train" + currentTs
        currentTs = currentTs.replace(".csv", "")
        results['tsIdx'][idx] = int(currentTs)


        # Read train, test, and forecasts
        # Header=None is necessary to load the first value
        # [1:] is necessary to remove the first element (a text) created in the import
        Y = pd.read_csv("data/" + str(frequency) + "/" + trainFile, header=None).to_numpy()[1:].astype(pd.np.float)
        Ytest = pd.read_csv("data/" + str(frequency) + "/" + testFile, header=None).to_numpy()[1:].astype(pd.np.float)
        etsFcast = pd.read_csv("data/" + str(frequency) + "/" + etsFile, header=None).to_numpy()[1:].astype(pd.np.float)
        etsSigma = (etsFcast[:, 1] - etsFcast[:, 0]) / stat.norm.ppf(0.975)
        etsFcast = etsFcast[:, 0]  # in the old code we were calling etsFcast, so I keep this variable
        etsFcast = etsFcast.reshape([len(Ytest), 1])

        arimaFcast = pd.read_csv("data/" + str(frequency) + "/" + arimaFile, header=None).to_numpy()[1:].astype(
            pd.np.float)
        arimaSigma = (arimaFcast[:, 1] - arimaFcast[:, 0]) / stat.norm.ppf(0.975)
        arimaFcast = arimaFcast[:, 0]  # in the old code we were calling etsFcast, so I keep this variable
        arimaFcast = arimaFcast.reshape([len(Ytest), 1])

        prophetFcast = pd.read_csv("data/" + str(frequency) + "/" + prophetFile, header=None).to_numpy()[1:].astype(
            pd.np.float)
        prophetSigma = (prophetFcast[:, 2] - prophetFcast[:, 1]) / stat.norm.ppf(0.975)
        prophetFcast = prophetFcast[:, 1]  # in the old code we were calling etsFcast, so I keep this variable
        prophetFcast = prophetFcast.reshape([len(Ytest), 1])

        # Metrics for competitors
        crps_ets = ps.crps_gaussian(Ytest, mu=etsFcast, sig=etsSigma)#np.zeros(len(Ytest))
        ll_ets = stat.norm.logpdf(x=Ytest, loc=etsFcast, scale=etsSigma)#np.zeros(len(Ytest))
        ll_arima = stat.norm.logpdf(x=Ytest, loc=arimaFcast, scale=arimaSigma)#np.zeros(len(Ytest))
        crps_arima = ps.crps_gaussian(Ytest, mu=arimaFcast, sig=arimaSigma)#np.zeros(len(Ytest))
        ll_prophet = stat.norm.logpdf(x=Ytest, loc=prophetFcast, scale=prophetSigma)#np.zeros(len(Ytest))
        crps_prophet = ps.crps_gaussian(Ytest, mu=prophetFcast, sig=prophetSigma)#np.zeros(len(Ytest))
        
#         for jj in range(len(Ytest)):
#             crps_ets[jj] = ps.crps_gaussian(Ytest[jj], mu=etsFcast[jj], sig=etsSigma[jj])
#             crps_arima[jj] = ps.crps_gaussian(Ytest[jj], mu=arimaFcast[jj], sig=arimaSigma[jj])
#             crps_prophet[jj] = ps.crps_gaussian(Ytest[jj], mu=prophetFcast[jj], sig=prophetSigma[jj])
#             ll_ets[jj] = stat.norm.logpdf(x=Ytest[jj], loc=etsFcast[jj], scale=etsSigma[jj])
#             ll_arima[jj] = stat.norm.logpdf(x=Ytest[jj], loc=arimaFcast[jj], scale=arimaSigma[jj])
#             ll_prophet[jj] = stat.norm.logpdf(x=Ytest[jj], loc=prophetFcast[jj], scale=prophetSigma[jj])

        results['etsMae'][idx] = np.mean(np.abs(Ytest - etsFcast))
        results['arimaMae'][idx] = np.mean(np.abs(Ytest - arimaFcast))
        results['prophetMae'][idx] = np.mean(np.abs(Ytest - prophetFcast))

        results['etsLL'][idx] = np.mean(ll_ets)
        results['arimaLL'][idx] = np.mean(ll_arima)
        results['prophetLL'][idx] = np.mean(ll_prophet)

        results['etsCRPS'][idx] = np.mean(crps_ets)
        results['arimaCRPS'][idx] = np.mean(crps_arima)
        results['prophetCRPS'][idx] = np.mean(crps_prophet)
        # ets, arima and prophet forecasts have been evaluated

        # Build X for the GP, where time increases of one after one year has passed.
        X = np.linspace(1 / samplingFreq, len(Y) / samplingFreq, len(Y))
        X = X.reshape(len(X), 1)
        endTest = X[-1] + 1 / samplingFreq * len(Ytest)
        Xtest = np.linspace(X[-1] + 1 / samplingFreq, endTest, len(Ytest))
        Xtest = Xtest.reshape(len(Xtest), 1)

        # yearly_flag is a boolean, whether the current experiment
        # is yearly or not, we need it to drop
        priors = "prior"
        yearly_flag = (frequency == "yearly")
        start = time.time()

        # train the GP with priors
        stats, gpModel, gpFcast, gpSigma = train_test_gp(X, Y, Xtest, Ytest,
                                                  RESTARTS, currentTs,
                                                  Q=Q, scoreType='penalized_lik', priors=priors,
                                                  yearly=yearly_flag)
        end = time.time()

        mae, crps, ll = stats
        model_str = 'gp'
        results[model_str + "Mae"][idx] = mae
        results[model_str + "CRPS"][idx] = crps
        results[model_str + "LL"][idx] = ll
        results['trainingTime'][idx] = end - start

        gpFcastDf = pd.DataFrame(gpFcast)
        gpFcastDf.columns = ['gpMean']
        gpFcastDf['sigma'] = gpSigma
        gpFcastDf.to_csv("data/" + str(frequency) + "/" + gpFile)

        # training the GP without priors (referred to as GP0 in the paper)
        stats0, gp0Model, _, _ = train_test_gp(X, Y, Xtest, Ytest,
                                             RESTARTS, currentTs,
                                             Q=Q, scoreType='marg_lik', priors="none",
                                             yearly=yearly_flag)

        mae, crps, ll = stats0
        model_str = 'gp0'
        results[model_str + "Mae"][idx] = mae
        results[model_str + "CRPS"][idx] = crps
        results[model_str + "LL"][idx] = ll

        # =============================================================================
        # Plotting the time series and the GP forecast
        from matplotlib import rcParams

        rc('text', usetex=True)
        sns.set(style='darkgrid', font='sans-serif', font_scale=3)
        rcParams.update({'figure.autolayout': True})
        fig, ax = plt.subplots(figsize=(12, 7))
        # cut the beginning of the training set
        actualX = np.concatenate((X[-(samplingFreq * 2):]))
        actualY = np.concatenate((Y[-(samplingFreq * 2):]))
        ax.plot(actualX, actualY, 'b', label='historical', linewidth=6)
        try:
            label = "forecast"
            ax.plot(Xtest, gpFcast, 'r--', label=label, linewidth=6)
            ax.set_ylabel("Normalized values")
            ax.set_xlabel("Time (years)")
            upper = np.array(gpFcast + gpSigma).reshape(len(Xtest), 1)
            lower = np.array(gpFcast - gpSigma).reshape(len(Xtest), 1)
            ax.fill_between(Xtest[:, 0], lower[:, 0], upper[:, 0], alpha=0.2, facecolor='red')
            ax.legend(prop={'size': 15})
            figFile = 'results/forecast/' + str(frequency) + "/" + str(currentTs) + '.pdf'
            fig.savefig(figFile)
            plt.close(fig)
        except:
            pass

        # save results to file after every time series has been parsed
        resultsFile = "results/" + str(frequency) + str(batch) + "_results.csv"
        results.to_csv(resultsFile)




