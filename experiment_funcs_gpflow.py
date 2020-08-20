from sklearn.preprocessing import StandardScaler
import experiment_config_gpflow as config
import pandas as pd
import dataclasses
from pathlib import Path
import numpy as np
import analysis_gpflow as analysis
import dill as pickle
from decorators import lazy
import model_gpflow as model
from gpflow.models import GPR
from filelock import FileLock

import bz2
import _pickle as cPickle
import copy

# Pickle a file and then compress it into a file with extension
def compressed_pickle(title, data):
    with bz2.BZ2File(str(title) + '.pbz2', 'w') as f:
        cPickle.dump(data, f)

# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

def get_idx(file):
    return int(file.replace('train', '').replace('.csv', ''))

# Read data
@lazy
def read_experiment_data(folder, location, file):
    idx = get_idx(file)
    train = pd.read_csv(location.joinpath(file))[['x']].values
    test = pd.read_csv(location.joinpath('test{0}.csv'.format(idx)))[['x']].values

    fcasts = {}
    if location.joinpath('fcast{0}.csv'.format(idx)).exists():
        other_fcasts = pd.read_csv(location.joinpath('fcast{0}.csv'.format(idx)))

        until_TEST = list(other_fcasts.columns).index('TEST')
        other_models = other_fcasts.columns[:until_TEST]

        fcasts = {key: {} for key in other_models}
        for key in other_models:
            fcasts[key]['fcast'] = np.float32(other_fcasts[key].values)
            fcasts[key]['95'] = np.float32(other_fcasts['{0}_UPPER_95%'.format(key)].values)

    # Size and Normalization variables
    sizeTrain = len(train)

    y = np.vstack((train, test)).ravel()
    data = np.array(list(enumerate(y)), dtype=np.float32)

    # Starting from 1
    data[:, 0] += 1

    # Transforming x axis to years (instead of months or quarters)
    # so the fixed period in the experiment makes sense
    # we know that the data is 1y periodic
    period = config.DATA_FOLDER_PERIOD[folder]
    data[:, 0] /= period

    # Separating the data
    xtrain, xtest = data[:sizeTrain, 0], data[sizeTrain:, 0]
    ytrain, ytest = data[:sizeTrain, 1], data[sizeTrain:, 1]

    # Adding a dimension (column vector)
    xtrain = xtrain[None].T
    xtest = xtest[None].T
    ytrain = ytrain[None].T
    ytest = ytest[None].T

    # Normalizing data
    scaler = StandardScaler()

    ytrain = scaler.fit_transform(ytrain)
    ytest = scaler.transform(ytest)

    # Other predictions data, rescaling:
    for key in other_models:
        for m in fcasts[key].keys():
            fcasts[key][m] = scaler.transform(fcasts[key][m][None].T)

    # Making keys lower-case for version compatibility
    for key in fcasts.keys():
        fcasts[key.lower()] = fcasts.pop(key)

    #etsFcast = scaler.transform(etsFcast[None].T)
    #ets95Up = scaler.transform(ets95Up[None].T)
    #arimaFcast = scaler.transform(arimaFcast[None].T)
    #arima95Up = scaler.transform(arima95Up[None].T)
    #fcasts = {'ets' : {'fcast': etsFcast, '95': ets95Up},
    #          'arima' : {'fcast': arimaFcast, '95': arima95Up}}

    data_dict = {'xtrain': xtrain, 'ytrain': ytrain,
                 'xtest': xtest, 'ytest': ytest, 'scaler': scaler,
                 'fcasts': fcasts, 'idx':idx,
                 'folder': folder, 'location': location, 'file': file,
                 'period': period}

    return data_dict

@dataclasses.dataclass(frozen=False, init=True)
class ExperimentInput:

    name: str
    data: dict
    kernel: object
    likelihood: str
    model: str
    priors: dict
    score: str
    inducing: tuple
    restarts: int

def train_model(input: ExperimentInput):

    # Changing data inside class
    # Evaluate the lazy function
    if not(type(input.data) is dict):
        input.data = input.data()

    # Kernel
    kernel = input.kernel

    # Adding priors to kernel
    kernel_w_prior = model.set_kernel_priors(input.priors, kernel)

    # Emptying cache in case it is not empty
    model.CACHE_MODELS = {}

    # I dont care about output, I will get the cached models
    _, best_score = model.bayes_opt_training(input,
                                             kernel_w_prior,
                                             n_jobs=1,
                                             restarts=input.restarts,
                                             seed=config.SEED)

    keys = [key for key in model.CACHE_MODELS.keys() if key[1] == input.name]

    # Sorting by trial
    keys = sorted(keys, key=lambda tpl: tpl[-1])

    ms = []
    for CHECK in config.CHECKPOINTS:
        slice_keys = keys[:CHECK]
        models = [model.CACHE_MODELS[key]['model'] for key in slice_keys]
        scores = [model.CACHE_MODELS[key]['value'] for key in slice_keys]
        best_cached_score = np.nanmax(scores)
        best_idx = scores.index(best_cached_score)
        best_key = slice_keys[best_idx]
        reg = models[best_idx]
        goodness = best_cached_score

        print('[CHECKPOINT]: ', CHECK)
        print('[BEST TRIAL]:', best_key[-1])
        print('[SCORE]', goodness)

        in_ = copy.deepcopy(input)
        in_.name = in_.name + " " + str(CHECK)
        failed = np.isnan(goodness)
        ms.append((reg, goodness, in_, failed))

    model.CACHE_MODELS = {}

    return ms


def get_model_parameters(reg): return {p.name: p for p in reg.parameters}


def save_model_results(out_train):

    (reg, goodness, input, failed) = out_train

    # Path for saving
    result_path = Path(config.RESULT_ROOT).joinpath(input.data['folder']).joinpath(str(input.data['idx']))
    result_path.mkdir(parents=True, exist_ok=True)

    # Saving data
    data_path = result_path.joinpath("data_dict.dat")
    if data_path.exists() is False:
        compressed_pickle(data_path, input.data)

    # Analysing the results
    analyst = analysis.Analyst(reg, input.data, input.name, input.score, input.data['period'])

    # Making component plot, only works for GPR for now
    #if type(reg) is GPR:
    #    analyst.make_component_plot(result_path.joinpath("{0}_components.png".format(input.name)))

    # Making error table
    save_fig_path = None
    if config.SAVE_PLOTS:
        save_fig_path = result_path.joinpath("{0}.png".format(input.name))

    # Making error table
    info, fcasts = analyst.check_forecast(save_file=save_fig_path,
                                  comparisons=input.data['fcasts'], closeFig=True)

    compressed_pickle(result_path.joinpath("{0}.predictions".format(input.name)), fcasts)
    #pickle.dump(fcasts, open(result_path.joinpath("{0}.predictions".format(input.name)), 'wb'))

    #info[0]['failed_training'] = fail_train
    info[0]['score'] = goodness
    info[0]['score_name'] = input.score
    for error in info:
        error['folder'] = input.data['folder']
        error['idx'] = input.data['idx']

    df = pd.DataFrame(info)
    save_df = result_path.joinpath("stats{0}.csv".format(input.data['idx']))

    lock_path = str(save_df) + ".lock"
    lock = FileLock(lock_path)
    with lock:
        if save_df.exists():
            df_exist = pd.read_csv(save_df).append(df)
            cols = list(filter(lambda name: 'Unnamed' not in name, df_exist.columns))
            df_exist[cols].to_csv(save_df, header=True, mode='w')
        else:
            df.to_csv(save_df, header=True, mode='w')
    return df


def train_and_save(input: ExperimentInput):

    m = train_model(input)
    for models in m:
        save_model_results(models)
