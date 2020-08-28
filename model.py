import GPy
import optuna
import numpy as np
from scores import waic
from decorators import timeit
from sklearn.preprocessing import Normalizer
import numpy.random as npr
import pathos

OPTIMIZER = 'lbfgsb'
CACHE_MODELS = {}

class ExtraKernels:

    @staticmethod
    def SpectralMixture(Q, input_dim=1, enforce_periodic_comp=0, enforced_period=1,
                        name='SpectralMixture'):
        """
        rbf.variance plays the role of the weights (wi)
        rbf.lengthscale plays the role of SM variance (vq)
        cos.lengthscale plays the role of SM means (uq)
        """
        for i in range(Q):
            cos = GPy.kern.Cosine(input_dim=input_dim)
            cos.variance.fix(1)
            if (i < enforce_periodic_comp):
                cos.lengthscale.fix(enforced_period)

            rbf = GPy.kern.RBF(input_dim=input_dim)
            # Rbf's sigma will take the weight
            # Cosine's lengthscale will take the mean
            sm = cos * rbf
            sm.name = 'MixtureComponent{0}'.format(i)
            if i == 0:
                SM = sm
            else:
                SM += sm
        SM.name = name
        if Q == 1:
            SM.name = 'MixtureComponent0'

        return SM

    @staticmethod
    def Quasiperiodic(input_dim=1, name='Quasiperiodic', period=None):
        per = GPy.kern.StdPeriodic(input_dim=input_dim)
        if period is not None:
            per.period.fix(period)
        rbf = GPy.kern.RBF(input_dim=input_dim)
        rbf.variance.fix(1)
        qper = per * rbf
        qper.name = name
        return qper

def set_kernel_parameters(param_dict, kernel):
    tmp = kernel.copy()
    for param in param_dict.keys():
        tmp[param] = param_dict[param]
    return tmp

def set_kernel_priors(priors, kernel):

    tmp = kernel.copy()
    parameters = kernel.parameter_names_flat()

    # Checking if a mixture is present
    # so the priors are set accordingly
    mix = 'MixtureComponent'
    mixcomp = [int(name[name.index(mix) + len(mix)]) for name in list(filter(lambda name: mix in name, parameters))]
    one_comp = False
    if len(mixcomp) > 0:
        n_mix = max(mixcomp) + 1
        one_comp = (n_mix == 1)

    for param in parameters:
        param = ".".join(param.split(".")[1:])
        type_ = param.split('.')[-1]
        name = param.split('.')[0]
        for kernName in priors['kernels'].keys():
            if kernName in name: # Found a match
                # Treating different priors for each mixture component
                if 'Mixture' in name:
                    # If only one component, we use a broader default prior
                    if one_comp:
                        var_priors = priors['kernels'][kernName]['default']
                    # Otherwise use indexed prior.
                    else:
                        # Last character in name is the idx
                        var_priors = priors['kernels'][kernName][int(name[-1])]
                # Other kernels use standard
                else:
                    var_priors = priors['kernels'][kernName]
                if type_ in var_priors.keys():
                    tmp[param].set_prior(var_priors[type_])
                else:
                    pass
    return tmp

def suggest_kernel_initialization(trial: optuna.Trial, kernel):
    parameters = kernel.parameter_names_flat()
    for parameter in parameters:
        parameter = ".".join(parameter.split(".")[1:])
        if 'Mixture' in parameter:
            suggestion = trial.suggest_uniform(parameter, 0.1, 1.0)
        else:
            type_ = parameter.split('.')[-1]
            if 'variance' in type_:
                suggestion = trial.suggest_uniform(parameter, 0.1, 1.0)
            elif 'lengthscale' in type_:
                suggestion = trial.suggest_uniform(parameter, 0.5, 2.0)
            elif 'period' in type_:
                suggestion = trial.suggest_uniform(parameter, 0.0, 1.5)
            else:
                suggestion = trial.suggest_uniform(parameter, 0.1, 1.0)
        kernel[parameter] = suggestion
    return kernel

# The optimizer maximizes scores
def score(model, scoreType, Xvalid=None, Yvalid=None):
    X, Y = model.X, model.Y
    if scoreType == 'll':
        # Increase likelihood
        scoreValue = model.log_likelihood() + model.log_prior()
    elif scoreType == 'loo':
        loo = model.inference_method.LOO(model.kern, X, Y, model.likelihood, model.posterior)
        # He returns - the neg_log_marginal_LOO, therefore the log_marginal_LOO which we want to maximize
        scoreValue = np.mean(loo)
    elif scoreType == 'waic':
        # waic = -predLogLik + varLik
        # We want to maximize predLogLik, then multiply by -1
        scoreValue = -1 * waic(model, X, Y)
    else:
        raise NotImplemented('Score {0} not implemented'.format(score))
    return scoreValue

def build_model(input, kernel):

    if input.likelihood == 'Gaussian':

        reg = GPy.models.GPRegression(input.data['xtrain'],
                                      input.data['ytrain'],
                                      kernel=kernel)

    elif input.likelihood == 'StudentT':

        t_distribution = GPy.likelihoods.StudentT(deg_free=5)
        laplace_inf = GPy.inference.latent_function_inference.Laplace()

        reg = GPy.core.GP(input.data['xtrain'],
                          input.data['ytrain'],
                          kernel=kernel,
                          likelihood=t_distribution,
                          inference_method=laplace_inf)
        reg.Student_T.deg_free.set_prior(GPy.priors.Gamma(2.0, 0.1))#.fix(5)  # degrees are fixed
    else:
        raise Exception("Wrong likelihood name: {0}".format(input.likelihood))

    # Inserting prior on likelihood variance
    if 'likelihood' in input.priors.keys():
        if input.likelihood is 'Gaussian':
            reg.likelihood.variance.set_prior(input.priors['likelihood'])
        elif input.likelihood is 'StudentT':
            reg.likelihood.t_scale2.set_prior(input.priors['likelihood'])

    return reg

@timeit
def bayesian_opt_training(input, kernel, restarts=20, seed=0, n_jobs=1,
                          timeout=None, **kwargs):

    def objective(trial: optuna.Trial):

        # Randomize and copy kernel to make sure it is different
        kernel.randomize()
        kernel_cpy = kernel.copy()

        # Suggest init
        init_kernel = suggest_kernel_initialization(trial, kernel_cpy)

        # Make cache key to get save models
        hash_key = hash(tuple(np.round(init_kernel.param_array, 5).tolist()))
        key = (hash_key, input.name, trial.number)

        reg = build_model(input, init_kernel)
        try:
            reg.optimize(optimizer=OPTIMIZER, **kwargs)
            goodness = score(reg, input.score, input.data['xtrain'], input.data['ytrain'])
        except:
            goodness = float('nan')

        CACHE_MODELS[key] = {'model': reg, 'value': goodness}

        return goodness

    # Make the sampler behave in a deterministic way and set numpy seed
    np.random.seed(seed)
    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=restarts, n_jobs=n_jobs,
                   timeout=timeout)



    return study

# @timeit
# def suggest_best_start_sparse(regBase, restarts=5, seed=0, n_jobs=1,
#                        timeout=None, **kwargs):
#
#     X, Y, Z = regBase.X, regBase.Y, np.array(regBase.Z.tolist())
#     ll = list(regBase.likelihood.variance.priors.items())[0][0]
#     def objective(trial: optuna.Trial):
#
#         kernel = regBase.kern.copy()
#         kernel.randomize()
#         init_kernel = suggest_kernel_initialization(trial, kernel)
#         hash_key = hash(tuple(np.round(init_kernel.param_array, 3).tolist()))
#
#         np.random.seed(seed)
#         reg = GPy.models.SparseGPRegression(X, Y, kernel=init_kernel, Z=Z)
#         reg.likelihood.set_prior(ll)
#         reg.inducing_inputs.constrain_fixed()
#         try:
#             reg.optimize(optimizer=OPTIMIZER, **kwargs)
#             goodness = -1 * waic(reg, X, Y) # To be maximized, so -1
#         except:
#             goodness = float('nan')
#
#         CACHE_MODELS[hash_key] = {'model': reg, 'value': goodness}
#
#         return goodness
#
#     # Make the sampler behave in a deterministic way
#     sampler = optuna.samplers.TPESampler(seed=seed)
#
#     study = optuna.create_study(direction='maximize', sampler=sampler)
#     study.optimize(objective, n_trials=restarts, n_jobs=n_jobs,
#                    timeout=timeout)
#
#     return study.best_params, study.best_value
















