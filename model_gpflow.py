import gpflow
from gpflow.utilities import set_trainable, parameter_dict, deepcopy
import optuna
import numpy as np
from scores import waic_gpflow
from decorators import timeit
import tensorflow as tf
from sklearn.cluster import KMeans
import time
from gpflow.ci_utils import ci_niter, ci_range
from gpflow.utilities import print_summary
from gpflow.kernels import AnisotropicStationary
import shutil
import tensorflow_probability as tfp
import os
gpflow.config.set_default_float(np.float32)
gpflow.config.set_default_jitter(1E-2)
config = gpflow.config.config()

CACHE_MODELS = {}
TMP_NAME = 'tmp2/'
CURRENT_MODEL = None
TRAIN_LENGTH = None

ADAM_LR = 0.01
GAMMA_LR = 0.1

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 256

MAX_ITERS= 5000
MIN_ITERS = 1000

CHECKPOINT_EVERY = 50
CHECK_EVERY = 50

PRINT_EVERY = 100

REQ_IMPROV = 50
REQ_IMPROV_PCT = 0.001 # Improvement of at least 0.001

MAX_REQ_IMPROV = 500 # After 500 without improvement it stops
UPPER_BOUND_NORM = np.sqrt(10) * 10
MIN_NORM = np.sqrt(10)
NORM_TOO_SMALL = 1E-1 # If norm gets lower than this, it stops
CV_THRES = 0.01
SCIPY_OPT = False

# Rule for stopping
# (LAST_IMPROV > REQ_IMPROV AND NORM < MIN_NORM) OR (LAST_IMPROV > MAX_REQ_IMPROV AND NORM < UPPER_BOUND_NORM) OR (NORM < NORM_TOO_SMALL)
print('[SCIPY OPT: ]', SCIPY_OPT)

def rename_parameter(name, old_parameter: gpflow.Parameter):
    new_parameter = gpflow.Parameter(
        old_parameter,
        trainable=old_parameter.trainable,
        prior=old_parameter.prior,
        name=name,  # tensorflow is weird and adds ':0' to the name
        transform=old_parameter.transform,
    )
    old_parameter.transform
    return new_parameter

class CosineLikeGPy(AnisotropicStationary):
    """
    The Cosine kernel. Functions drawn from a GP with this kernel are sinusoids
    (with a random phase).  The kernel equation is

        k(r) = σ² cos{2πd}

    where:
    d  is the sum of the per-dimension differences between the input points, scaled by the
    lengthscale parameter ℓ (i.e. Σᵢ [(X - X2ᵀ) / ℓ]ᵢ),
    σ² is the variance parameter.
    """

    def K_d(self, d):
        d = tf.reduce_sum(d, axis=-1)
        return self.variance * tf.cos(d) #2 * np.pi *


class ExtraKernels:

    @staticmethod
    def SpectralMixture(Q, name='SpectralMixture'):
        """
        rbf.variance plays the role of the weights (wi)
        rbf.lengthscale plays the role of SM variance (vq)
        cos.lengthscale plays the role of SM means (uq)
        """
        for i in range(Q):
            name = 'mixture_{0}.'.format(i)
            cos = CosineLikeGPy(variance=1)
            set_trainable(cos.variance, False)
            cos.variance = rename_parameter(name + 'cos.variance', cos.variance)
            cos.lengthscales = rename_parameter(name + 'cos.lengthscale', cos.lengthscales)

            rbf = gpflow.kernels.RBF()
            rbf.variance = rename_parameter(name + 'rbf.variance', rbf.variance)
            rbf.lengthscales = rename_parameter(name + 'rbf.lengthscale', rbf.lengthscales)
            # Rbf's sigma will take the weight
            # Cosine's lengthscale will take the mean
            sm = cos * rbf
            if i == 0:
                SM = sm
            else:
                SM += sm
        return SM

    @staticmethod
    def Quasiperiodic(period=None):
        name = 'quasiperiodic.'
        per = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), period=1.0)
        if period is not None:
            per.period.assign(period)
            set_trainable(per.period, False)
        per.base_kernel.variance = rename_parameter(name + 'periodic.variance', per.base_kernel.variance)
        per.base_kernel.lengthscales = rename_parameter(name + 'periodic.lengthscale', per.base_kernel.lengthscales)
        per.period = rename_parameter(name + 'periodic.period', per.period)

        rbf = gpflow.kernels.RBF(variance=1.0)
        set_trainable(rbf.variance, False)

        rbf.variance = rename_parameter(name + 'rbf.variance', rbf.variance)
        rbf.lengthscales = rename_parameter(name + 'rbf.lengthscale', rbf.lengthscales)

        qper = per * rbf
        return qper


def set_kernel_parameters(param_dict, kernel):
    var_dict = {var.name: var for var in kernel.trainable_variables}
    for param in param_dict.keys():
        var_dict[param].assign(param_dict[param])
    return kernel


def set_kernel_priors(priors, kernel):
    tmp = deepcopy(kernel)
    parameters = parameter_dict(tmp)
    for variable in parameters.values():
        name, trainable = variable.name, variable.trainable
        if trainable:
            names = name.split(".")
            var = names[-1].split(":")[0]
            type_ = names[0]
            if 'mixture' in type_:
                idx = int(type_.split('_')[-1])
                variable.prior = priors['kernels']['mixture'][idx][var]
            else:
                type_ = type_.split('_')[0]
                variable.prior = priors['kernels'][type_][var]
    return tmp


def suggest_kernel_initialization(trial: optuna.Trial, kernel):
    parameters = parameter_dict(kernel)
    for variable in parameters.values():
        if variable.trainable:
            name = variable.name
            var = name.split(".")[-1]
            if 'variance' in var:
                suggestion = trial.suggest_uniform(name, 0.1, 1.0)
            elif 'lengthscale' in var:
                suggestion = trial.suggest_uniform(name, 0.5, 2.0)
            elif 'period' in var:
                suggestion = trial.suggest_uniform(name, 0.0, 1.5)
            else:
                raise Exception(f"Not found parameter {var}")
            variable.assign(suggestion)
    return kernel


# The optimizer maximizes scores
def score(reg, inputs):
    X, Y = inputs.data['xtrain'], inputs.data['ytrain']
    if inputs.score == 'll':
        # Log likelihood + Log Prior
        scoreValue = reg.log_posterior_density() + reg.log_prior_density()
    elif inputs.score == 'waic':
        # waic = -predLogLik + varLik
        # We want to maximize predLogLik, then multiply by -1
        scoreValue = -1 * waic_gpflow(reg, X, Y)
    else:
        raise NotImplemented('Score {0} not implemented'.format(score))
    return scoreValue


def build_model(input, kernel):

    if input.model == 'GPR':
        if input.likelihood == 'Gaussian':
            reg = gpflow.models.GPR(data=(input.data['xtrain'], input.data['ytrain']),
                                                                    kernel=kernel)
        else:
            raise Exception("Wrong likelihood name: {0}".format(input.likelihood))
    elif input.model == 'SGPR':
        if input.likelihood == 'Gaussian':
            Z = KMeans(n_clusters=input.inducing[0]).fit(input.data['xtrain']).cluster_centers_
            reg = gpflow.models.SGPR(data=(input.data['xtrain'], input.data['ytrain']),
                                     kernel=kernel, inducing_variable=Z)

            if input.inducing[1]:# Fix inducing
                set_trainable(reg.inducing_variable, False)
        else:
            raise Exception("Wrong likelihood name: {0}".format(input.likelihood))
    elif input.model == 'SVGP':
        if input.likelihood ==  'Gaussian':

            Z = KMeans(n_clusters=input.inducing[0]).fit(input.data['xtrain']).cluster_centers_
            reg = gpflow.models.SVGP(kernel=kernel, inducing_variable=Z,
                                     likelihood=gpflow.likelihoods.Gaussian())

            # Trick to stop adam from optimizing q_mu and q_sqrt
            set_trainable(reg.q_mu, False)
            set_trainable(reg.q_sqrt, False)

            if input.inducing[1]:# Fix inducing
                set_trainable(reg.inducing_variable, False)
        else:
            raise NotImplemented("Likelihood {0} not implemented yet for SVGP".format(input.likelihood))
    else:
        raise Exception("Wrong model name: {0}".format(input.model))

    # Inserting prior on likelihood variance
    if 'likelihood' in input.priors.keys():
        reg.likelihood.variance.prior = input.priors['likelihood']

    reg.likelihood.variance = rename_parameter('likelihood.variance',
                                                reg.likelihood.variance)

    return reg

def optimization_step_natgrad(loss_f, parameters, optimizer):
  variables = [p.unconstrained_variable for p in parameters]
  with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(variables)
        loss = loss_f()
  grads = tape.gradient(loss, variables)
  optimizer._natgrad_apply_gradients(grads[0], grads[1], parameters[0], parameters[1], None)
  return loss, grads

def optimization_step(loss_f, variables, optimizer):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(variables)
        loss = loss_f()
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return loss, grads

def optimize_scipy(loss_f, variables):
    OPT = gpflow.optimizers.Scipy()
    OPT.minimize(loss_f, variables, options=dict(maxiter=ci_niter(10000)))

def optimize(manager, loss_f, optimizer_var_tuples, iters=MAX_ITERS):

  print(CURRENT_MODEL)

  start = time.time()

  tf_opt = tf.function(optimization_step)
  tf_opt_natgrad = tf.function(optimization_step_natgrad)

  best_measure = 1E9
  best_measure_it = 0
  best_loss = 1E9
  last_saved = 0
  its_per_set = TRAIN_LENGTH // BATCH_SIZE if CURRENT_MODEL == 'SVGP' else 1
  print("[ITERATIONS PER WHOLE DS]: ", its_per_set)
  last_losses = []
  last_improvement = 0

  for i in range(ci_niter(iters)):
    
    for opt, var in optimizer_var_tuples:

        if type(opt) is gpflow.optimizers.natgrad.NaturalGradient:
            loss, grads = tf_opt_natgrad(loss_f, var, opt)
        else:
            loss, grads = tf_opt(loss_f, var, opt)

            if (i % CHECK_EVERY == 0) or (i % PRINT_EVERY == 0):
                # Filtering out grad of the inducing variable
                only_vars = [g for g in grads if len(g.shape) == 0]
                grad_tensor = tf.convert_to_tensor(only_vars)
                norm = tf.linalg.norm(grad_tensor)

    # Only get the last 100 iterations
    last_losses.append(loss)
    last_losses = last_losses[-3*its_per_set:]
    mean_loss = tf.math.reduce_mean(last_losses)
    measure = mean_loss if CURRENT_MODEL == 'SVGP' else loss

    # Check if improved after whole data passed
    if (measure < best_measure) and ((i - best_measure_it) >= its_per_set):
        last_improvement = 0
        best_measure = measure
        best_measure_it = i
        best_loss = loss
        if i > MIN_ITERS:
            if (i - last_saved) >= CHECKPOINT_EVERY:
                manager.save()
                last_saved = i
    else:
        last_improvement += 1


    if i % PRINT_EVERY == 0:
        name = repr(opt).split(".")[-1].split(" ")[0]
        tf.print(f"{name} at iteration {i}: batch loss {loss}")
        tf.print(f"Best Loss: batch loss {best_loss}")
        tf.print(f"Last improvement {last_improvement} iterations ago")
        tf.print(f"Mean Loss {mean_loss:.5f}.")
        tf.print(f"Best measure so far {best_measure:.5f}.")
        tf.print(f"At iteration {best_measure_it:.5f}.")
        if type(opt) != gpflow.optimizers.natgrad.NaturalGradient:
            tf.print(f"{name} at iteration {i}: norm {norm}")
        tf.print("")

    if i % CHECK_EVERY == 0:
        if i > MIN_ITERS:
            if ((last_improvement >= REQ_IMPROV) and (norm < MIN_NORM) and (mean_loss < CV_THRES)) or (norm < NORM_TOO_SMALL) or ((last_improvement >= MAX_REQ_IMPROV) and (norm < UPPER_BOUND_NORM)):
                tf.print(f"Stopped at iteration: {i}")
                tf.print(f"Last improvement: {last_improvement}")
                tf.print(f"Adam norm: {norm}")
                tf.print("Training Elapsed: ", time.time() - start)

                if (measure < best_measure):
                    manager.save()
                return

  tf.print("Training Elapsed: ", time.time() - start)
  if (measure < best_measure):
      manager.save()
  return

def train(manager, loss_f):
    global CURRENT_MODEL
    # Getting model from checkpoint obj
    reg = manager._checkpoint.model
    OPTIMIZER = tf.optimizers.Adam(learning_rate=ADAM_LR, amsgrad=True)
    NATGRAD = gpflow.optimizers.NaturalGradient(GAMMA_LR)
    if type(reg) is gpflow.models.SVGP:
        CURRENT_MODEL = 'SVGP'
        variational_params = [(reg.q_mu, reg.q_sqrt)]
        trainable_vars = reg.trainable_variables
        opt_var_tuple = [(NATGRAD, variational_params[0]), (OPTIMIZER, trainable_vars)]
        optimize(manager, loss_f, opt_var_tuple)
    else:
        CURRENT_MODEL = '#GPR'
        trainable_vars = reg.trainable_variables
        if SCIPY_OPT:
            #tf_opt = tf.function(optimize_scipy)
            start = time.time()
            optimize_scipy(loss_f, trainable_vars)
            print("Trained in: ",time.time()-start)
        else:
            opt_var_tuple = [(OPTIMIZER, trainable_vars)]
            optimize(manager, loss_f, opt_var_tuple)

def compile_loss(reg, input):
    global TRAIN_LENGTH
    if 'GPR' in input.model:
        TRAIN_LENGTH = reg.data[0].shape[0]
        return reg.training_loss_closure()
    else:
        data = (input.data['xtrain'], input.data['ytrain'])
        TRAIN_LENGTH = data[0].shape[0]
        N = data[0].shape[0]
        data_minibatch = tf.data.Dataset.from_tensor_slices(data).prefetch(AUTOTUNE).repeat().shuffle(N).batch(BATCH_SIZE)
        data_minibatch_it = iter(data_minibatch)
        return reg.training_loss_closure(data_minibatch_it)

@timeit
def bayes_opt_training(input, kernel, restarts=20, seed=0, n_jobs=1,
                       timeout=None):

    reg = build_model(input, kernel)
    print_summary(reg)

    # Is it making it reuse the same model??!?!? I dont think so because of the
    # suggest initialization
    loss_f = compile_loss(reg, input)

    def objective(trial: optuna.Trial, reg=reg, loss_f=loss_f):

        # Changes variables on kernel
        suggest_kernel_initialization(trial, kernel)

        # Make cache key to get save models
        hash_key = hash(tuple(np.round([var.numpy() for var in kernel.variables], 3).tolist()))
        key = (hash_key, input.name, trial.number)

        #try:

        if os.path.exists(TMP_NAME):
            shutil.rmtree(TMP_NAME)
        os.mkdir(TMP_NAME)
        ckpt = tf.train.Checkpoint(model=reg)
        manager = tf.train.CheckpointManager(ckpt, TMP_NAME, max_to_keep=1)
        train(manager, loss_f)

        # Restoring best loss model
        status = ckpt.restore(manager.latest_checkpoint)

        del manager
        del ckpt

        shutil.rmtree(TMP_NAME)

        goodness = score(reg, input)
        #except:
        #  goodness = float('nan')
          
        CACHE_MODELS[key] = {'model': deepcopy(reg), 'value': goodness}

        return goodness

    # Make the sampler behave in a deterministic way
    np.random.seed(seed)
    tf.random.set_seed(seed)
    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=restarts, n_jobs=n_jobs,
                   timeout=timeout)

    return study
