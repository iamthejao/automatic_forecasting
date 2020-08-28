from tensorflow_probability import distributions as tfd
import numpy as np
import kernel_setup

FORCE_RERUN_IF_EXISTS = True
NORMALIZED_METRICS = False

DATA_FOLDERS = ["calls"]

FILTER = {'monthly':[],
          'quarterly':[],
          'weekly': [],
          'calls': [],
          'gas': []}

# This takes the x axis and divide by this value
# The x-axis is generated as x = np.linspace(1, len(train)+1, 1) / DATA_FOLDER_PERIOD[period]
DATA_FOLDER_PERIOD = {"monthly": 12, "quarterly": 4, "daily": 365.25,
                      "weekly": 365.25/7, "calls": 169*5, "electricity": 365.25,
                      "gas": 365.25/7.0}

# one observation every month (12 obs = 1y) or one observation every 3 months (4 obs = 1y)

DATA_ROOT = "data_full/multiple_seasonality"
RESULT_ROOT = "from_cluster/all ts/results_sparse"

# This is the experiment setup dictionary, if you delete here it wont run
# even if the key exists in the others

# Electricity ds has 3 periodic components
periods = [169.0, 169*5.0]#[7., 354.37, 365.25]#
max_p = max(periods)
year_periods = [p/max_p for p in periods]
print(year_periods)
#year_periods = [1.0]
EXPERIMENTS = \
    {
        #"SGPR 100 WAIC": kernel_setup.sm_linear_rbf_np(year_periods),
        #"SVGP 100 WAIC": kernel_setup.sm_linear_rbf_np(year_periods),
        "SVGP 200 WAIC": kernel_setup.sm_linear_rbf_np(year_periods),
        #"GPR SP WAIC": kernel_setup.sm_linear_rbf_np(year_periods)
    }


no_priors = {'kernels': {}}

priors_alessio = {
    'kernels': {
        'linear': {'variance': tfd.LogNormal(-1.6, np.float32(1.0))},
        'periodic': {'variance': tfd.LogNormal(-1.6, np.float32(1.0)),
                         'lengthscale': tfd.LogNormal(0.35, np.float32(1.44)),
                         'period': tfd.LogNormal(0.0, 0.2)},
        'rbf': {'variance': tfd.LogNormal(-1.6, np.float32(1.0)),
                'lengthscale': tfd.LogNormal(1.04, np.float32(0.75))},
        'mixture': {
            0 : {'variance': tfd.LogNormal(-1.6, np.float32(1.0)),
                    'lengthscale': tfd.LogNormal(-0.71, np.float32(0.84))},
            1 : {'variance': tfd.LogNormal(-1.6, np.float32(1.0)),
                    'lengthscale': tfd.LogNormal(0.97, np.float32(0.73))}
        }
    },
    'likelihood': tfd.LogNormal(-1.6, np.float32(1.0)),
}

# Force error if does not find
PRIOR_DEFAULT = None
PRIORS_EXPERIMENT = {
    "SGPR 100 WAIC": priors_alessio,
    "SVGP 100 WAIC": priors_alessio,
    "SVGP 200 WAIC": priors_alessio,
    "GPR SP WAIC": priors_alessio
}

LIKELIHOOD_DEFAULT = 'Gaussian'
LIKELIHOOD_EXPERIMENT = {}

MODEL_DEFAULT  = None
MODEL_EXPERIMENT = {"SGPR 100 WAIC": "SGPR",
                    "SVGP 100 WAIC": "SVGP",
                    "SVGP 200 WAIC": "SVGP",
                    "GPR SP WAIC": "GPR"}

# (Number of points, fixed or train) -> train is False, fixed is True
INDUCING_DEFAULT  =  None
INDUCING_EXPERIMENT = { "SGPR 100 WAIC": (100, False),
                        "SVGP 100 WAIC": (100, False),
                        "SVGP 200 WAIC": (200, False),
                        "GPR SP WAIC": (100, False)
                        }

SCORE_DEFAULT = 'waic'
SCORE = {key: 'waic' for key in EXPERIMENTS.keys()}

SEED = 0 #0x0D15EA5E
N_RESTARTS_DEF = 5
N_RESTARTS = {}
SAVE_PLOTS = True
# Save model 5 restarts
CHECKPOINTS = [1, 5]#, 10]























# OLD STUFF

# Test 2 - Checking if both kernels get the same result
#SM_test = model.ExtraKernels.SpectralMixture(1)
#SM_test.Cosine.lengthscale.set_prior(pLscal_long)
#SM_test.rbf.lengthscale.set_prior(pLscal_long)

#FILTER = {'monthly': pickle.load(open('idx_ts_giorgio.list', 'rb'))}

# Priors
# pLscal_long = GPy.priors.LogGaussian(0.28, 0.8)
# pLscal_short = GPy.priors.LogGaussian(-1.03, 1.03)
# pLscal = GPy.priors.LogGaussian(-.34, 1)
# pVariance = GPy.priors.LogGaussian(-0.6, 1.0)
#
# priors = {
#     'kernels': {
#         'linear': {'variances': pVariance},
#         'std_periodic': {'variance': pVariance,
#                          'lengthscale': pLscal_long},
#         'rbf': {'variance': pVariance,
#                 'lengthscale': pLscal_short},
#         'Mixture': {'variance': pVariance} # Lengthscale prior is set by hand down there.
#     },
#     'likelihood': pVariance,
# }
#
# priors_logfit = {
#     'kernels': {
#         'linear': {'variances': GPy.priors.LogGaussian(-1.2, 0.05)},
#         'std_periodic': {'variance': GPy.priors.LogGaussian(-1., 0.4),
#                          'lengthscale': GPy.priors.LogGaussian(0.22, 0.84)},
#         'rbf': {'variance': GPy.priors.LogGaussian(-1.4, 0.72),
#                 'lengthscale': GPy.priors.LogGaussian(-0.24, 0.84)},
#         'Mixture': {'variance': GPy.priors.LogGaussian(-1.42, 0.72),
#                     'lengthscale': GPy.priors.LogGaussian(0.0, 0.9)} # Lengthscale prior is set by hand down there.
#     },
#     'likelihood': GPy.priors.LogGaussian(-1., 1.8),
# }

#
#
#
# # No prior on SM
# priors_default = {
#     'kernels': {
#         'linear': {'variances': GPy.priors.LogGaussian(-0.6, 1)},
#         'std_periodic': {'variance': GPy.priors.LogGaussian(-0.6, 1),
#                          'lengthscale': GPy.priors.LogGaussian(-.34, 1)},
#         'rbf': {'variance': GPy.priors.LogGaussian(-0.6, 1),
#                 'lengthscale': GPy.priors.LogGaussian(-.34, 1)},
#     },
#     'likelihood': GPy.priors.LogGaussian(-.6, 1),
# }
#
#
#


# Version with best results
# priors_ep = {
#     'kernels': {
#         'linear': {'variances': GPy.priors.LogGaussian(-1.2, 0.7)}, #0.05 is crazy small
#         'std_periodic': {'variance': GPy.priors.LogGaussian(-1., 0.5), #0.47
#                          'lengthscale': GPy.priors.LogGaussian(0.82, 1.52),
#                          'period': GPy.priors.LogGaussian(0,0.2)},
#         'rbf': {'variance': GPy.priors.LogGaussian(-1.25, 0.87),
#                 'lengthscale': GPy.priors.LogGaussian(0.18, 1.45)},
#         'Mixture': {
#
#             'default': {'variance': GPy.priors.LogGaussian(-1.32, 0.87),
#                     'lengthscale': GPy.priors.LogGaussian(1.0, 1.81)}, # GPy.priors.LogGaussian(1.63, 1.81)}, this can only be wrong
#             0 : {'variance': GPy.priors.LogGaussian(-1.32, 0.87),
#                     'lengthscale': GPy.priors.LogGaussian(-1.8, 0.35)},
#             1 : {'variance': GPy.priors.LogGaussian(-1.32, 0.87),
#                     'lengthscale': GPy.priors.LogGaussian(0.4, 0.33)}
#         }
#     },
#     'likelihood': GPy.priors.LogGaussian(-1., 1.8),
# }
