import gpflow
import model_gpflow as model
from gpflow.utilities import set_trainable, print_summary, parameter_dict

# Builds and rename variables inside the kernels for easy setup of priors and others
def sm_linear_rbf_1p(known_period=True):

    # Periodic with SqExp base
    periodic_known = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), period=1.0)
    if known_period:
        set_trainable(periodic_known.period, False)
    periodic_known.base_kernel.variance = model.rename_parameter('periodic.variance',
                                                                 periodic_known.base_kernel.variance)
    periodic_known.base_kernel.lengthscales = model.rename_parameter('periodic.lengthscale',
                                                                     periodic_known.base_kernel.lengthscales)
    periodic_known.period = model.rename_parameter('periodic.period', periodic_known.period)

    # Linear
    linear = gpflow.kernels.Linear()
    linear.variance = model.rename_parameter('linear.variance', linear.variance)

    # RBF
    rbf = gpflow.kernels.RBF()
    rbf.variance = model.rename_parameter('rbf.variance', rbf.variance)
    rbf.lengthscales = model.rename_parameter('rbf.lengthscale', rbf.lengthscales)

    # SM
    Q = 2
    SM_Q = model.ExtraKernels.SpectralMixture(Q)

    return SM_Q + linear + rbf + periodic_known

def sm_linear_rbf_np(periods=[], fixed=True):

    # Periodic with SqExp base
    periodics = [gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), period=p) for p in periods]
    if fixed:
        for k in periodics:
            set_trainable(k.period, False)

    for i in range(len(periodics)):
        k = periodics[i]
        k.base_kernel.variance = model.rename_parameter(f'periodic_{i}.variance',
                                                                      k.base_kernel.variance)
        k.base_kernel.lengthscales = model.rename_parameter(f'periodic_{i}.lengthscale',
                                                                          k.base_kernel.lengthscales)
        k.period = model.rename_parameter(f'periodic_{i}.period', k.period)

    # Linear
    linear = gpflow.kernels.Linear()
    linear.variance = model.rename_parameter('linear.variance', linear.variance)

    # RBF
    rbf = gpflow.kernels.RBF()
    rbf.variance = model.rename_parameter('rbf.variance', rbf.variance)
    rbf.lengthscales = model.rename_parameter('rbf.lengthscale', rbf.lengthscales)

    # SM
    Q = 2
    SM_Q = model.ExtraKernels.SpectralMixture(Q)

    K = SM_Q + linear + rbf
    for p in periodics:
        K += p

    return K
