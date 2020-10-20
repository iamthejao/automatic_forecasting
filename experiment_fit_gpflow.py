import os
from pathlib import Path
import pandas as pd
import time
import experiment_config_gpflow as config
import experiment_funcs_gpflow as ef

import warnings
warnings.filterwarnings(action='ignore', module='GPy')

records = pd.DataFrame()

# Data for acquiring MP
experiments = []
triggerAt = 1
tstart = time.time()
count_experiments = 0.0
for folder in config.DATA_FOLDERS:

    location = Path(config.DATA_ROOT).joinpath(folder)
    trainFiles = list(filter(lambda name: 'train' in name, os.listdir(location))) ##

    for file in trainFiles:

        data = ef.read_experiment_data(folder, location, file)
        idx = ef.get_idx(file)

        print(folder, file)

        # Now the experiment (different kernels) start
        for kName in list(config.EXPERIMENTS.keys()):
            print(kName)
            inputs = ef.ExperimentInput(kName, data, config.EXPERIMENTS[kName],
                                        config.LIKELIHOOD_EXPERIMENT.get(kName, config.LIKELIHOOD_DEFAULT),
                                        config.MODEL_EXPERIMENT.get(kName, config.MODEL_DEFAULT),
                                        config.PRIORS_EXPERIMENT.get(kName, config.PRIOR_DEFAULT),
                                        config.SCORE.get(kName, config.SCORE_DEFAULT),
                                        config.INDUCING_EXPERIMENT.get(kName, config.INDUCING_DEFAULT),
                                        config.N_RESTARTS.get(kName, config.N_RESTARTS_DEF))

            if config.FORCE_RERUN_IF_EXISTS:
                ef.train_and_save(inputs)
            else:
                result_path = Path(config.RESULT_ROOT).joinpath(folder).joinpath(str(idx))
                model_file = result_path.joinpath('{0} {1}.predictions.pbz2'.format(kName, config.CHECKPOINTS[-1]))

                # Check so we dont rerun the same model if it is already saved from a previous run
                if model_file.exists() is False:
                    ef.train_and_save(inputs)
                else:
                    print("{0} already exists for {1}-{2}".format(kName, folder, idx))


