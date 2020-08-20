import matplotlib.pyplot as plt
import seaborn as sns
import gpflow
import numpy as np
import scipy.stats as sps
from scores import crpsGaussian, loglikelihood
from experiment_config_gpflow import NORMALIZED_METRICS
sns.set()

def test(model: gpflow.models.SVGP):
    model.predict_f()


class Analyst:

    def __init__(self, model: gpflow.models.GPR, data: dict, model_name, scoreType, period=1):

        self.data = data
        self.model = model
        self.model_name = model_name
        self.scoreType = scoreType
        self.period = int(np.ceil(period))

    def measures_dict(self, pred, sigma, truth, name):

        results = {'name': name}

        results['mae'] = self.mae(pred, truth)
        results['smape'] = self.smape(pred, truth)
        results['mase_1'] = self.mase(pred, truth, m=1)
        results['mase_p'] = self.mase(pred, truth, m=self.period)
        results['period'] = self.period
        results['rmse'] = self.rmse(pred, truth)
        results['ll'] = loglikelihood(pred, sigma, truth)
        results['crps'] = crpsGaussian(pred, sigma, truth)
        results['fcast_len'] = len(truth)

        return results

    def plot(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(13, 10))

        minx = min(self.data['xtrain'].min(), self.data['xtest'].min())
        maxx = max(self.data['xtrain'].max(), self.data['xtest'].max())
        stdx = self.data['xtrain'].std()
        from_ = minx - stdx
        to_ = maxx + stdx
        size = self.data['xtrain'].shape[0]

        xx = np.float32(np.linspace(from_, to_, int(size * 2)).reshape([int(size * 2), 1]))
        mean, var = self.model.predict_y(xx)

        ## generate 10 samples from posterior
        #samples = self.model.predict_f_samples(self.data['xtrain'], 10)

        ## plot
        ax.plot(self.data['xtrain'], self.data['ytrain'], "kx", mew=2)
        ax.plot(self.data['xtest'], self.data['ytest'], "rx", mew=2, label='test')
        ax.plot(self.data['xtest'], self.data['ytest'], "r--", lw=1.0)
        ax.plot(xx, mean, "C0", lw=2, label='GPFcast')
        ax.fill_between(
            xx[:, 0],
            mean[:, 0] - 2.2414 * np.sqrt(var[:, 0]),
            mean[:, 0] + 2.2414 * np.sqrt(var[:, 0]),
            color="C0",
            alpha=0.2,
        )

        if type(self.model) in (gpflow.models.SVGP, gpflow.models.SGPR):
            Z = self.model.inducing_variable.variables[0].numpy().ravel()
            ax.plot(Z, [0 for i in range(len(Z))], "ro")

        #ax.plot(self.data['xtrain'], samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
        ax.set_xlim([minx-0.5, maxx+0.5])
        return ax

    def check_forecast(self, save_file=None, comparisons={}, closeFig=False):

        sns.set(style='dark', font_scale=1.25)

        xtest, ytest = self.data['xtest'], self.data['ytest']
        size_train = len(self.data['xtrain'])
        info = []
        fcastsDict = {'truth': ytest}

        fcasts, varFcasts = self.model.predict_y(xtest)
        stdFcasts = np.sqrt(varFcasts)

        fcasts_train, varFcasts_train = self.model.predict_y(self.data['xtrain'])
        stdFcasts_train = np.sqrt(varFcasts_train)
        
        fcastsDict[self.model_name] = {'test': {'mean': fcasts, 'std': stdFcasts},
                                       'train': {'mean': fcasts_train, 'std': stdFcasts_train}}

        results = self.measures_dict(fcasts, stdFcasts, ytest, self.model_name)
        results['train_len'] = size_train
        textstr = 'GP mae = {0:.3f} \n'.format(results['mae'])
        info.append(results)

        fig, ax = plt.subplots(figsize=(13, 10))
        self.plot(ax=ax)
        for key in comparisons.keys():
            std = (comparisons[key]['95'] - comparisons[key]['fcast'])/sps.norm.ppf(0.975)
            fcastsDict[key] = {'mean': comparisons[key]['fcast'], 'std': std}
            res_comp = self.measures_dict(comparisons[key]['fcast'], std, ytest, key)
            res_comp['train_len'] = size_train
            ax.plot(xtest, comparisons[key]['fcast'], label=key)
            textstr += '{0} mae = {1:.3f} \n'.format(key, res_comp['mae'])
            info.append(res_comp)

        ax.legend()

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(0.15, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        if save_file is not None:
            fig.savefig(save_file)

        if closeFig:
            plt.close(fig)

        return info, fcastsDict

    def make_component_plot(self, save_file=None, type_='line'):

        xtest, ytest = self.data['xtest'], self.data['ytest']
        comp, name = self.components_predictions(xtest)

        fig, ax = plt.subplots(2, 1, figsize=(15, 15))
        self.plot(ax=ax[0])
        ax[0].legend()

        self.plot(ax=ax[1])
        for i in range(len(name)):
            if type_ is 'scatter':
                ax[1].scatter(xtest, comp[:, i], label=name[i], alpha=0.7, s=5)
            elif type_ is 'line':
                ax[1].plot(xtest, comp[:, i], label=name[i], alpha=0.7)
            else:
                raise Exception("Wrong plot type.")
        ax[1].legend()

        if save_file is not None:
            fig.savefig(save_file)
            plt.close(fig)

    @staticmethod
    def mae(prediction, truth):
        return np.nanmean(np.abs(truth - prediction))

    @staticmethod
    def smape(prediction, truth):
        return np.nanmean(np.abs(truth - prediction) / ((np.abs(truth) + np.abs(prediction)) / 2.0))

    def mase(self, prediction, truth, m=1):
        e_num = self.mae(prediction, truth) #np.mean(np.abs(truth - prediction))
        if m == 1:
            e_div = np.mean(np.abs(np.diff(self.data['ytrain'], axis=0)))
        else:
            y_flat = self.data['ytrain'].ravel()
            e_div = np.mean(np.abs([y_flat[i + m] - y_flat[i] for i in range(len(y_flat) - m)]))
        return e_num / e_div

    @staticmethod
    def rmse(prediction, truth):
        return np.sqrt(np.nanmean(np.power(truth - prediction, 2)))

    @staticmethod
    def symm_improvement(first, second):
        return (first - second)/((first + second) * 0.5)

    def components_predictions(self, points):
        return kernel_components_pred_gpflow(self.model, points, self.data['xtrain'], self.data['ytrain'])

#
def kernel_components_pred_gpflow(model, xtest, xtrain, ytrain):

    bigKern = model.kernel
    parts = model.kernel.kernels
    X, Y = xtrain, ytrain

    # Put all components of mixture together
    parts = [k.parameters[0].name.split(".")[0] if 'product' in k.name else k.name for k in parts]
    uparts = parts

    preds_components = np.zeros([len(xtest), len(uparts)])

    # Whole model training covariance
    K = bigKern.K(X).numpy() + np.eye(len(X)) * 1E-8
    K += np.eye(len(X)) * model.likelihood.variance.numpy()

    # below we implement the following
    # pred = Kx * K^(-1)*Y = Kx * (L Lt )^(-1)*Y = Kx * Lt(-1) * L(-1)*Y
    L = np.linalg.cholesky(K)
    A = np.linalg.solve(L, Y)

    for i in range(len(bigKern.kernels)):

        k = bigKern.kernels[i]
        name = uparts[i]

        #componentModel = gpflow.models.GPR(data=model.data, kernel=k, noise_variance=1E-5)
        # Kernel for xtest of the component
        Kx = k.K(xtest, X)#componentModel.kernel.K(xtest, X)

        # B = Kx * Lt(-1), where Lt(-1) is the inverse of L transpose
        # we compute Kx * Lt(-1) as follows.
        # Kx * Lt(-1) = [[Kx * Lt(-1)]T ]T = [Lt(-1) * Kx_t]T = [Lt(-1) * Kx]T]T
        # implementation with inv
        B = np.transpose(np.linalg.solve(L, np.transpose(Kx)))
        preds = np.matmul(B, A)
        preds_components[:, i] = preds.reshape(len(xtest))

    return preds_components, uparts






