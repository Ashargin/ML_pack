import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.settings import TARGET_COLS

# Settings
sns.set()
sns.set_style('whitegrid')


class ModelError:
    def __init__(self, model, pred_var=TARGET_COLS[0]):
        self.model = model
        self.data = self.model.data_val.copy()
        self.pred_var = pred_var
        self._run_predict()

    def _run_predict(self):
        data = self.data  # Shortcut
        pred_var = self.pred_var

        data['n_predict'] = self.model.predict(data)
        data['error'] = data.n_predict - data[pred_var]
        data['error_rel'] = data.error / data[pred_var]

    def get_error(self, abs_val=True, relative=False, frame=False):
        data = self.data

        error, err_name = (data.error_rel, 'error_rel') if relative else (data.error, 'error')
        if abs_val:
            error = error.abs()
            err_name += '_abs'
        if frame:
            data[err_name] = error
            return data
        else:
            return error

    def info(self):
        data = self.data
        pred_var = self.pred_var
        print(
            '================= ' + self.model.name + ' =================\n' +
            'Données de test :\n' +
            '\t' + str(data.shape[0]) + ' lignes.\n' +
            '\t' + str(len(data.REGION.unique())) + ' régions.\n\n' +
            'En moyenne pour une prédiction :\n' +
            '\t' + str(round(data[pred_var].mean(), 2)) + ' pour ' + self.pred_var + '.\n' +
            '\t' + str(round(data.error.abs().mean(), 2)) + " d'erreur en valeur absolue.\n" +
            '\t' + str(round(data.error_rel.abs().mean(), 3)) + " d'erreur relative en valeur absolue.\n\n" +
            'Médiane pour les prédictions :\n' +
            '\t' + str(round(data[pred_var].median(), 2)) + ' pour ' + self.pred_var + '.\n' +
            '\t' + str(round(data.error.abs().median(), 2)) + " d'erreur en valeur absolue.\n" +
            '\t' + str(round(data.error_rel.abs().median(), 3)) + " d'erreur relative en valeur absolue.\n" +
            '============================================='
        )

    def plot_error_distrib(self, abs_val=True, relative=False, show_mean=False, figure=False, color=None, xlim=None,
                           ylim=None):
        data = self.get_error(abs_val=abs_val, relative=relative, frame=True)
        err_name = 'error'
        if relative:
            err_name += '_rel'
        if abs_val:
            err_name += '_abs'

        error = data[err_name].sort_values()
        x = np.linspace(0, 1, len(error))

        if figure:
            plt.figure()
        if color is None:
            plt.plot(x, error, label='Erreurs : ' + self.model.name)
        else:
            plt.plot(x, error, label='Erreurs : ' + self.model.name, color=color)
        if show_mean:
            m = error.mean()
            plt.plot([0, 1], [m, m], label='Erreur moyenne : ' + self.model.name)
        if not (xlim is None):
            plt.xlim(xlim)
        if not (ylim is None):
            plt.ylim(ylim)
        plt.xlabel('Probabilité')
        plt.ylabel('Quantiles de ' + err_name)
        plt.title('Distribution de ' + err_name + ' pour ' + self.pred_var)
        plt.legend(loc='upper left')
