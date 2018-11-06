from ML.model_error import ModelError
from utils.settings import TARGET_COL


def sim_model(Model, data, pred_var=TARGET_COL, sub_pred_var=None, n_sim=1):
    mean_err = 0
    median_err = 0
    mean_err_rel = 0
    median_err_rel = 0

    if sub_pred_var is None:
        model = Model(data)
    else:
        model = Model(data, sub_pred_var=sub_pred_var)
    for i in range(n_sim):
        if i > 0:
            if sub_pred_var is None:
                model = Model(data)
            else:
                model = Model(data, sub_pred_var=sub_pred_var)

        model_error = ModelError(model, pred_var=pred_var)
        error = model_error.get_error()
        error_rel = model_error.get_error(relative=True)

        mean_err += error.mean() / n_sim
        median_err += error.median() / n_sim
        mean_err_rel += error_rel.mean() / n_sim
        median_err_rel += error_rel.median() / n_sim

    print('================= ' + model.name + ' =================\n' +
          'Données de test :\n' +
          '\t{} lignes.\n'.format(str(model.data_val.shape[0])) +
          'En moyenne pour une prédiction :\n' +
          '\t{} pour {}.\n'.format(str(round(model.data_val[pred_var].mean(), 2)), pred_var) +
          "\t{} d'erreur en valeur absolue.\n".format(str(round(mean_err, 2))) +
          "\t{} d'erreur relative en valeur absolue.\n\n".format(str(round(mean_err_rel, 3))) +
          'Médiane pour les prédictions :\n' +
          '\t{} pour {}.\n'.format(str(round(model.data_val.CONDOMINIUM_EXPENSES.median(), 2)), pred_var) +
          "\t{} d'erreur en valeur absolue.\n".format(str(round(median_err, 2))) +
          "\t{} d'erreur relative en valeur absolue.\n".format(str(round(median_err_rel, 3))) +
          '============================================='
          )
