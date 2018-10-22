import numpy as np
from numpy import ma
import pandas as pd
from skgarden.quantile.ensemble import generate_sample_indices
from skgarden.quantile.utils import weighted_percentile
from sklearn.ensemble.forest import ForestRegressor
from sklearn.utils import check_array, check_X_y

from work.tree import MyDecisionTreeQuantileRegressor


class MyRandomForestQuantileRegressor(ForestRegressor):
    def __init__(self,
                 n_estimators=20,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=160,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(MyRandomForestQuantileRegressor, self).__init__(
            base_estimator=MyDecisionTreeQuantileRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes

    def fit(self, X, y):
        """
        Build a forest from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        X_idx_sorted : array-like, shape = [n_samples, n_features], optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.

        Returns
        -------
        self : object
            Returns self.
        """
        # apply method requires X to be of dtype np.float32

        X, y = check_X_y(
            X, y, accept_sparse="csc", dtype=np.float32, multi_output=False)
        super(MyRandomForestQuantileRegressor, self).fit(X, y)

        self.y_train_ = y
        self.y_train_leaves_ = -np.ones((self.n_estimators, len(y)), dtype=np.int32)
        self.y_weights_ = np.zeros_like(self.y_train_leaves_, dtype=np.float32)

        for i, est in enumerate(self.estimators_):
            if self.bootstrap:
                bootstrap_indices = generate_sample_indices(
                    est.random_state, len(y))
            else:
                bootstrap_indices = np.arange(len(y))

            est_weights = np.bincount(bootstrap_indices, minlength=len(y))
            y_train_leaves = est.y_train_leaves_
            for curr_leaf in np.unique(y_train_leaves):
                y_ind = y_train_leaves == curr_leaf
                self.y_weights_[i, y_ind] = (
                        est_weights[y_ind] / np.sum(est_weights[y_ind]))

            self.y_train_leaves_[i, bootstrap_indices] = y_train_leaves[bootstrap_indices]
        return self

    def predict(self, X, quantiles=None):
        """
        Predict regression value for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        quantile : int, optional
            Value ranging from 0 to 100. By default, the mean is returned.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array of shape = [n_samples]
            If quantile is set to None, then return E(Y | X). Else return
            y such that F(Y=y | x) = quantile.
        """
        # apply method requires X to be of dtype np.float32

        if quantiles is None:
            quantiles = [0.50]

        if quantiles == 'mean':
            quantiles = None
            column_names = ['mean']
        else:
            column_names = [str(round(100 * quantile, 1)) + '%' for quantile in quantiles]
        index = X.index

        X = check_array(X, dtype=np.float32, accept_sparse="csc")
        if quantiles is None:
            preds = super(MyRandomForestQuantileRegressor, self).predict(X)
            return pd.DataFrame(preds, index=index, columns=column_names)

        sorter = np.argsort(self.y_train_)
        X_leaves = self.apply(X)
        quantile_values = np.zeros((X.shape[0], len(quantiles)))
        for i, x_leaf in enumerate(X_leaves):
            mask = self.y_train_leaves_ != np.expand_dims(x_leaf, 1)
            x_weights = ma.masked_array(self.y_weights_, mask)
            weights = x_weights.sum(axis=0)
            for i_q, quantile in enumerate(quantiles):
                quantile_values[i, i_q] = weighted_percentile(self.y_train_, int(100 * quantile), weights, sorter)
        return pd.DataFrame(quantile_values, index=index, columns=column_names)
