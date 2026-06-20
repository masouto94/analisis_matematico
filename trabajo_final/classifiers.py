import numpy as np
import pandas as pd
import numpy.linalg as LA
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg.lapack import dtrtri


class BaseBayesianClassifier:
    def __init__(self):
        pass

    def _estimate_a_priori(self, y):
        a_priori = np.bincount(y.flatten().astype(int)) / y.size
        # Q3: para que sirve bincount?
        # Bincount calcula la frecuencia absoluta en enteros no negativos de una categoría en un array
        # Es necesario para luego, al dividir por y.size, poder obtener la frecuencia relativa de la k-ésima clase, que representa su
        # probabilidad a priori
        return np.log(a_priori)

    def _fit_params(self, X, y):
        # estimate all needed parameters for given model
        raise NotImplementedError()

    def _predict_log_conditional(self, x, class_idx):
        # predict the log(P(x|G=class_idx)), the log of the conditional probability of x given the class
        # this should depend on the model used
        raise NotImplementedError()

    def fit(self, X, y, a_priori=None):
        # if it's needed, estimate a priori probabilities
        self.log_a_priori = (
            self._estimate_a_priori(y) if a_priori is None else np.log(a_priori)
        )

        # now that everything else is in place, estimate all needed parameters for given model
        self._fit_params(X, y)
        # Q4: por que el _fit_params va al final? no se puede mover a, por ejemplo, antes de la priori?
        # porque necesitas la probabilidad a priori para calcular en cada clase las covarianzas

    def predict(self, X):
        # this is actually an individual prediction encased in a for-loop
        m_obs = X.shape[1]
        y_hat = np.empty(m_obs, dtype=int)

        for i in range(m_obs):
            y_hat[i] = self._predict_one(X[:, i].reshape(-1, 1))

        # return prediction as a row vector (matching y)
        return y_hat.reshape(1, -1)

    def _predict_one(self, x):
        # calculate all log posteriori probabilities (actually, +C)
        log_posteriori = [
            log_a_priori_i + self._predict_log_conditional(x, idx)
            for idx, log_a_priori_i in enumerate(self.log_a_priori)
        ]

        # return the class that has maximum a posteriori probability
        return np.argmax(log_posteriori)


class QDA(BaseBayesianClassifier):

    def _fit_params(self, X, y):
        # estimate each covariance matrix
        self.inv_covs = [
            LA.inv(np.cov(X[:, y.flatten() == idx], bias=True))
            for idx in range(len(self.log_a_priori))
        ]
        # Q5: por que hace falta el flatten y no se puede directamente X[:,y==idx]?
        # Q6: por que se usa bias=True en vez del default bias=False?
        self.means = [
            X[:, y.flatten() == idx].mean(axis=1, keepdims=True)
            for idx in range(len(self.log_a_priori))
        ]
        # Q7: que hace axis=1? por que no axis=0?

    def _predict_log_conditional(self, x, class_idx):
        # predict the log(P(x|G=class_idx)), the log of the conditional probability of x given the class
        # this should depend on the model used
        inv_cov = self.inv_covs[class_idx]
        unbiased_x = x - self.means[class_idx]
        return 0.5 * np.log(LA.det(inv_cov)) - 0.5 * unbiased_x.T @ inv_cov @ unbiased_x


class TensorizedQDA(QDA):

    def _fit_params(self, X, y):
        # ask plain QDA to fit params
        super()._fit_params(X, y)

        # stack onto new dimension
        self.tensor_inv_cov = np.stack(self.inv_covs)
        self.tensor_means = np.stack(self.means)

    def _predict_log_conditionals(self, x):
        unbiased_x = x - self.tensor_means
        inner_prod = unbiased_x.transpose(0, 2, 1) @ self.tensor_inv_cov @ unbiased_x

        return 0.5 * np.log(LA.det(self.tensor_inv_cov)) - 0.5 * inner_prod.flatten()

    def _predict_one(self, x):
        # return the class that has maximum a posteriori probability
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))


class QDA_Chol1(BaseBayesianClassifier):
    def _fit_params(self, X, y):
        self.L_invs = [
            LA.inv(cholesky(np.cov(X[:, y.flatten() == idx], bias=True), lower=True))
            for idx in range(len(self.log_a_priori))
        ]

        self.means = [
            X[:, y.flatten() == idx].mean(axis=1, keepdims=True)
            for idx in range(len(self.log_a_priori))
        ]

    def _predict_log_conditional(self, x, class_idx):
        L_inv = self.L_invs[class_idx]
        unbiased_x = x - self.means[class_idx]

        y = L_inv @ unbiased_x

        return np.log(L_inv.diagonal().prod()) - 0.5 * (y**2).sum()


class QDA_Chol2(BaseBayesianClassifier):
    def _fit_params(self, X, y):
        self.Ls = [
            cholesky(np.cov(X[:, y.flatten() == idx], bias=True), lower=True)
            for idx in range(len(self.log_a_priori))
        ]

        self.means = [
            X[:, y.flatten() == idx].mean(axis=1, keepdims=True)
            for idx in range(len(self.log_a_priori))
        ]

    def _predict_log_conditional(self, x, class_idx):
        L = self.Ls[class_idx]
        unbiased_x = x - self.means[class_idx]

        y = solve_triangular(L, unbiased_x, lower=True)

        return -np.log(L.diagonal().prod()) - 0.5 * (y**2).sum()


class QDA_Chol3(BaseBayesianClassifier):
    def _fit_params(self, X, y):
        self.L_invs = [
            dtrtri(
                cholesky(np.cov(X[:, y.flatten() == idx], bias=True), lower=True),
                lower=1,
            )[0]
            for idx in range(len(self.log_a_priori))
        ]

        self.means = [
            X[:, y.flatten() == idx].mean(axis=1, keepdims=True)
            for idx in range(len(self.log_a_priori))
        ]

    def _predict_log_conditional(self, x, class_idx):
        L_inv = self.L_invs[class_idx]
        unbiased_x = x - self.means[class_idx]

        y = L_inv @ unbiased_x

        return np.log(L_inv.diagonal().prod()) - 0.5 * (y**2).sum()
