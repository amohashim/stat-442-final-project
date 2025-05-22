"""predictors.py
=================

Functions for predicting from spatial models. 

Nothing here *fits* a model – that is done in ``fitters.py`` – we merely
re‑compute the model equation out‑of‑sample so that cross‑validation in
``diagnostics.py`` can be fast and library‑agnostic.

Notation
--------
N  : number of neighbourhoods (always 100)  
k  : number of amenity covariates (k = 7)  
W  : N×N row‑standardised rook‑adjacency matrix  
ρ,λ: spatial autoregressive / error coefficients  
β0 : intercept; β : (k,) slope vector

Parts of this module were written or refined with the assistance of ChatGPT, especially parts of 
    docstrings.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from utilities import _split_beta   # helper reused in fitters.py

# ---------------------------------------------------------------------
# Ordinary Least Squares
# ---------------------------------------------------------------------

def predict_ols(model, X: np.ndarray) -> np.ndarray:
    """
    Return OLS fitted values for some X matrix

    Parameters
    ----------
    model : spreg.OLS
        Fitted regression whose attribute ``betas`` has shape (k+1,).
    X : ndarray, shape (N, k)
        coavariate binaries, NO INTERCEPT.

    Returns
    -------
    y_hat : ndarray, shape (N,)
        Predicted price for every row in *X*.

    Notes
    -----
    The intercept is prepended manually so we never rely on how the
    upstream library stored the design matrix (this was a big problem).
    """

    def _linear_predict(Xm: np.ndarray, beta_vec: np.ndarray) -> np.ndarray:
        """Internal utility: compute [1, X] · β with minimal copies."""
        beta = np.ravel(beta_vec)                          # ensure (k+1,)
        X_const = np.hstack([np.ones((Xm.shape[0], 1)), Xm])
        return (X_const @ beta).ravel()                    # (N,)

    return _linear_predict(X, model.betas)


# ---------------------------------------------------------------------
# Spatial Error Model  (SEM)
# ---------------------------------------------------------------------

def predict_sem(model, X: np.ndarray) -> np.ndarray:
    """
    Mean prediction for a fitted SEM

    The SEM correlation* structure lives in the errors, so the
    conditional mean is identical to OLS. So we don't have to change much from OLS.
    """
    intercept, slopes = _split_beta(model.betas, X.shape[1])
    return intercept + X @ slopes


# ---------------------------------------------------------------------
# Spatial Lag / SAR  (requires full lattice)
# ---------------------------------------------------------------------

def predict_sar_full(model, X_all: np.ndarray, w) -> np.ndarray:
    """
    Had a lot of trouble with this one, ChatGPT helped a lot here.
    
    Return ŷ for *every* neighbourhood in the original lattice.

    The SAR model solves

        (I – ρW) y  =  β₀ + Xβ

    so   ŷ  =  (I – ρW)⁻¹ · (β₀ + Xβ).

    Parameters
    ----------
    model  : spreg.ML_Lag
        Object with attributes ``rho`` and ``betas``.
    X_all  : ndarray, shape (N, k)
        Amenity matrix for **all** N rows, even if the caller will
        eventually slice a subset.
    w      : libpysal.weights.W
        Same *W* that was used during fitting (row‑standardised).

    Returns
    -------
    y_hat_full : ndarray, shape (N,)
    """

    # ---------- unpack β₀, β (dropping any nuisance parameter) ----------
    beta_vec = np.ravel(model.betas)
    if len(beta_vec) == X_all.shape[1] + 2:      # intercept + k + (σ²)
        beta_vec = beta_vec[:-1]                 # drop trailing variance
    β0, β = beta_vec[0], beta_vec[1:]

    rho = float(getattr(model, "rho", 0.0))

    # Right‑hand side: η = β₀ + Xβ
    eta = β0 + X_all @ β                         # (N,)

    # Solve (I – ρW) ŷ = η          using a sparse linear solver
    A = sp.eye(w.n) - rho * w.sparse            # (N,N)
    return np.asarray(spsolve(A, eta))               # (N,)


# ---------------------------------------------------------------------
# Eigenvector Spatial Filtering
# ---------------------------------------------------------------------

def predict_esf(model, X: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Prediction for ESF: 
    
    add the eigenvectors to X with selected spatial eigenvectors

    Parameters
    ----------
    model : sklearn‑like estimator
        Any object exposing ``betas`` *or* ``params``.
    X     : ndarray, shape (N, k)
        Amenity matrix.
    E     : ndarray, shape (N, m)
        Matrix of m selected Moran eigenvectors used during fitting.

    Returns
    -------
    y_hat : ndarray, shape (N,)
    """
    # Robustly grab the coefficient vector irrespective of the class
    beta = np.ravel(getattr(model, "params", model.betas))

    # Design matrix with intercept and eigenvectors
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X, E])
    if len(beta) == X_aug.shape[1] + 1:
        beta = beta[:-1]

    return (X_aug @ beta).ravel()


# ---------------------------------------------------------------------
# Geographically Weighted Regression
# ---------------------------------------------------------------------

def predict_gwr(model, X: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Return local predictions from a fitted mgwr model.

    Parameters
    ----------
    model   : mgwr.sel_bw.GWRResults or mgwr.gwr.GWRResults
    X       : ndarray, shape (N, k)
        Covariate matrix *without* intercept.
    coords  : ndarray, shape (N, 2)
        Centroid coordinates (x, y) for each observation.

    Returns
    -------
    y_hat : ndarray, shape (N,)
    """
    
    if hasattr(model, "predict"): # had some trouble with different versions of python; newer one
                                  # should have this
        return model.predict(coords, X).predictions.ravel()

    return model.model.predict(
        coords,
        X,
        fit_params={"ini_params": model.params}
    ).predictions.ravel()
