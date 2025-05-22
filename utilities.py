
"""utilities.py
================

Parts of this module were written or refined with the assistance of ChatGPT, especially parts of 
    docstrings.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
from libpysal.weights import W, WSP

def to_xy(df):
    """
    Split a housing `DataFrame` into `(y, X)` numpy arrays.

    Intercept not included – `spreg` classes will add a constant
    column automatically when they detect it is missing.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the response `y` and exogenous columns `x0` … `x6`.

    Returns
    -------
    y : ndarray, shape (N, 1)
    X : ndarray, shape (N, 7)
    """
    y = df["y"].values.reshape(-1, 1)
    X = df[[f"x{i}" for i in range(7)]].astype(float).values
    return y, X

def whiten_sem_residual(model, w: W):
    """
    Transform SEM residuals into the white‑noise scale.

    The ML_Error estimator returns the structural residuals
    `û = y – X β̂`.  To check for remaining spatial correlation we
    compute

        ε̂ = (I – λ̂ W) û

    which, under a correctly specified SEM, should be iid.

    Parameters
    ----------
    model : spreg.ML_Error
        Fitted SEM object (must have attributes `.lam` and `.u`).
    w : libpysal.weights.W
        The same weight matrix used for estimation.

    Returns
    -------
    eps_hat : ndarray, shape (N,)
        Whitened residual vector.
    """
    lam_hat = float(model.lam)
    A = sp.eye(w.n, format="csr") - lam_hat * w.sparse
    return (A @ model.u).ravel()

def power_w(w: W, p: int = 1) -> W:
    """Return
    W^p (row‑standardised) without destroying the id order
    """
    if p == 1:
        return w
    wsp = WSP(w).sparse ** p
    out = WSP(wsp, id_order=w.id_order).to_W()
    out.transform = "r"
    return out

def w_subset(parent_w: W, keep: list[int]) -> W:
    """
    Extract the sub‑matrix of `parent_w` induced by the nodes in keep.

    Keeps the original row standardisation.
    """
    id2pos = {node_id: pos for pos, node_id in enumerate(parent_w.id_order)}
    pos = [id2pos[i] for i in keep]
    sub_sparse = parent_w.sparse[pos][:, pos]
    sub_w = WSP(sub_sparse, id_order=keep).to_W()
    sub_w.transform = parent_w.transform
    return sub_w


@lru_cache(maxsize=None)
def cached_esf_basis(w: W, k_max: int):
    """Return the top–k_max Moran eigen‑vectors of the binary layout.

    The costly `eigh` is cached so repeated calls with the same `(w, k)`
    are virtually free.
    """
    # binary, symmetric adjacency
    A = (w.sparse > 0).astype(float)
    A = (A + A.T) / 2.0

    n = A.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    eigval, eigvec = npl.eigh(J @ A.toarray() @ J)     # dense – small n²

    order = np.argsort(eigval)[::-1]
    return eigvec[:, order[:k_max]]



def loo_rmse_from_qr(Q: np.ndarray, R: np.ndarray, y: np.ndarray) -> float:
    """Cheap leave‑one‑out RMSE from the QR of the *full* design matrix."""
    # fitted values: ŷ = Q(Qᵀy)
    y_hat = Q @ (Q.T @ y)
    resid = y - y_hat

    # leverage hᵢ – squared row‑norms of Q
    h = np.sum(Q ** 2, axis=1)
    press = np.mean((resid / (1.0 - h)) ** 2)
    return float(np.sqrt(press))


def _split_beta(beta_vec: np.ndarray, n_slopes: int):
    """Separate intercept and slopes, ignoring spreg’s *extra* parameter."""
    b = np.ravel(beta_vec)
    if len(b) == n_slopes + 2:      # intercept + slopes + ϕ (SEM/SAR)
        b = b[:-1]
    return b[0], b[1:]