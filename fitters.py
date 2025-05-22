"""
fitters.py
===========

Functions for fitting each spatial model to data.

I had ChatGPT generate the following docstring:

* :func:`fit_ols`  – plain OLS (no spatial process)
* :func:`fit_sem`  – Spatial **Error** Model   (SEM)
* :func:`fit_sar`  – Spatial **Autoregressive** (lag) Model (SAR)
* :func:`fit_gwr`  – Geographically Weighted Regression (GWR)
* :func:`fit_esf_moran_rmse` – Eigenvector Spatial Filtering (ESF)
* :func:`fit_all` – Convenience wrapper that calls every fitter and
  stores all residual vectors in the input *DataFrame*.

The functions *mutate* the given :pyclass:`pandas.DataFrame` in‑place:

* They add one column  ``resid_<MODEL>``  that contains the n×1 residual
  vector for that model (white‑noise for the *right* specification).
* They tuck the fitted model objects away in  ``df.attrs[...]``  so that
  downstream diagnostics or plotting code can easily retrieve them.

All five estimators share the same *design matrix* – seven binary amenity
dummies – and the same binary **rook‑adjacency** spatial weight matrix *W*.

---------------------------------------------------------------------------
Dependencies
---------------------------------------------------------------------------
`spreg`      – PySAL sub‑package for spatial econometrics
`esda.moran` – Moran’s I statistic for residual whiteness
`mgwr`       – Geographically Weighted Regression
`utilities`  – Local helpers for converting the DataFrame to (y, X),
               building spatial powers Wᵖ, caching ESF eigenvectors, &
               computing PRESS‑type diagnostics.
               
Parts of this module were written or refined with the assistance of ChatGPT. Heads up, I think I 
gave up trying to typehint here because a lot of this was written at 3am.
"""

from __future__ import annotations

import numpy as np
import numpy.linalg as npl
from pandas import DataFrame

from spreg import OLS as spOLS, ML_Error, ML_Lag          # Provides models for OLS, SEM, SAR
from esda.moran import Moran                              # Moran's I
from mgwr.sel_bw import Sel_BW                            # GWR bandwitch
from mgwr.gwr import GWR, GWRResults                      # GWR core classes

# --- project utilities ---------------------------------------------------
from utilities import to_xy, whiten_sem_residual, power_w, cached_esf_basis, loo_rmse_from_qr

# ------------------------------------------------------------------------
# 1.  Ordinary Least Squares 
# ------------------------------------------------------------------------
def fit_ols(df:DataFrame, w):
    """
    Fit a *non‑spatial* OLS model to the input DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a response column ``'y'`` and seven amenity columns
        named ``'x0'`` … ``'x6'``. Modifies in palce
    w : libpysal.weights.W
        Rook‑adjacency weight matrix. Not used here, but helps with making making the modules 
        modular

    Returns
    -------
    spreg.OLS
        The fitted *spreg* OLS object (also stored under
        ``df.attrs['OLS_model']``).
    """
    y_vec, X_mat = to_xy(df)                     # (n,1), (n,7)
    model = spOLS(y_vec, X_mat)                  # spreg auto‑adds intercept!!
    df["resid_OLS"] = model.u                    # residuals
    df.attrs["OLS_model"] = model
    return model


# ------------------------------------------------------------------------
# 2.  Spatial Error Model 
# ------------------------------------------------------------------------
def fit_sem(df, w):
    """
    Fit theSpatial Error Model (SEM)

    Stores two kinds of of residuals:

    ``resid_SEMraw``   – the raw *spreg* residuals *û*  
    ``resid_SEMwhite`` – the same residuals, *pre‑whitened* by solving
                         (I – λ W) û so that a correct SEM shows no
                         remaining spatial autocorrelation (for the the heatmaps)

    Parameters
    ----------
    df : pandas.DataFrame
    w  : libpysal.weights.W
         Binary, row‑standardised rook weight matrix.

    Returns
    -------
    spreg.ML_Error
        Fitted SEM object (also in ``df.attrs['sem_model']``).
    """
    y_vec, X_mat = to_xy(df)
    model = ML_Error(y_vec, X_mat, w=w, name_y="y")         # MLE SEM
    # Raw residuals straight from the SEM
    df["resid_SEMraw"] = model.u
    df["resid_SEMwhite"] = whiten_sem_residual(model, w)
    df.attrs["sem_model"] = model
    return model


# ------------------------------------------------------------------------
# 3.  Spatial Lag (Autoregressive) Model 
# ------------------------------------------------------------------------
def fit_sar(df, w, p: int = 1):
    """
    Fit the Spatial Autoregressive (lag) Model of order p.

    Parameters
    ----------
    df: pandas.DataFrame
    w: libpysal.weights.W
        Binary, row‑standardised rook weight matrix. The p‑th power is taken with
        utilities.power_w
    p: int
        Spatial lag distance.  
        p = 1 -> rook neighbours,
        p = 2 -> two‑step neighbours, etc.

    Returns
    -------
    spreg.ML_Lag
        Fitted SAR model (also in ``df.attrs['sar_model']``).

    """
    y_vec, X_mat = to_xy(df)
    W_p = power_w(w, p)                          # rook contiguity to p‑th order
    model = ML_Lag(y_vec, X_mat, w=W_p, name_y="y")
    df[f"resid_SAR{p}"] = model.u
    df.attrs["sar_model"] = model
    return model


# ------------------------------------------------------------------------
# 4.  Geographically Weighted Regression (GWR)
# ------------------------------------------------------------------------
# Lots of Chat assistance here
def fit_gwr(df):
    """
    Fit Geographically Weighted Regression
    
    Steps
    -----
    1. Build coordinate array (col, row) in the original grid units.
    2. Use :class:`mgwr.sel_bw.Sel_BW` to select the optimal Gaussian
       kernel bandwidth via cross‑validation.
    3. Fit :class:`mgwr.gwr.GWR` with that bandwidth.
    4. Store the per‑observation residuals and the chosen bandwidth.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    (mgwr.GWRResults, float)
        Tuple of the fitted model and the *scalar* optimal bandwidth.

    Side effects
    ------------
    * Adds column ``resid_GWR`` to *df*.
    * Stores bandwidth under ``df.attrs['gwr_bw']`` and the full model
      under ``df.attrs['gwr_model']``.
    """
    coords = df[["col", "row"]].values                   # (n, 2)
    y_vec  = df["y"].values.reshape(-1, 1)               # (n, 1)
    X_mat  = df[[f"x{i}" for i in range(7)]].values      # no intercept

    # ---- bandwidth selection ------------------------------------------------
    bandwidth = Sel_BW(coords, y_vec, X_mat).search(bw_min=2)

    # ---- local regression ---------------------------------------------------
    gw_result = GWR(coords, y_vec, X_mat, bandwidth).fit()
    if not isinstance(gw_result, GWRResults):
        raise TypeError("Expected a GWRResults instance from mgwr")

    # ---- store diagnostics --------------------------------------------------
    df["resid_GWR"] = gw_result.resid_response
    df.attrs["gwr_bw"]    = bandwidth
    df.attrs["gwr_model"] = gw_result
    return gw_result, bandwidth

def fit_esf_moran_rmse(df, w, *,
                       k_max=None,
                       tol_I=0.02,           # “near zero”
                       alpha=0.10):          # or p‑value > 0.10
    """
    Fits Eigenvector Spectral Clusters (Moran's Eigenvector's). Adds eigenvectors until 
    autcorrelation in the residuals are near 0, and optimizes the number eigenvectors producing
    zero autocorrelation for minimal RMSE. Calculating many different models is computationally
    intensive, so I had Chat help me optimize. 
    
    Procedure:
    1.  Walk forward through the cached Moran eigen‑vectors.
    2.  Keep a running QR so each step is O(n·k).
    3.  Whenever residuals are white enough (|I|<tol_I *or* p>alpha),
        store (k, PRESS_RMSE).
    4.  After loop, pick the candidate with the smallest PRESS_RMSE.
        If none satisfies whiteness, fall back to plain OLS.

    The chosen model’s residuals go into df["resid_ESF"].
    k* is recorded in df.attrs["esf_k"], residual I in df.attrs["esf_I"].
    
    By default, will not try to fit more than 20% of total number of eigenvectors (i made this 
    up to avoid overfitting and reduce computational complexity, but it's kind of based on that 
    Bayes 80-20 rule)
    """
    y, X0 = to_xy(df)
    n     = y.shape[0]

    # ---------- dynamic upper bound ----------------------------------
    if k_max is None:
        k_max = max(1, int(0.20 * n))      # never less than 1
    else:
        k_max = min(k_max, n)              # guard against k_max > n

    # ----- get the top‑k_max eigen‑vectors (cached) -------------------
    E = cached_esf_basis(w, k_max)         # (n, k_max)

    # ----- start with plain OLS ---------------------------------------
    Q, R = npl.qr(np.hstack([np.ones((n,1)), X0]), mode='reduced')
    best_press = loo_rmse_from_qr(Q, R, y)
    resid      = y - Q @ (Q.T @ y)
    moran      = Moran(resid.ravel(), w)

    candidates = []                         # list of (k, press, moran.I, Q,R)

    # check whether OLS already white enough
    if abs(moran.I) < tol_I or moran.p_sim > alpha:
        candidates.append((0, best_press, moran.I, Q, R))

    # ----- forward selection ------------------------------------------
    for k in range(1, k_max + 1):
        e_k = E[:, [k-1]]
        Q, R = npl.qr(np.hstack([Q, e_k]), mode='reduced')   # incremental
        press = loo_rmse_from_qr(Q, R, y)
        resid = y - Q @ (Q.T @ y)
        moran = Moran(resid.ravel(), w)

        if abs(moran.I) < tol_I or moran.p_sim > alpha:
            candidates.append((k, press, moran.I, Q.copy(), R.copy()))

    # ----- choose the best PRESS among white models -------------------
    if not candidates:                      # no whitened model found
        k_star, Q_star, R_star = 0, *npl.qr(np.hstack([np.ones((n,1)), X0]), mode='reduced')
        I_star = moran.I
    else:
        k_star, _, I_star, Q_star, R_star = min(candidates, key=lambda t: t[1])

    # ----- final refit to get coefficients & residuals ----------------
    if k_star == 0:
        mdl = fit_ols(df, w)                # reuse the OLS helper
        df["resid_ESF"] = df["resid_OLS"]
    else:
        X_full = np.hstack([X0, E[:, :k_star]])
        mdl    = spOLS(y, X_full)           # spreg adds intercept
        df["resid_ESF"] = mdl.u

    df.attrs.update(esf_k=k_star, esf_I=I_star, esf_model=mdl)
    return mdl


# ------------------------------------------------------------------------
# 6.  Convenience wrapper: “fit every model we understand”
# ------------------------------------------------------------------------
def fit_all(
    df,
    w,
    *,
    p: int = 1,
    k_max_esf: int | None = None,
    tol_I: float = 0.02,
    alpha_I: float = 0.10,
):
    """
    Run all spatial models on the same DataFrame: OLS -> SEM -> SAR(p) -> GWR -> ESF.

    Parameters
    ----------
    df : pandas.DataFrame
    w  : libpysal.weights.W
    p  : int, default 1
        Order of the spatial lag Wᵖ in :func:`fit_sar`.
    k_max_esf : int, optional
        Upper bound for the ESF forward selection.
    tol_I, alpha_I : float
        Whiteness thresholds passed through to :func:`fit_esf_moran_rmse`.
    """
    
    # if statements was an optimization thing i didn't end up using
    if "resid_OLS" not in df:            # plain OLS
        fit_ols(df, w)

    if "resid_SEMwhite" not in df:       # SEM
        fit_sem(df, w)

    if f"resid_SAR{p}" not in df:        # SARᵖ
        fit_sar(df, w, p=p)

    if "resid_GWR" not in df:            # GWR
        fit_gwr(df)

    if "resid_ESF" not in df:            # ESF
        fit_esf_moran_rmse(
            df,
            w,
            k_max=k_max_esf,
            tol_I=tol_I,
            alpha=alpha_I,
        )