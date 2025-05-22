"""data_generating_process.py
================================
Functions to generate data for the architect and neighborhood example.
Every generator starts from the same n x n square lattice
of neighbourhood centroids and seven independent indicator variables, but with spatial 
characteristics well-suited to one or more of OLS, SEM, SAR, GWR, ESF models.

Parts of this module were written or refined with the assistance of ChatGPT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

# ------------------------------------------------------------------------------
#  Foundational utilities
# ------------------------------------------------------------------------------

def build_grid(n: int = 10, *, row_normalise: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    """Create an :math:`n x n` square lattice and its *rook* adjacency matrix.

    A *rook* neighbour shares an edge (north, south, east, west).  Optionally
    the weight matrix is row‑standardised so that each row sums to 1. 

    Parameters
    ----------
        Number of rows (and columns) in the lattice.  
        
        -`n::int` = generates a total of `n x n` blocks.
        - `row_normalise::bool` = If *True*, divide each non‑zero row by its sum.

    Returns
    -------
    grid_df
        ``DataFrame`` with columns ``'id'``, ``'row'``, ``'col'``.
        Index runs from 0 to *n* × *n* – 1.
    W
        ``(N, N)`` NumPy array of *rook* adjacency weights, where
        ``N = n*n``.  Row‑normalised if requested.

    Notes
    -----
    The “id” column follows row‑major order so that the mapping

    ``id = row * n + col``

    matches the construction of *W* we use throughout this package.
    
    """
    node_id, row_coord, col_coord = [], [], []
    for row in range(n):
        for col in range(n):
            node_id.append(row * n + col)
            row_coord.append(row)
            col_coord.append(col)

    grid_df = pd.DataFrame(
        {
            "id":  node_id,
            "row": row_coord,
            "col": col_coord,
        }
    )

    N = n * n
    W = np.zeros((N, N), dtype=float)

    # filling the adjacency matrix using rook adjacency
    for row in range(n):
        for col in range(n):
            idx = row * n + col
            if row > 0:         # north
                W[idx, (row - 1) * n + col] = 1
            if row < n - 1:     # south
                W[idx, (row + 1) * n + col] = 1
            if col > 0:         # west
                W[idx, row * n + (col - 1)] = 1
            if col < n - 1:     # east
                W[idx, row * n + (col + 1)] = 1

    if row_normalise:
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

    return grid_df, W


def gen_X(num_obs: int, *, p: int = 7, p_success: float = 0.30, 
          rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate amenity binaries

    Every element is an independent Bernoulli draw with probability
    ``p_success``. Defaults to 7, because that's the number of predictors i listed in the story
    """
    rng = rng or np.random.default_rng()
    return rng.binomial(1, p_success, size=(num_obs, p))


# ------------------------------------------------------------------------------
#  Data‑generating processes
# ------------------------------------------------------------------------------

def dgp_ols(beta: np.ndarray | None = None, sigma: float = 1.0,
            *, n: int = 10, X: np.ndarray | None = None,
            rng: np.random.Generator | None = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generates OLS directly using a normal linear model.

    Returns ``(data, W)`` where ``data`` already contains the amenity dummies
    and the response.
    """
    rng = rng or np.random.default_rng()
    grid_df, W = build_grid(n)
    N = grid_df.shape[0]

    if X is None:
        X = gen_X(N, rng=rng)

    beta = np.asarray(beta) if beta is not None else np.array(
        [2, 1.5, 1.0, 0.8, 1.2, 0.5, 0.3]
    )

    error = rng.normal(0, sigma, N)
    y = X @ beta + error

    data = grid_df.copy()
    data[[f"x{k}" for k in range(7)]] = X
    data["y"] = y
    return data, W


def dgp_sem(lam: float = 0.5, beta: np.ndarray | None = None, sigma: float = 1.0,
            *, n: int = 10, X: np.ndarray | None = None,
            rng: np.random.Generator | None = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Spatial Error Model data: 
    
    latent omitted factor induces correlated errors.

    Generates data according to

    ``u = (I – λ W)⁻¹ ε`` with ε ∼ 𝒩(0, σ² I)
    ``y = X β + u``
    """
    rng = rng or np.random.default_rng()
    grid_df, W = build_grid(n)
    N = grid_df.shape[0]

    if X is None:
        X = gen_X(N, rng=rng)

    beta = np.asarray(beta) if beta is not None else np.array(
        [2, 1.5, 1.0, 0.8, 1.2, 0.5, 0.3]
    )

    eps = rng.normal(0, sigma, N)
    
    # Solve for spatially‑autocorrelated error term
    u = np.linalg.solve(np.eye(N) - lam * W, eps)
    y = X @ beta + u

    data = grid_df.copy()
    data[[f"x{k}" for k in range(7)]] = X
    data["y"] = y
    return data, W


def dgp_sar(rho: float = 0.4, beta: np.ndarray | None = None, sigma: float = 1.0,
            *, n: int = 10, X: np.ndarray | None = None,
            rng: np.random.Generator | None = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Spatial Autogressive Model data: 
    
    In the story, this is price spill‑overs via spatial lag of *y*.

    Generates according to:

    ``(I – ρ W) y = X β + ε``, ε ∼ 𝒩(0, σ² I)
    """
    
    rng = rng or np.random.default_rng()
    grid_df, W = build_grid(n)
    N = grid_df.shape[0]

    if X is None:
        X = gen_X(N, rng=rng)

    beta = np.asarray(beta) if beta is not None else np.array(
        [2, 1.5, 1.0, 0.8, 1.2, 0.5, 0.3]
    )

    eps = rng.normal(0, sigma, N)
    
    # solving the system of equations for y: y = (I - ρW)⁻¹ (Xβ + ε)
    y = np.linalg.solve(np.eye(N) - rho * W, X @ beta + eps)

    data = grid_df.copy()
    data[[f"x{k}" for k in range(7)]] = X
    data["y"] = y
    return data, W


def dgp_gwr(beta_base: np.ndarray | None = None,
            beta_grad: np.ndarray | None = None,
            sigma: float = 1.0,
            *, n: int = 10, X: np.ndarray | None = None,
            rng: np.random.Generator | None = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Geographically Weighted Regression mode data: 
    
    The rail line in the story

    The coefficient for each amenity changes linearly with the rescaled
    position ``s = (row + col) / (2*(n-1))`` (ranges from 0 to 1).  
    
    So, β(s) is non‑stationary.

    """
    rng = rng or np.random.default_rng()
    grid_df, W = build_grid(n)
    N = grid_df.shape[0]

    if X is None:
        X = gen_X(N, rng=rng)

    beta_base = np.asarray(beta_base) if beta_base is not None else np.ones(7)
    beta_grad = (
        np.asarray(beta_grad)
        if beta_grad is not None
        else np.array([1.5, 0.5, 0.0, -0.5, 1.0, 0.3, -0.2])
    )

    # Spatial scalar s, so the stuff is varying smoothly with space
    # WARNING: NEED TO NOT HARDCODE THE 18
    s = (grid_df["row"].to_numpy() + grid_df["col"].to_numpy()) / (18.0) 
    
    # Local coefficients for every block:  β(s) = β₀ + s · β′
    local_betas = beta_base + np.outer(s, beta_grad)     # shape (N, 7)

    y = np.sum(X * local_betas, axis=1) + rng.normal(0, sigma, N)

    data = grid_df.copy()
    data[[f"x{k}" for k in range(7)]] = X
    data["y"] = y
    return data, W


# -----------------------------
#   dgp_grf : latent Gaussian random field
# -----------------------------
def dgp_grf(beta: np.ndarray | None = None,
            sigma2: float = 1.0, ell: float = 2.5, nugget: float = 0.1,
            *, n: int = 10, X: np.ndarray | None = None,
            rng: np.random.Generator | None = None) -> pd.DataFrame:
    """
    Latent Gaussian Random Field (motivates ESF)

    Draws a zero‑mean Gaussian field z with squared‑exponential kernel

    Cov[z_i, z_j] = σ² exp(−‖x_i − x_j‖² / 2ℓ²) + δ_{ij} · nugget 
    and sets ``y = X β + z``.  
    
    Provides spatially correlated residuals without explicitly specifying a parametric SEM; 
    so, SEMs fit here might not be as good.

    Notes
    -----
    W is not returned here because the GRF is completely determined by its
    kernel – it does not rely on the rook adjacency matrix.
    """
    rng = rng or np.random.default_rng()
    grid_df, _ = build_grid(n)
    N = grid_df.shape[0]

    if X is None:
        X = gen_X(N, rng=rng)

    beta = np.asarray(beta) if beta is not None else np.array(
        [2, 1.5, 1.0, 0.8, 1.2, 0.5, 0.3]
    )

    coords = grid_df[["row", "col"]].values
    sq_dists = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)

    # WARNING: I hard coded this kernel bc it's easier
    # Covariance matrix under the squared‑exponential kernel
    K = sigma2 * np.exp(-sq_dists / (2.0 * ell ** 2)) + np.eye(N) * nugget

    latent_z = rng.multivariate_normal(np.zeros(N), K)
    y = X @ beta + latent_z

    data = grid_df.copy()
    data[[f"x{k}" for k in range(7)]] = X
    data["y"] = y
    return data