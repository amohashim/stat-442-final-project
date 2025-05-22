"""

"""
import sys
import contextlib
import os
import warnings

import numpy as np

from data_generating_process import gen_X, dgp_ols, dgp_sem, dgp_sar, dgp_gwr, dgp_grf
from diagnostics import diagnostics_and_heatmaps

if __name__ == "__main__":

    @contextlib.contextmanager
    def suppress_stdout():
        with open(os.devnull, 'w') as fnull:
            old_stdout = sys.stdout
            sys.stdout = fnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

    warnings.filterwarnings("ignore", message=".*is an island.*|not fully connected")
    warnings.filterwarnings(
        "ignore",
        message="The weights matrix is not fully connected",
        category=UserWarning,
        module="libpysal.weights.weights"
    )
    warnings.filterwarnings(
        "ignore",
        message="Casting complex values to real discards the imaginary part",
        category=np.exceptions.ComplexWarning,
        module="spreg.diagnostics"
    )

    rng = np.random.default_rng(121)
    
    n = 10
    
    X_shared = gen_X(n*n, rng=rng)

    df_ols, _ = dgp_ols(sigma=1.0, n=n, X=X_shared, rng=rng)
    # SEM: stronger spatial correlation, lower white‑noise variance
    df_sem, _ = dgp_sem(lam=0.8, sigma=0.3, n=n, X=X_shared,  rng=rng)

    # SAR: bigger spill‑over
    df_sar, _ = dgp_sar(rho=0.8, sigma=0.3, n=n, X=X_shared, rng=rng)

    # GWR: steeper coefficient gradient
    df_gwr, _ = dgp_gwr(beta_grad=np.array([3,1,0,-1,2,0.6,-0.4]),
                        sigma=0.3, n=n, X=X_shared, rng=rng)
    
    df_esf = dgp_grf(sigma2=1.0, ell=2.5, nugget=0.05, n=n, X=X_shared, rng=rng)
    
    datasets = dict(OLS=df_ols, SEM=df_sem, SAR=df_sar, GWR=df_gwr, ESF=df_esf)
        
    with suppress_stdout():
        diagnostics_and_heatmaps(datasets, sar_order=1, b_size=25, n_splits=10)