"""diagnostics.py
=================

Functions for generating the heatmaps from the paper. 

Parts of this module were written or refined with the assistance of ChatGPT. 

"""
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from libpysal.weights import lat2W
from esda.moran import Moran
from sklearn.model_selection import KFold

from fitters import fit_all
from predictors import predict_ols, predict_sem, predict_sar_full, predict_esf, predict_gwr
from utilities import w_subset, cached_esf_basis

# ------------------------------------------------------------
# =====  random k‑fold CV  (over‑optimistic by design)   =====
# ------------------------------------------------------------
def cv_mse_random(df, w, model_name, *, n_splits=5, order=1, seed=0):
    """
    Random k-fold cross validation. 
    
    REFITS models!!! So, we're not just fitting the model and then doing a train-test split and
    then doing cv, which is what it may look like bc we have separate fit routines in here (unless
    i went back and changed that)
    
    """
    y = df["y"].values
    X = df[[f"x{i}" for i in range(7)]].values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    mse = []

    for train, test in kf.split(X):
        sub_df = df.iloc[train].copy()
        sub_w  = w_subset(w, sub_df.index.tolist())
        fit_all(sub_df, sub_w, p=order)       # fits & stores models

        if model_name == "OLS":
            yhat = predict_ols(sub_df.attrs["OLS_model"], X[test])
        elif model_name == "SEM":
            yhat = predict_sem(sub_df.attrs["sem_model"], X[test])
        elif model_name == "SAR":
            X_all = df[[f"x{i}" for i in range(7)]].values
            yhat_full = predict_sar_full(sub_df.attrs["sar_model"],
                                        X_all, w)
            yhat = yhat_full[test]          # slice test rows
        elif model_name == "ESF":
            k   = sub_df.attrs["esf_k"]
            E   = cached_esf_basis(w, k)[:, :k]
            yhat = predict_esf(sub_df.attrs["esf_model"],
                               X[test], E[test])
        else:  # GWR
            yhat = predict_gwr(sub_df.attrs["gwr_model"],
                               X[test],
                               df.loc[test, ["col","row"]].values)
        mse.append(((y[test] - yhat) ** 2).mean())

    return float(np.mean(mse))


# ------------------------------------------------------------
# =====  spatial block CV (leave‑one‑square‑out)  =============
# ------------------------------------------------------------
def cv_mse_block(df, w, model_name, *, block_size=2, order=1):
    
    """
    Leave-one-out block cross validation
    
    REFITS models!!! So, we're not just fitting the model and then doing a train-test split and
    then doing cv, which is what it may look like bc we have separate fit routines in here (unless
    i went back and changed that)
    
    """
    # 1) Build the block labels 
    # --------------------------------
    N      = w.n
    n_grid = int(np.sqrt(N))        # for example, 25 -> 5, 100 -> 10
    bs     = block_size
    n_per  = n_grid // bs           # how many blocks per row/col

    df = df.copy()
    df["block"] = (
        (df["row"] // bs) * n_per   # block-row index
      + (df["col"] // bs)           # block-col index
    )

    y   = df["y"].values
    X   = df[[f"x{i}" for i in range(7)]].values
    mse = []

    for blk in df["block"].unique():
        train_idx = df["block"] != blk
        test_idx  = ~train_idx

        sub_df  = df.loc[train_idx].copy()
        sub_w   = w_subset(w, sub_df.index.tolist())
        fit_all(sub_df, sub_w, p=order)

        if model_name == "OLS":
            m  = sub_df.attrs["OLS_model"]
            yhat = predict_ols(m, X[test_idx])
        elif model_name == "SEM":
            m  = sub_df.attrs["sem_model"]
            yhat = predict_sem(m, X[test_idx])
        elif model_name == "SAR":
            yhat_full = predict_sar_full(sub_df.attrs["sar_model"],
                                         X, w)
            yhat = yhat_full[test_idx]
        elif model_name == "ESF":
            k  = sub_df.attrs["esf_k"]
            E  = cached_esf_basis(w, k)[:, :k]
            m  = sub_df.attrs["esf_model"]
            yhat = predict_esf(m, X[test_idx], E[test_idx])
        else:  # GWR
            m  = sub_df.attrs["gwr_model"]
            yhat = predict_gwr(
                m,
                X[test_idx],
                df.loc[test_idx, ["col", "row"]].values
            )

        mse.append(((y[test_idx] - yhat) ** 2).mean())

    return float(np.mean(mse))


def plot_diag_heatmaps(table,
                       *,
                       cmap="BrBG",
                       fmt=".3f",
                       gens_order=("OLS", "SEM", "SAR", "GWR", "ESF"),
                       fits_order=("ESF", "GWR", "SAR", "SEM", "OLS")):
    """
    Draw three 4×4 heat‑maps (RMSE, MoranI, Param) from the 4×12 diagnostics table.

    Colour rules
    ------------
    * RMSE, and CV's   : per‑column scaling (min‑max of that column)
    * Moran's I : fixed −1 … +1
    """
    stats = ["RMSE", "MoranI", "CV_Rand", "CV_Block"]


    table = table.reindex(columns=gens_order, level=0)      # reorder columns
    fits_rev = list(fits_order)[::-1]                       # bottom‑to‑top

    for stat in stats:
        sub = (table.xs(stat, level=1, axis=1)
                     .reindex(fits_rev))                    # 4×4 slice
        mat = sub.to_numpy(dtype=float)

        # ---- choose scaling strategy ---------------------------------
        if stat == "MoranI":
            img   = mat
            vmin, vmax = -1.0, 1.0                          # fixed global scale
        elif stat in {"RMSE", "CV_Rand", "CV_Block"}:
            cmin  = mat.min(axis=0, keepdims=True)
            cptp  = (mat.max(axis=0, keepdims=True) - mat.min(axis=0, keepdims=True)) + 1e-9
            img   = (mat - cmin) / cptp 
            vmin, vmax = 0.0, 1.0
        else:  # vestigial, was going to have one for rho
            img   = mat
            vmin, vmax = mat.min(), mat.max()              # one global scale
            scale_tag  = "global"

        # ---- plotting -------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        cmap = "OrRd" if stat in {"RMSE", "CV_Rand", "CV_Block"} else "BrBG"
        im  = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        # axis labels
        ax.set_xticks(range(len(gens_order)))
        ax.set_xticklabels(gens_order, rotation=45)
        ax.set_yticks(range(len(fits_rev)))
        ax.set_yticklabels(fits_rev)

        # annotate real numbers (not the 0‑1 normalised ones)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, format(mat[i, j], fmt),
                        ha="center", va="center",
                        color="black",
                        fontsize=8,
                        fontweight="bold")

        if stat == "RMSE":
            title = "Model RMSE"
        elif stat == "MoranI":
            title = "Residual Moran's I"
        elif stat == "CV_Rand":
            title = "Random K-Fold Cross-Validation MSE"
        elif stat == "CV_Block":
            title = "Leave-One-Out Cross-Validation MSE"
        else:
            title = "Something Went Wrong"
        
        # ax.set_title(f"{title}")
        # plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.show()

# ------------------------------------------------------------
# ===    Diagnostics table + heat‑maps  ====================
# ------------------------------------------------------------
def diagnostics_and_heatmaps(datasets, sar_order=1, cmap="inferno",
                             n_splits=5, b_size=2):
    
    """
    Will create a 5x5 table of heatmaps of residuals from each model, but the heatmaps are 
    hard-coded to be smooth/continuous for an easier visual interpretation. Also, it'll make
    tables (though the main file suppresses those tables in print) and turn those tables
    into a single 5x5 heatmap for each metric/statistic: Data x Model Fit; so (SAR, OLS) reports
    Moran's I in the residuals of an OLS model fit to SAR data, for example. 
    
    The colorscale is redetermined for *each column* (each x value) of the heatmap to make 
    comparisons within columns consistent, since we really care about how well each model 
    did within a given dataset
    
    """

    methods = ["OLS", "SEM", "SAR", "GWR", "ESF"]
    res_map = {                     # which column to plot / test
        "OLS": "resid_OLS",
        "SEM": "resid_SEMwhite",
        "SAR": f"resid_SAR{sar_order}",
        "GWR": "resid_GWR",
        "ESF": "resid_ESF"
    }
    
    # --------------------------------------------------------------
    # 0.  infer grid size (n × n) from any DataFrame in `datasets`
    # --------------------------------------------------------------
    sample_df = next(iter(datasets.values()))
    n_grid    = int(sample_df["row"].max() + 1)            # 10 or 25 …

    W = lat2W(n_grid, n_grid, rook=True)                   # ← CHANGE
    W.transform = "r"

    # ---- fit everything once -----------------------------------------
    
    for df in datasets.values():
        fit_all(df, W, p=sar_order)
        
    # --------------------------------------------------------------
    #  report how many eigen‑vectors ESF kept
    # --------------------------------------------------------------
    
    # in the main file, i ovewrite this by sliencing messages! remove the muzzle on
    # stdout to unsilence (but also you're gonna see a lot of messages)
    esf_k_tbl = pd.DataFrame({
        "ESF_k": {name: int(df.attrs.get("esf_k", -1))
                for name, df in datasets.items()}
    }).T 

    print("\n=== ESF: number of eigen‑vectors retained ===")
    print(esf_k_tbl.to_string())
    print()   # blank line for readability

    # ---- numeric diagnostics -----------------------------------------
    
    rows = []
    for dname, df in datasets.items():
        for mname in methods:
            
            r = df[res_map[mname]].values
            
            cv_r = cv_mse_random(df, W, mname, n_splits=n_splits, order=sar_order)
            cv_s = cv_mse_block(df, W, mname, block_size=b_size, order=sar_order)

            rows.extend([
                dict(fit=mname, data=dname, stat="RMSE",   value=np.sqrt((r**2).mean())),
                dict(fit=mname, data=dname, stat="MoranI", value=Moran(r, W).I),
                dict(fit=mname, data=dname, stat="CV_Rand",value=cv_r),
                dict(fit=mname, data=dname, stat="CV_Block",value=cv_s),
            ])

    table = (pd.DataFrame(rows)
               .set_index(["fit", "data", "stat"])["value"]     # <‑‑ keep only the numbers
               .unstack(("data", "stat"))                       # rows = 4 (fits); cols = 4×3
               .round(3))

    print("\n=== 4 × 12 diagnostics table ===")
    print(table)
           
    plot_diag_heatmaps(table)

    # ---- heat‑map panel (row‑wise colour scale) ----------------------
    fig, axes = plt.subplots(5,5, figsize=(8,8))
    for i, fit in enumerate(methods):
        # vmax = max(abs(datasets[d][res_map[fit]]).max() for d in methods) <- for one global 
        #                                                                       colorscale
        col_vmax = {d: max(abs(datasets[d][res_map[f]]).max() for f in methods)
            for d in methods}

        for j, data in enumerate(methods):
            ax  = axes[i,j]
            grid = np.full((n_grid, n_grid), np.nan)
            for _, r in datasets[data].iterrows():
                grid[int(r["row"]), int(r["col"])] = abs(r[res_map[fit]])
            im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=col_vmax[data],
                           interpolation="bilinear") # using bilinear to make smooth/cts heatmaps
            ax.set_xticks([])
            ax.set_yticks([])
            if i==0:
                ax.set_title(f"Data: {data}", fontsize=9)
            if j==0: 
                ax.set_ylabel(f"Fit: {fit}", fontsize=9)
    
    rect: Tuple[float, float, float, float] = (0.92, 0.15, 0.015, 0.7)
    cax = fig.add_axes(rect)
    # cax = fig.add_axes([0.92,0.15,0.015,0.7])
    fig.colorbar(im, cax=cax)
    plt.tight_layout(rect=(0,0,0.9,0.9))
    plt.show()