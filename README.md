# Spatial Regression Simulation Replication Package

A Python package for simulating spatial data under various models (OLS, SEM, SAR, GWR, ESF), fitting those models, and producing diagnostic heatmaps and cross-validation summaries. 
I built this package for the final project in my Statistics senior seminar, STAT 442: Statistical Learning.

Leave-One-Out cross-validation function might be broken: It produces identical estimates with k-fold random cross validation. It might also be that some other part of the workflow 
accidentally leads to these identical results. 

---

## Features

- **Data–Generating Processes**  
  • `dgp_ols`, `dgp_sem`, `dgp_sar`, `dgp_gwr`, `dgp_grf`  
  • Simulate areal data on an _n × n_ rook‐adjacency grid with user-specified parameters.  

- **Model Fitters**  
  • `fit_ols`, `fit_sem`, `fit_sar`, `fit_gwr`, `fit_esf_moran_rmse`  
  • Wraps PySAL’s `spreg`, MGWR, ESF, and GWR routines, storing residuals and model objects.  

- **Predictors**  
  • `predict_ols`, `predict_sem`, `predict_sar_full`, `predict_esf`, `predict_gwr`  
  • Prediction helpers for fast cross-validation.  

- **Diagnostics & Visualization**  
  • `cv_mse_random`, `cv_mse_block`, `plot_diag_heatmaps`  
  • Compute RMSE, Moran’s I, random/block cross-validation; render 4×4 heatmaps of performance.  

- **Utilities**  
  • Grid building, weight‐matrix powers/subsets, ESF‐basis caching, PRESS diagnostics.

---

## Installation

```bash
pip install numpy pandas scipy matplotlib scikit-learn libpysal spreg esda mgwr
```

Clone this repo:

```
git clone https://github.com/amohashim/stat‐442‐final-project.git
cd stat-442-final-project
```

## Using this Package

I highly reccommend using the the main.jypnb file to get replication results. There are two code blocks, one that conducts the simulation and plots 
diagnostic plots for the residuals, and one that plots the response variables themselves. You can also get fun extra output like a table with 
the number of eigevectors chosen for each ESF run if you choose not suppress warnings. However, you will also get many warnings.














