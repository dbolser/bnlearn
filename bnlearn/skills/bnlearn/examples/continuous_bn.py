# -*- coding: utf-8 -*-
"""
Continuous Bayesian Network Example
====================================

Demonstrates structure learning from continuous numerical data with bnlearn.

Important limitations (bnlearn ≥ 0.14)
--------------------------------------
* Gaussian scores (bic-g, aic-g, loglik-g) and LiNGAM methods learn a DAG.
* bnlearn's parameter_learning.fit / inference.fit / sampling are designed for
  *discrete* Conditional Probability Distributions (TabularCPD).
* Do NOT call parameter_learning.fit(..., methodtype='bayes') on continuous
  data and expect a working continuous probabilistic model.
* For continuous data the practical bnlearn workflow is:

      inspect data → structure learning (bic-g / LiNGAM) → inspect DAG

  Optional: discretize (bn.discretize) if a discrete BN is required downstream.

Workflow in this script
-----------------------
1. Generate continuous data with a known linear structure.
2. Learn structure with Hill Climbing + bic-g.
3. Compare aic-g and loglik-g.
4. Run DirectLiNGAM for causal orientation.
5. Show the discretization path when a discrete BN is needed.
"""

# %% Libraries
import matplotlib
matplotlib.use('Agg')
import bnlearn as bn
import pandas as pd
import numpy as np


# %% Generate continuous example data
#
# True generating process (linear Gaussian):
#
#     X1 → X2
#     X1 → X3
#     X2 → X3

np.random.seed(42)
n = 500

X1 = np.random.normal(0, 1, n)
X2 = 2.0 * X1 + np.random.normal(0, 0.5, n)
X3 = 1.0 * X1 + 1.5 * X2 + np.random.normal(0, 0.5, n)

df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})

print('\n[bnlearn] > Continuous example data:')
print(df.head())
print('\n[bnlearn] > Data types:')
print(df.dtypes)
print('\n[bnlearn] > Shape:', df.shape)
print('\n[bnlearn] > Statistics:')
print(df.describe())


# %% Structure learning with Gaussian BIC
DAG = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic-g',
    verbose=3,
)

print('\n[bnlearn] > Learned DAG (bic-g):')
print(DAG['model_edges'])
print('\n[bnlearn] > Model type:', type(DAG['model']))
print('\n[bnlearn] > Structure scores:', DAG.get('structure_scores'))

# Note: the returned object is a structure (DAG). It does not contain
# continuous CPDs suitable for bn.inference.fit / bn.sampling.


# %% Alternative Gaussian scores
DAG_aic = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='aic-g',
    verbose=0,
)
print('\n[bnlearn] > DAG (aic-g):', DAG_aic['model_edges'])

DAG_loglik = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='loglik-g',
    verbose=0,
)
print('\n[bnlearn] > DAG (loglik-g):', DAG_loglik['model_edges'])


# %% LiNGAM causal discovery (continuous)
# DirectLiNGAM uses non-Gaussian residual assumptions to orient edges.
try:
    DAG_lingam = bn.structure_learning.fit(
        df,
        methodtype='direct-lingam',
        verbose=3,
    )
    print('\n[bnlearn] > DirectLiNGAM edges:')
    print(DAG_lingam.get('model_edges'))
except Exception as exc:
    print('\n[bnlearn] > DirectLiNGAM skipped:', type(exc).__name__, exc)


# %% Optional path: discretize then use a discrete BN
# Only when the user explicitly needs discrete inference/sampling.
# Example (commented — can be slow on large data):
#
# edges = DAG["model_edges"] or [("X1", "X2"), ("X2", "X3")]
# df_disc = bn.discretize(df, edges, continuous_columns=["X1", "X2", "X3"], max_iterations=8)
# DAG_disc = bn.structure_learning.fit(df_disc, methodtype="hc", scoretype="bic")
# model_disc = bn.parameter_learning.fit(DAG_disc, df_disc, methodtype="bayes")

print("\n[bnlearn] > For discrete inference on continuous data, discretize first")
print("            (bn.discretize) then run the discrete BN pipeline.")

# %% Summary
print('\n' + '=' * 70)
print('Continuous Bayesian Network example completed successfully.')
print('=' * 70)
print('Gaussian structure edges (bic-g):', DAG['model_edges'])
print('Remember: continuous structure ≠ discrete parameter learning.')
print('Use discretization only when a discrete BN is required downstream.')
