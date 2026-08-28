# -*- coding: utf-8 -*-

"""
Continuous Bayesian Network Example
====================================

This example demonstrates the workflow for learning and using a
Bayesian Network from continuous numerical data with bnlearn.

Workflow
--------
1. Load continuous data.
2. Learn the Bayesian Network structure using a Gaussian score.
3. Learn the parameters of the resulting network.
4. Inspect the learned model.
5. Perform probabilistic inference.
6. Generate synthetic data.

Notes
-----
For continuous data, Gaussian structure-learning scores are used:

    loglik-g
    aic-g
    bic-g

The Gaussian score is different from the discrete scores used for
categorical Bayesian Networks.

"""

# %% Libraries
import bnlearn as bn
import pandas as pd
import numpy as np


# %% Generate continuous example data
#
# Create a simple continuous dataset with known dependencies:
#
#     X1 → X2
#     X1 → X3
#     X2 → X3
#
# The variables are continuous numerical measurements.

np.random.seed(42)

n = 100

X1 = np.random.normal(0, 1, n)
X2 = 2.0 * X1 + np.random.normal(0, 0.5, n)
X3 = 1.0 * X1 + 1.5 * X2 + np.random.normal(0, 0.5, n)

df = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
})

print('\n[bnlearn] > Continuous example data:')
print(df.head())

print('\n[bnlearn] > Data types:')
print(df.dtypes)

print('\n[bnlearn] > Shape:')
print(df.shape)


# %% Structure learning
#
# For continuous Gaussian data use a Gaussian score.
#
# BIC-G evaluates candidate DAGs under a Gaussian model.

DAG = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic-g',
    verbose=3,
)

print('\n[bnlearn] > Learned DAG:')
print(DAG['model_edges'])


# %% Plot learned structure

bn.plot(DAG)
# or
bn.plot_graphviz(DAG)


# %% Parameter learning
#
# Learn the parameters of the Bayesian Network from the continuous data.

model = bn.parameter_learning.fit(
    DAG,
    df,
    methodtype='bayes',
    verbose=3,
)

print('\n[bnlearn] > Bayesian Network:')
print(model['model'])


# %% Inspect CPDs
#
# Inspect the learned conditional distributions.

print('\n[bnlearn] > Learned CPDs:')
for cpd in model['model'].get_cpds():
    print(cpd)


# %% Plot Bayesian Network

bn.plot(model)
# or
bn.plot_graphviz(model)


# %% Exact inference
#
# Query the distribution of X3 conditioned on observed values of X1
# and X2.
#
# The exact representation and supported inference behavior depend on
# the Bayesian Network/model implementation used by the installed
# bnlearn/pgmpy versions.

query = bn.inference.fit(
    model,
    variables=['X3'],
    evidence={
        'X1': 0.5,
        'X2': 1.0,
    },
    verbose=3,
)

print('\n[bnlearn] > Inference result:')
print(query)

print('\n[bnlearn] > Inference DataFrame:')
print(query.df)


# %% Query multiple variables

query_joint = bn.inference.fit(
    model,
    variables=['X2', 'X3'],
    evidence={
        'X1': 0.5,
    },
    verbose=3,
)

print('\n[bnlearn] > Joint inference:')
print(query_joint)

print('\n[bnlearn] > Joint inference DataFrame:')
print(query_joint.df)


# %% Synthetic sampling
#
# Generate synthetic observations from the learned Bayesian Network.

samples = bn.sampling(
    model,
    n=1000,
    methodtype='bayes',
    verbose=3,
)

print('\n[bnlearn] > Synthetic samples:')
print(samples.head())

print('\n[bnlearn] > Sample shape:')
print(samples.shape)

print('\n[bnlearn] > Synthetic data statistics:')
print(samples.describe())


# %% Compare original and synthetic distributions

print('\n[bnlearn] > Original data statistics:')
print(df.describe())

print('\n[bnlearn] > Synthetic data statistics:')
print(samples.describe())


# %% Alternative Gaussian score: AIC-G
#
# AIC-G can be used instead of BIC-G.
#
DAG_aic = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='aic-g',
    verbose=3,
)

print('\n[bnlearn] > DAG learned using AIC-G:')
print(DAG_aic['model_edges'])


# %% Alternative Gaussian score: Log-Likelihood-G
#
# The Gaussian log-likelihood can also be used as the structure score.
#
DAG_loglik = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='loglik-g',
    verbose=3,
)

print('\n[bnlearn] > DAG learned using Gaussian log-likelihood:')
print(DAG_loglik['model_edges'])


# %% Summary

print('\n' + '=' * 70)
print('Continuous Bayesian Network example completed successfully.')
print('=' * 70)

print('\nNodes:')
print(list(model['model'].nodes()))

print('\nEdges:')
print(list(model['model'].edges()))

print('\nNumber of CPDs:')
print(len(model['model'].get_cpds()))

print('\nSynthetic data:')
print(samples.shape)