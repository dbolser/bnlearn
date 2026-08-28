# -*- coding: utf-8 -*-

"""
Hill Climbing Structure Learning Example
=========================================

This example demonstrates score-based Bayesian Network structure learning
using the Hill Climb Search algorithm in bnlearn.

Workflow
--------
1. Load a discrete example dataset.
2. Learn a DAG using Hill Climbing.
3. Inspect the learned structure.
4. Compare different scoring methods.
5. Constrain the search using max_indegree.
6. Use a starting DAG and structural constraints.

Hill Climbing
-------------
Hill Climbing is a score-based structure-learning algorithm.

Starting from an initial DAG, the algorithm evaluates neighboring DAGs
created by graph operations such as:

    - adding an edge
    - removing an edge
    - reversing an edge

The search continues while an operation improves the selected score.

For discrete data, common scores include:

    bic
    k2
    bdeu
    bds
    aic

For continuous Gaussian data, use:

    bic-g
    aic-g
    loglik-g
"""

# %% Libraries
import bnlearn as bn


# %% Load example dataset
#
# The sprinkler dataset contains discrete variables:
#
#     Cloudy
#     Sprinkler
#     Rain
#     Wet_Grass

df = bn.import_example('sprinkler')

print('\n[bnlearn] > Example data:')
print(df.head())

print('\n[bnlearn] > Shape:')
print(df.shape)

print('\n[bnlearn] > Data types:')
print(df.dtypes)


# %% Basic Hill Climbing
#
# Learn the structure using Hill Climbing and BIC.

DAG = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    verbose=3,
)

print('\n[bnlearn] > Hill Climbing DAG:')
print(DAG['model_edges'])


# %% Plot the learned DAG

bn.plot(DAG)
# or
bn.plot_graphviz(DAG)


# %% Inspect adjacency matrix

print('\n[bnlearn] > Adjacency matrix:')
print(DAG['adjmat'])


# %% Inspect structure score
#
# The structure score can be useful when comparing alternative models.

if 'structure_scores' in DAG:
    print('\n[bnlearn] > Structure scores:')
    print(DAG['structure_scores'])


# %% Hill Climbing with a larger tabu list
#
# Tabu search can help prevent the algorithm from immediately revisiting
# recently explored structures.

DAG_tabu = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    tabu_length=10,
    verbose=3,
)

print('\n[bnlearn] > Hill Climbing with tabu search:')
print(DAG_tabu['model_edges'])


# %% Limit the maximum number of parents
#
# max_indegree limits the maximum number of incoming edges for a node.
#
# This is useful when a very dense network is undesirable or when domain
# knowledge suggests that a node should have only a limited number of
# direct parents.

DAG_indegree = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    max_indegree=2,
    verbose=3,
)

print('\n[bnlearn] > Hill Climbing with max_indegree=2:')
print(DAG_indegree['model_edges'])


# %% Start from an existing DAG
#
# A starting DAG can be supplied when domain knowledge provides a useful
# initial structure.

start_dag = bn.make_DAG(
    [
        ('Cloudy', 'Sprinkler'),
        ('Cloudy', 'Rain'),
    ],
    verbose=0,
)

DAG_start = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    start_dag=start_dag,
    verbose=3,
)

print('\n[bnlearn] > Hill Climbing from start DAG:')
print(DAG_start['model_edges'])


# %% Required edges
#
# fixed_edges can be used when an edge must be present in the resulting
# structure.

DAG_fixed = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    fixed_edges=[
        ('Cloudy', 'Rain'),
    ],
    verbose=3,
)

print('\n[bnlearn] > Hill Climbing with fixed edge:')
print(DAG_fixed['model_edges'])


# %% White list
#
# A white list restricts the allowed relationships.
#
# This is different from fixed_edges:
#
#     fixed_edges → edge must be present
#     white_list  → edge is allowed during the search

DAG_white = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    white_list=[
        ('Cloudy', 'Sprinkler'),
        ('Cloudy', 'Rain'),
        ('Sprinkler', 'Wet_Grass'),
        ('Rain', 'Wet_Grass'),
    ],
    verbose=3,
)

print('\n[bnlearn] > Hill Climbing with white list:')
print(DAG_white['model_edges'])


# %% Black list
#
# A black list prevents specified relationships from being considered.

DAG_black = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    black_list=[
        ('Wet_Grass', 'Cloudy'),
    ],
    verbose=3,
)

print('\n[bnlearn] > Hill Climbing with black list:')
print(DAG_black['model_edges'])


# %% Compare discrete scoring methods
#
# Different scores can prefer different structures.
#
# BIC, AIC, K2, BDeu, and BDs are applicable to discrete Bayesian
# Network structure learning.

scoretypes = [
    'bic',
    'aic',
    'k2',
    'bdeu',
    'bds',
]

results = {}

for scoretype in scoretypes:

    print(f'\n[bnlearn] > Running Hill Climbing with score: {scoretype}')

    result = bn.structure_learning.fit(
        df,
        methodtype='hc',
        scoretype=scoretype,
        verbose=0,
    )

    results[scoretype] = result

    print(f'{scoretype}:')
    print(result['model_edges'])


# %% Compare learned structures
#
# Print all discovered edges so that the effect of the scoring method
# can be inspected.

print('\n[bnlearn] > Structure comparison:')

for scoretype, result in results.items():
    print(f'\nScore: {scoretype}')
    print(result['model_edges'])


# %% Parameter learning
#
# Once a DAG has been selected, learn its conditional probability
# distributions from the data.

model = bn.parameter_learning.fit(
    DAG,
    df,
    methodtype='bayes',
    verbose=3,
)

print('\n[bnlearn] > Parameterized Bayesian Network:')
print(model['model'])


# %% Inspect CPDs

print('\n[bnlearn] > Learned CPDs:')

for cpd in model['model'].get_cpds():
    print(cpd)


# %% Inference
#
# Use the learned Bayesian Network to calculate:
#
#     P(Wet_Grass | Rain=1)

query = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    evidence={
        'Rain': 1,
    },
    verbose=3,
)

print('\n[bnlearn] > Inference result:')
print(query.df)


# %% Sampling
#
# Generate synthetic observations from the learned network.

samples = bn.sampling(
    model,
    n=1000,
    methodtype='bayes',
    verbose=3,
)

print('\n[bnlearn] > Synthetic samples:')
print(samples.head())

print('\n[bnlearn] > Synthetic sample shape:')
print(samples.shape)


# %% Final summary

print('\n' + '=' * 70)
print('Hill Climbing example completed successfully.')
print('=' * 70)

print('\nNodes:')
print(list(DAG['model'].nodes()))

print('\nEdges:')
print(list(DAG['model'].edges()))

print('\nNumber of edges:')
print(len(DAG['model'].edges()))

print('\nNumber of CPDs:')
print(len(model['model'].get_cpds()))

print('\nAvailable score comparisons:')
for scoretype, result in results.items():
    print(f'  {scoretype}: {len(result["model_edges"])} edges')