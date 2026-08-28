# -*- coding: utf-8 -*-

"""Example: Sampling from a Bayesian Network.

This example demonstrates how to generate synthetic data from a Bayesian
Network using bnlearn.

Sampling can be used to simulate observations from a fitted Bayesian
Network. This is useful for generating synthetic datasets, validating
models, performing simulations, and understanding the probabilistic
behavior of a network.

Example
-------
Run the example from the command line:

    python sampling.py
"""

import bnlearn as bn


# =============================================================================
# Define a Bayesian Network
# =============================================================================

# Define a simple Bayesian Network:
#
#       A       B
#        \     /
#         \   /
#           C
#           |
#           D
#
DAG = [
    ("A", "C"),
    ("B", "C"),
    ("C", "D"),
]

# =============================================================================
# Create the Bayesian Network
# =============================================================================

# Create a Bayesian Network from the DAG.
model = bn.make_DAG(DAG)


# =============================================================================
# Generate synthetic data
# =============================================================================

# Generate 5000 observations from the Bayesian Network.
df = bn.sampling(
    model,
    n=5000,
)


# Display the first observations.
print(df.head())


# Display the shape of the generated dataset.
print("\nShape:")
print(df.shape)


# Display basic statistics for the generated variables.
print("\nValue counts:")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].value_counts())
    