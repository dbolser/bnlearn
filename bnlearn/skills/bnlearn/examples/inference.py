# -*- coding: utf-8 -*-

"""Example: Bayesian Network inference.

This example demonstrates how to perform probabilistic inference on a
Bayesian Network using bnlearn.

Inference can be used to answer questions about the probability of a
variable given observed evidence. For example, given that a patient has
certain symptoms, we can estimate the probability of a disease.

The example creates a simple Bayesian Network, fits the parameters from
synthetic data, and then performs inference using observed evidence.

Example
-------
Run the example from the command line:

    python inference.py
"""

import bnlearn as bn


# =============================================================================
# Create a Bayesian Network
# =============================================================================

# Define the network structure:
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


# Create the Bayesian Network.
model = bn.make_DAG(DAG)


# =============================================================================
# Generate data and fit the parameters
# =============================================================================

# Generate synthetic observations from the network.
df = bn.sampling(model, n=5000)


# Learn the Conditional Probability Distributions (CPDs) from the data.
model = bn.parameter_learning.fit(model, df)


# =============================================================================
# Perform inference
# =============================================================================

# Query the marginal probability of variable D.
query = bn.inference.fit(
    model,
    variables=["D"],
)

print("P(D):")
print(query.df)


# =============================================================================
# Inference with evidence
# =============================================================================

# Estimate P(D | A=1).
query = bn.inference.fit(
    model,
    variables=["D"],
    evidence={"A": 1},
)

print("\nP(D | A=1):")
print(query.df)


# Estimate P(D | A=1, B=1).
query = bn.inference.fit(
    model,
    variables=["D"],
    evidence={
        "A": 1,
        "B": 1,
    },
)

print("\nP(D | A=1, B=1):")
print(query.df)