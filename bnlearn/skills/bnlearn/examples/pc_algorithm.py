# -*- coding: utf-8 -*-

"""Example: PC algorithm for causal discovery.

This example demonstrates how to use the PC algorithm to discover the
structure of a Bayesian Network from observational data.

The PC algorithm is a constraint-based causal discovery method. It uses
conditional independence tests to identify which variables are connected
and then determines the orientation of edges based on conditional
independencies and collider structures.

The resulting graph is a CPDAG (Completed Partially Directed Acyclic Graph),
which represents a Markov equivalence class of DAGs.

Example
-------
Run the example from the command line:

    python pc_algorithm.py

The example generates synthetic data from a known Bayesian Network,
applies the PC algorithm, and visualizes the discovered structure.
"""

import bnlearn as bn


# =============================================================================
# Generate synthetic data
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
# The network is intentionally small so that the discovered structure can
# easily be compared with the ground truth.

DAG = [
    ("A", "C"),
    ("B", "C"),
    ("C", "D"),
]


# Generate synthetic observations from the Bayesian Network.
model = bn.make_DAG(DAG)

df = bn.sampling(model, n=5000)


# =============================================================================
# PC algorithm
# =============================================================================

# Learn the network structure using the PC algorithm.
#
# The PC algorithm is constraint-based and relies on conditional
# independence tests rather than a score function.
model_pc = bn.structure_learning.fit(
    df,
    methodtype="pc",
    scoretype="bic",
)


# =============================================================================
# Visualize the discovered network
# =============================================================================

bn.plot(model_pc)