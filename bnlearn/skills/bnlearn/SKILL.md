# bnlearn

## Purpose

Use this skill when working with Bayesian Networks, probabilistic graphical
models, structure learning, parameter learning, inference, prediction,
sampling, and causal discovery using the Python `bnlearn` library.

The goal of this skill is to help an AI agent:

1. Understand the statistical problem.
2. Select an appropriate Bayesian Network approach.
3. Select an appropriate bnlearn algorithm, score, or test.
4. Prepare and validate the data.
5. Execute the analysis using the bnlearn API.
6. Validate the resulting network.
7. Interpret the results without making unsupported causal claims.

---

# 1. Core Concepts

A Bayesian Network (BN) is a directed acyclic graph (DAG) in which:

* Nodes represent random variables.
* Directed edges represent conditional dependencies.
* Each node has an associated conditional probability distribution (CPD).
* The graph encodes conditional independence relationships.

The main Bayesian Network workflow is:

```text
Data
  │
  ▼
Data inspection
  │
  ▼
Structure learning
  │
  ▼
Network structure
  │
  ▼
Parameter learning
  │
  ▼
Bayesian Network
  │
  ├── Inference
  ├── Prediction
  ├── Sampling
  ├── Intervention
  └── Visualization
```

Do not confuse:

```text
association
    ≠
conditional dependence
    ≠
directed edge
    ≠
causal relationship
```

A learned DAG should not automatically be interpreted as a causal graph.

---

# 2. When to Use bnlearn

Use bnlearn when the task involves:

* Learning dependencies between variables.
* Learning a Bayesian Network from observational data.
* Learning a DAG structure.
* Estimating Bayesian Network parameters.
* Performing probabilistic inference.
* Predicting variables using a Bayesian Network.
* Sampling from a Bayesian Network.
* Finding conditional dependencies.
* Testing conditional independence.
* Comparing Bayesian Network structures.
* Discovering potentially causal structures.
* Performing interventions.
* Visualizing Bayesian Networks.

Do not automatically use bnlearn for:

* Ordinary regression.
* Ordinary clustering.
* Dimensionality reduction.
* Classification when a Bayesian Network is not required.
* Time-series forecasting when temporal structure is the primary problem.
* Deep learning problems.

---


# 3. Identify the User's Task

Before selecting a bnlearn function, determine what the user is trying to accomplish.

| User goal                       | bnlearn workflow                |
| ------------------------------- | ------------------------------- |
| Discover relationships          | Structure learning              |
| Find a DAG                      | Structure learning              |
| Test conditional independence   | Independence testing            |
| Estimate probabilities          | Parameter learning              |
| Ask probability questions       | Inference                       |
| Predict a variable              | Prediction                      |
| Generate synthetic observations | Sampling                        |
| Compare networks                | Structure scoring               |
| Ask whether X causes Y          | Causal discovery / intervention |
| Visualize a network             | Network visualization           |

Do not select an algorithm before identifying the problem type.

---

# 4. Standard Workflow

For a new Bayesian Network problem, follow this workflow:

1. Identify the user's goal.
2. Select the appropriate method.
3. Consult the relevant reference documentation.
4. Prefer an existing example when one matches the task.
5. Adapt the example to the user's data.

Now we move to the data set:

1. Inspect the dataset.
2. Identify variable types.
3. Check missing values.
4. Check categorical cardinalities.
5. Inspect continuous-variable distributions.
6. Determine whether preprocessing is required.
7. Determine whether the data are discrete, continuous, or mixed.
8. Select a structure-learning strategy.
9. Select an appropriate score or independence test.
10. Apply domain constraints when available.
11. Learn the network structure.
12. Inspect the learned structure.
13. Validate the structure.
14. Learn the network parameters.
15. Validate the fitted model.
16. Perform inference, prediction, sampling, or intervention.
17. Interpret the results.
18. Clearly distinguish statistical dependency from causal interpretation.


---

# 5. Data Preparation

Before structure learning, inspect:

* Number of observations.
* Number of variables.
* Variable types.
* Missing values.
* Constant variables.
* Duplicate variables.
* Categorical cardinality.
* Continuous-variable distributions.
* Outliers.
* Highly correlated variables.
* Sample size relative to network complexity.

Do not silently modify the dataset.

If preprocessing is required, explain why it is being performed.

---

## 5.1 Discrete Variables

Discrete variables may represent:

* Categories.
* Binary states.
* Ordinal states.
* Discretized measurements.

Check whether categorical variables are represented consistently.

Avoid unnecessarily creating a very large number of states because this can make
parameter estimation unreliable.

---

## 5.2 Continuous Variables

Do not automatically discretize continuous variables.

First determine whether a continuous Bayesian Network is appropriate.

Inspect:

* Distribution shape.
* Outliers.
* Transformations.
* Approximate Gaussianity.
* Sample size.
* Nonlinear relationships.
* Measurement scale.

When the assumptions are appropriate, prefer a continuous/Gaussian Bayesian
Network rather than unnecessarily discarding information through
discretization.

See:

`references/continuous_models.md`

---

## 5.3 Mixed Variables

When variables contain a mixture of discrete and continuous values:

1. Identify the variable types.
2. Determine whether the selected bnlearn method supports the combination.
3. Select a compatible scoring or testing method.
4. Do not silently convert variables without explaining the consequences.

See:

`references/continuous_models.md`

---

# 6. Structure Learning

Structure learning determines the dependency structure of the Bayesian Network.

The main approaches are:

```text
Structure Learning
│
├── Score-based
│   └── Search for a graph that optimizes a score
│
├── Constraint-based
│   └── Infer structure using conditional independence tests
│
└── Hybrid
    └── Combine constraint-based and score-based methods
```

---

## 6.1 Score-Based Learning

Use score-based learning when the goal is to find a graph that optimizes a
statistical score.

Typical approaches include:

* Hill Climbing.
* Other search-based methods supported by bnlearn.

Typical scores include:

* BIC.
* AIC.
* K2.
* BDeu.
* BDs.
* Gaussian scores for continuous variables where supported.

See:

`references/structure_learning.md`

and:

`references/scoring.md`

---

## 6.2 Constraint-Based Learning

Constraint-based methods infer graph structure from conditional independence
relationships.

Typical approach:

* PC algorithm.

Use this approach when conditional independence testing is central to the
analysis.

Consider:

* Variable type.
* Independence test.
* Significance level.
* Sample size.
* Multiple testing.
* Causal assumptions.

See:

`references/structure_learning.md`

---

## 6.3 Hybrid Learning

Hybrid approaches combine constraint-based and score-based learning.

A typical example is:

* MMHC.

Use hybrid methods when a combination of conditional independence discovery
and score-based optimization is appropriate.

See:

`references/structure_learning.md`

---

# 7. Choosing a Structure-Learning Method

Use the following decision logic:

```text
Is the network structure unknown?
│
├── No
│   └── Use the supplied structure and perform parameter learning.
│
└── Yes
    │
    ├── Is conditional independence testing central?
    │      └── Consider PC.
    │
    ├── Is score optimization preferred?
    │      └── Consider Hill Climbing.
    │
    └── Is a hybrid strategy appropriate?
           └── Consider MMHC.
```

Do not choose an algorithm solely because it is the default.

Consider:

* Data type.
* Sample size.
* Number of variables.
* Expected graph complexity.
* Computational budget.
* Causal assumptions.
* Available domain knowledge.

---

# 8. Structure Constraints

When domain knowledge is available, use it to constrain the search space.

Possible constraints include:

* Whitelist edges.
* Blacklist edges.
* Maximum number of parents.
* Forbidden directions.
* Required relationships.
* Temporal ordering.

Constraints can substantially improve both computational efficiency and
interpretability.

Do not invent domain constraints.

Only apply constraints supplied by the user or justified by the problem domain.

---

# 9. Parameter Learning

Parameter learning estimates the probability distributions associated with a
known network structure.

The typical workflow is:

```text
Data
  │
  ▼
Learn or define DAG
  │
  ▼
Parameter learning
  │
  ▼
Fitted Bayesian Network
```

Do not use parameter learning as a substitute for structure learning when the
network structure is unknown.

See:

`references/parameter_learning.md`

---

# 10. Bayesian Inference

Use inference when the user wants to calculate probabilities conditioned on
observed evidence.

Typical problem:

```text
Given:
    X = x

Calculate:
    P(Y | X = x)
```

General workflow:

```text
Fitted Bayesian Network
        │
        ▼
Evidence
        │
        ▼
Inference
        │
        ▼
Posterior distribution
```

Clearly distinguish:

* Prior probability.
* Likelihood.
* Posterior probability.
* Marginal probability.
* Conditional probability.

See:

`references/inference.md`

---

# 11. Prediction

Use prediction when the user wants to estimate an unknown variable from
observed variables.

Before prediction:

1. Verify that the target variable exists.
2. Verify that the model has been fitted.
3. Identify available evidence variables.
4. Determine whether the target is discrete or continuous.
5. Report uncertainty where available.

Do not describe prediction as causal intervention.

---

# 12. Sampling

Use sampling when the user wants to:

* Generate synthetic observations.
* Explore the model distribution.
* Perform simulation.
* Generate conditional samples.

Sampling should use the fitted Bayesian Network rather than independently
sampling each variable.

See:

`references/sampling.md`

---

# 13. Causal Discovery

Causal interpretation requires additional assumptions.

Do not automatically interpret:

```text
X → Y
```

as:

```text
X causes Y
```

Before making a causal claim, consider:

* Causal sufficiency.
* Hidden confounding.
* Measurement quality.
* Sampling assumptions.
* Temporal ordering.
* Faithfulness.
* Markov assumptions.
* Correct specification of the causal graph.
* Intervention semantics.

When assumptions are not established, describe the result as a dependency
structure.

See:

`references/causal_discovery.md`

---

# 14. Intervention

An intervention asks a different question from ordinary inference.

Observational question:

```text
P(Y | X = x)
```

Interventional question:

```text
P(Y | do(X = x))
```

Do not replace an intervention with ordinary conditioning.

When the user asks:

> What happens if I force X to a particular value?

interpret this as an intervention problem.

See:

`references/causal_discovery.md`

---

# 15. Model Scoring

Use structure scores to compare candidate Bayesian Network structures.

Common scores include:

* BIC.
* AIC.
* K2.
* BDeu.
* BDs.
* Gaussian scores for continuous data where supported.

When comparing scores:

* Use compatible data types.
* Consider model complexity.
* Consider the assumptions behind each score.
* Do not compare scores calculated under incompatible conditions without
  explaining the limitation.

See:

`references/scoring.md`

---

# 16. Model Validation

Do not assume that a learned network is correct merely because the algorithm
successfully completed.

Evaluate:

* Structural plausibility.
* Stability.
* Predictive performance where appropriate.
* Parameter estimates.
* Conditional independence relationships.
* Sensitivity to preprocessing.
* Sensitivity to algorithm settings.
* Sensitivity to sample variation.

Where possible, assess whether important edges are stable under resampling.

---

# 17. Visualization

Use visualization to inspect:

* Nodes.
* Directed edges.
* Network structure.
* Parent-child relationships.
* Markov blankets.
* Edge strengths where available.

Visualization is an exploratory and communication tool.

Do not infer statistical significance merely from visual appearance.

---

# 18. Canonical bnlearn Patterns

## Structure learning

```python
import bnlearn as bn

model = bn.structure_learning.fit(
    df,
    method='hc',
    scoretype='bic'
)
```

## Parameter learning

```python
model = bn.parameter_learning.fit(
    model,
    df
)
```

## Inference

```python
query = bn.inference.fit(
    model,
    variables=['target'],
    evidence={'feature': value}
)
```

These examples are canonical patterns only. Always verify the current API and
available arguments before introducing less-common functionality.

---

# 19. Common Mistakes

## Mistake: Treating every edge as causal

Incorrect:

```text
X → Y
therefore X causes Y
```

Correct:

```text
X → Y
is evidence of a learned dependency structure.
```

Causal interpretation requires additional assumptions.

---

## Mistake: Discretizing continuous variables automatically

Do not discretize continuous variables simply because discrete Bayesian
Networks are easier to use.

First consider whether a continuous/Gaussian model is appropriate.

---

## Mistake: Confusing inference and intervention

These are different operations:

```text
P(Y | X = x)
```

versus:

```text
P(Y | do(X = x))
```

---

## Mistake: Ignoring variable types

Do not select a score or independence test without considering whether the
variables are:

* Discrete.
* Continuous.
* Mixed.

---

## Mistake: Ignoring sample size

A network with many parameters can be poorly estimated even when structure
learning succeeds computationally.

Prefer simpler structures when the available data cannot support a highly
complex model.

---

## Mistake: Treating a single learned DAG as ground truth

Structure learning is sensitive to:

* Data.
* Algorithm.
* Score/test.
* Hyperparameters.
* Sampling variation.
* Model assumptions.

Consider stability and uncertainty.

---

# 20. Troubleshooting

## Unexpected network structure

Check:

* Variable types.
* Missing values.
* Score selection.
* Independence test.
* Significance level.
* Search constraints.
* Maximum parent count.
* Sample size.
* Outliers.
* Transformations.
* Algorithm assumptions.

---

## Poor continuous-variable results

Check:

* Gaussian assumptions.
* Outliers.
* Transformations.
* Nonlinear relationships.
* Sample size.
* Score selection.
* Whether discretization is appropriate.

---

## Too many edges

Consider:

* Stronger model-selection criteria.
* Search constraints.
* Maximum parent count.
* Domain knowledge.
* Larger sample size.
* Stability analysis.

Do not arbitrarily delete edges merely to make the graph look simpler.

---

## Too few edges

Check:

* Independence-test threshold.
* Score selection.
* Sample size.
* Variable preprocessing.
* Search constraints.
* Whether the underlying relationships are detectable with the available
  data.

---

# 21. Reproducibility

When applicable:

* Set random seeds.
* Record algorithm settings.
* Record score/test settings.
* Record preprocessing steps.
* Record bnlearn and dependency versions.
* Record structural constraints.
* Record the input variables.
* Save the learned network.

A Bayesian Network result should be reproducible from the documented
configuration and input data.

---

# 22. Performance

For large datasets or many variables:

1. Reduce unnecessary variables.
2. Remove constant variables.
3. Use domain constraints where justified.
4. Limit graph complexity where appropriate.
5. Select computationally appropriate algorithms.
6. Avoid unnecessarily expensive exhaustive searches.
7. Validate the result after optimization.

Do not sacrifice statistical validity solely for speed.

---

# 23. Recommended Response Pattern

When solving a bnlearn problem, structure the response as:

### 1. Problem identification

Explain what Bayesian Network task is being solved.

### 2. Data assessment

Identify:

* Variable types.
* Sample size.
* Missing values.
* Relevant assumptions.

### 3. Method selection

Explain:

* Structure-learning algorithm.
* Score or independence test.
* Constraints.
* Continuous/discrete strategy.

### 4. Implementation

Provide the relevant bnlearn code.

### 5. Validation

Explain how to evaluate the learned model.

### 6. Interpretation

Explain what the network means and explicitly state limitations.

---

# 24. Reference Documentation

Use the following references for detailed guidance:

* `references/structure_learning.md`
* `references/continuous_models.md`
* `references/scoring.md`
* `references/parameter_learning.md`
* `references/inference.md`
* `references/sampling.md`
* `references/causal_discovery.md`
* `references/troubleshooting.md`

Examples:

* `examples/discrete_bn.py`
* `examples/continuous_bn.py`
* `examples/hill_climbing.py`
* `examples/pc_algorithm.py`
* `examples/inference.py`
* `examples/sampling.py`

---

# 25. API Accuracy

The bnlearn API can evolve.

When implementing a solution:

1. Prefer the API exposed by the installed bnlearn version.
2. Do not assume that APIs from other Bayesian Network libraries are interchangeable.
3. Use only functions that exists in bnlearn. Do not invent new function names.
4. Do not invent function arguments.
5. Verify estimator, score, test, and inference arguments before using them.
6. Prefer the current bnlearn documentation and source code over outdated examples.
7. If an API differs between versions, explicitly state the version-specific behavior.

Never silently substitute APIs from pgmpy, pomegranate, DoWhy, CausalNex, or
other Bayesian Network libraries.

---

# 26. Final Decision Checklist

Before returning a bnlearn solution, verify:

* [ ] The Bayesian Network task has been identified.
* [ ] Variable types have been considered.
* [ ] Missing values have been considered.
* [ ] Continuous variables have not been discretized unnecessarily.
* [ ] The structure-learning algorithm is appropriate.
* [ ] The score or independence test is appropriate.
* [ ] Domain constraints have been considered.
* [ ] Sample size has been considered.
* [ ] The learned structure has been validated.
* [ ] Parameter learning is performed when required.
* [ ] Inference and intervention have not been confused.
* [ ] Causal claims are supported by appropriate assumptions.
* [ ] The bnlearn API matches the relevant version.
* [ ] The result is reproducible where appropriate.
* [ ] Limitations are clearly communicated.
