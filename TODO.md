# bnlearn modernization TODO

Working notes for cleaning up the package, roughly ordered by impact for a
downstream consumer (installs, CI, maintenance). Each item is intended to be a
small, self-contained PR: land on `dbolser/main` (bot reviews) first, then
propose upstream to `erdogant/bnlearn:master`.

## Done

- [x] **CPU-only torch** — pgmpy 0.1.25 imports torch at import time, so the
  default CUDA build pulled ~2 GB of nvidia wheels that are never used. Resolve
  torch from the CPU-only PyTorch index via `[tool.uv.sources]` + committed
  `uv.lock`; CI installs with `--extra-index-url .../whl/cpu`.
  (dbolser#2, upstream erdogant/bnlearn#123)

## In flight (2026-07-28)

- [ ] **`bn.save()` silently fails on macOS temp paths.** pypickle >=2 refuses
  (returns `False`, logs a warning, no exception) to write under hardcoded
  "critical paths" including `/private` and `/var` — where every macOS temp dir
  resolves. Affects master. Surfaced by the characterization test on the macOS
  runners; test now writes under the working tree. Upstream fix TBD: `bn.save`
  should probably raise (or at least warn loudly) instead of returning `False`.
- [x] **lingam 1.13.0 breaks Python 3.13 installs** — pins `scipy<=1.13.1`
  (no cp313 wheels) → from-source scipy build fails without OpenBLAS. Capped
  `lingam<1.13` (dbolser#11; also carried on the migration branch).
- [x] **`_constraintsearch` stale `best_model.edges()` debug print** —
  `NameError` at `verbose>=4` (master `structure_learning.py:582`). Removed on
  the migration branch; upstream still has it.

## High impact — dependency & packaging

- [ ] **Drop dead runtime deps.** Not imported anywhere in `bnlearn/*.py`:
  `funcsigs` (Python 2 `inspect.signature` backport), `fsspec`, `ipywidgets`
  (notebook-only), `statsmodels` (only used by a blog script; transitive of
  lingam anyway). Removing shrinks every install. `pyproject.toml:43,45,49,52`.
- [x] ~~**Fix dynamic version.**~~ **NOT NEEDED — verified 2026-07-13.** The
  audit claimed `version = { attr = "bnlearn.__version__" }` forces setuptools to
  import the full runtime stack (torch/pgmpy) at build time. This is false:
  modern setuptools reads the `__version__` literal via static AST parsing
  (`setuptools.config.expand.read_attr`) without importing the package. Confirmed
  by building a wheel in a torch-free env with `--no-isolation` — it succeeds and
  resolves 0.13.0. No `_version.py` refactor required. (Audit accuracy note: this
  is the third bogus audit finding, after the phantom `numpy==1.26.4` pin and the
  #111 requirements.txt tie-in. Re-verify audit claims before acting on them.)
- [ ] **Kill requirements.txt drift.** It re-declares the whole runtime set and
  pins numpy twice, inconsistently (`numpy==1.26.4` and `numpy>=1.24.1`).
  Have CI `pip install .[dev]` instead; delete or slim requirements.txt.
- [x] **Reconcile the pgmpy version message.** `__init__.py` error text said
  "== 0.1.26" while the pin is `==0.1.25`; message now points at 0.1.25. (dbolser#5)

## Medium impact — correctness & robustness

- [ ] **Replace `print()` with `logging`.** ~170 `verbose`-guarded prints
  (`bnlearn.py` alone has 88). A module logger lets consumers control/silence
  output.
- [x] **Fix bare `except:`** (swallows KeyboardInterrupt/SystemExit):
  `network.py:188,403`, `impute.py:259`, `bnlearn.py:340`. (dbolser#6)
- [x] **Mutable default arg:** `structure_learning.py:671` `Zs=[]`. Also removed
  two dead `return`-after-`raise` lines. (dbolser#6)
- [ ] **Import-time `raise ImportError` gauntlet** in `__init__.py` (four
  version checks). Aggressive; prefer warnings, or rely on pyproject minimums.
- [ ] **Broken `network.py` `plot()`** (line ~187, surfaced by Gemini on
  dbolser#6). Passes drawing kwargs (`labels`, `node_size`, `node_color`,
  `cmap`, `font_size`, `with_labels`) to networkx *layout* functions, which
  reject them → always raises `TypeError` → falls into the except → the fallback
  raises the same way. Also has an `alhpa=alpha` typo and discards the computed
  layout (never draws). This is the legacy `bn.network.plot`, distinct from the
  working public `bn.plot` (`bnlearn.bnlearn.plot`). Needs a real repair or
  removal, not a one-line typo fix. Out of scope for the hygiene PR.
- [ ] **Harden the pgmpy runtime check (optional).** The `< 0.1.18` check in
  `__init__.py` doesn't catch a *too-new* pgmpy (0.1.26+), which is the case the
  pin actually guards against. Gemini suggested `!= "0.1.25"` — but an exact `!=`
  would hard-fail import for anyone on a nearby working version. A warn (not
  raise) on mismatch would be safer. Maintainer's call; separate from the message
  fix in dbolser#5.
- [ ] **Declare optional heavy deps as extras.** `d3blocks` (`bnlearn.py:1421`)
  and `bokeh` (`network.py:295`) are imported inside functions but never
  declared → raw ImportError. Add `[project.optional-dependencies]`.

## Lower impact — testing, typing, hygiene

- [ ] **Offline tests.** `test_bnlearn.py:86` `import_example('titanic')` and
  others download via datazets; no network marker → flaky CI. Bundle a small
  fixture or mark network tests.
- [ ] **Add `py.typed` + type the public API** (`structure_learning.fit`,
  `parameter_learning.fit`, `inference.fit`). Currently untyped.
- [ ] **Reduce shipped dead weight.** `examples.py` (2725 lines) and
  `examples_discretize.py` ship inside the wheel but aren't imported; move to
  `docs/`. Split the 2550-line `bnlearn.py` god-module. Remove commented-out
  exports (`__init__.py`).
- [ ] **Empty `MANIFEST.in`** — delete or populate explicitly.

## Test coverage (measured 2026-07-14, `pytest --cov` over `bnlearn/tests/`)

49 tests pass. Per-module line coverage of the runtime package (ignoring
`examples*.py` and the non-`test_` scripts, which are dead weight at 0% and just
inflate the denominator — reported TOTAL is only 42% because of them):

| Module | Cover | Verdict |
|--------|-------|---------|
| discretize / learn_discrete_bayes_net | 94% | good |
| confmatrix, sampling | 88% | good |
| __init__ | 76% | fine (misses = Windows hotfix + version `raise`s) |
| parameter_learning | 71% | gaps |
| bnlearn.py (god module) | 67% | 253 stmts uncovered |
| structure_learning | 64% | not all methodtypes exercised |
| inference | 63% | some query/predict paths uncovered |
| **impute** | **16%** | **public API essentially untested** |
| **network** | **17%** | **whole `bn.network` module essentially untested** |

Bottom line: the *happy path* of the headline flow (structure → parameter →
inference → sampling) is exercised, but a large slice of the public API is not.
Prioritized gaps to add tests for:

- [ ] **`impute.py` (16%)** — `knn_imputer` and `mice_imputer` are public
  (exported in `__init__`) but almost entirely untested. Add tests for both
  (numeric + mixed string/numeric frames, scaling on/off).
- [ ] **`network.py` (17%)** — `to_graph`, `adjmat2graph`, `compute_centrality`,
  `cluster`, `df2G`, `G2df` are untested. This is also why the broken
  `network.plot()` bug (see above) slipped through. Add tests for the graph
  builders + centrality/cluster; the plot funcs need fixing first.
- [ ] **`bnlearn.py` key untested functions:** `predict()` (~1794-1825),
  `save()`/`load()` model persistence (~1866-1911), and the `import_DAG` BIF-file
  load path (~1657-1685; only the built-in sprinkler DAG is tested, never
  loading an actual `.bif`). `_prune` in `independence_test` (~2026-2061) is also
  uncovered.
- [ ] **`structure_learning.py` (64%)** — audit which `methodtype` values
  (hc/ex/cs/cl/tan/naivebayes/lingam variants) are actually exercised; several
  branches (376-406, 715-745) are not.
- [ ] **Plotting** (`plot`, `plot_graphviz`, `_plot_interactive`, `network.plot`)
  is largely uncovered. At minimum add non-interactive (`Agg`) smoke tests that
  assert a figure/axes is returned, so plot regressions are caught.
- [ ] **Housekeeping:** the two non-`test_` scripts in `tests/`
  (`comparison_and_scripts_in_blog.py`, `learn_discrete_bayes_net.py`) are not
  collected by pytest — either convert to real tests or move out of `tests/`.
