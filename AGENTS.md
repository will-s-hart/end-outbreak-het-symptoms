# CLAUDE.md

Project conventions for Claude Code in this repo. Personal preferences live in the gitignored
`CLAUDE.local.md`.

## Project

Analysis of the risk of additional cases at the apparent end of an outbreak, accounting for
superspreading and symptom-linked infectiousness. A library package drives a Snakemake workflow
that produces the numerical results and figures.

**Status:** the library (model fitting, risk, evidence, the rolling-fit pipeline and
plotting), the `delay_distributions` and `model_fits` analysis scripts (the first discretises
the delay distributions the fits read) and the `delay_distributions_plots` and
`model_comparison_plots` plot scripts exist, each with Snakemake rules. Model fitting, risk,
evidence and the pipeline support under-reporting with a known reporting probability
(`reporting_prob`, set per model spec, with 60% and 80% reporting specs alongside the
complete-reporting ones). Risk covers additional cases, transmission, sustained transmission
and reported cases (DLO: cases and transmission only), and the pipeline records every risk
type the model has. Keep this file in step as more land: describe what exists, not what is
planned.

## Layout

- `src/endoutbreakhet/` — library package (src layout, installed editable by pixi, so edits
  take effect immediately). `__init__.__all__` is the public API, except `plotting`, which
  scripts import as `endoutbreakhet.plotting` so that importing the package (as every fit
  worker does) doesn't pull in matplotlib. Modules lead with their public functions; private
  `_helpers` sit below them in depth-first order of first use, with a family of per-model
  functions kept together and followed by the helpers it shares. Keep imports between
  modules one-way, with no cycles.
- `scripts/` — analysis and plot scripts plus shared `inputs.py` (see below) and
  `snakemake_utils.py` (`as_param`, for the Snakefile). A package at the repo root, **not**
  under `src/`.
- `tests/` — top-level pytest suite (not nested under the package).
- `Snakefile` — orchestrates the full results-and-figures pipeline.
- `data/` — inputs. `results/`, `figures/` — generated outputs, committed, except full MCMC
  traces (`results/**/trace.nc`, gitignored); what plots need from a trace is saved alongside
  it as a tracked CSV.

## Keyword-only arguments

Every function the repo defines (package, scripts, tests, private and nested helpers) makes
all its parameters keyword-only, `def f(*, a, b=1)` (after `self`/`cls` for methods), and every
call to one passes keyword arguments. Call sites then stay correct when parameters are added,
removed or reordered.

- Exception: signatures dictated by a third-party API, such as callbacks (a
  `scipy.integrate.quad` integrand takes its variable positionally) or Protocols mirroring
  scipy methods. Keep only the parameters the API passes positionally; make the rest
  keyword-only and bind them by keyword (e.g. `functools.partial(f, x=x)`).
- Calls into third-party libraries follow those libraries' usual style (`np.zeros(n)`,
  `pm.Normal("name", ...)`).

## Running scripts

Scripts are run **as modules from the repo root**: `pixi run python -m scripts.<name>`. Never
`python scripts/<name>.py`: with the src layout, the editable install puts only `src/` on
`sys.path`, so `from scripts.inputs import ...` fails when a script is run by path. pytest gets
the root via `pythonpath = ["."]` in `pyproject.toml`.

## Snakemake workflow

The Snakefile is the single source of truth for reproduction. **Don't add ad-hoc reproduction
commands** — wire them in as rules.

- `all` (default) builds every figure; `results` builds every results file only.
- Each analysis script gets a rule running `pixi run python -m scripts.<name>`, with a wildcard
  for each parameter it takes (e.g. `--spec {spec}`, one job per value). Each plot script gets a
  rule running `pixi run python -m scripts.<topic>_plots`, whose inputs are the results files
  it reads.
- When the set of outputs depends on the parameter value (as `model_comparison_plots`' figures
  do), a wildcard rule can't declare them, so define one rule per value in a `for` loop
  instead (`name:` per value, value passed via `params:`).
- Rule inputs and outputs come from the path getters imported from `scripts/inputs.py`
  (calling one with `"{spec}"` gives a wildcard pattern). Don't declare gitignored files such as
  `trace.nc` as outputs: on a fresh clone their absence would force every job to rerun.
- Fit rules take `threads: lambda wildcards: workflow.cores`, since the pipeline parallelises
  over all cores itself (`n_jobs=-1`). It's a callable because `workflow.cores` raises when
  `--cores` is unset, as under a cluster executor, where `--set-threads` gives the width.
- Inputs are chosen so a rule reruns when, and only when, something it consumes changes:
  - **Analysis rules** list `analysis_package_paths` (the package modules, globbed, minus
    `plotting.py`, which analysis scripts never import; `tests/test_imports.py` checks), their
    script, data files and upstream results. In place of `scripts/inputs.py` they take
    `params: inputs=as_param(inputs=get_inputs_<name>(...))` (in a `lambda wildcards:` when
    the call needs a wildcard). `as_param` converts the inputs to builtins, which Snakemake
    records by `repr`, so editing `inputs.py` reruns only the analyses whose values change. It
    raises `TypeError` on a type it can't convert: extend it rather than leave a value
    untracked.
  - **Plot rules** list `shared_input_paths` (all package modules and `scripts/inputs.py`),
    their script, data files and the results they read. They're cheap, so they rerun on any
    edit.
- Snakemake always compares mtimes (leaving `mtime` out of `--rerun-triggers` changes
  nothing), but an input newer than an output still counts as unchanged if its checksum
  matches the one recorded in the output's metadata (inputs under 1 MB). So a touch or a
  checkout reruns nothing; a content change does.
- Rules with seconds of compute, such as `delay_distributions`, go in `localrules`, so on a
  cluster they don't queue for a slot.
- Add each analysis's output paths to the `results_paths` / `figure_paths` aggregates, which
  are built before `rule all` (Snakemake resolves a rule's inputs when it is defined).

## Analysis vs plotting convention

Compute and plotting are separate scripts, not fixed pairs, since a figure can combine results
from several analyses or several runs of one (e.g. overlapping sets of model specs).

- **Analysis scripts** `scripts/<name>.py` do the heavy compute and write results. One may take
  parameters (e.g. `--spec`), each value run as its own job.
- **Plot scripts** `scripts/<topic>_plots.py` only read results (never fit), and make figures.
- **All input/parameter assembly lives in `scripts/inputs.py`** as `get_inputs_<name>(...)`, one
  per script, returning a dict that includes its `results_paths` and/or `fig_paths` and the
  prepared arrays/parameters. Those paths come from public `get_results_paths_<name>(...)` /
  `get_fig_paths_<name>(...)` getters, which the Snakefile also imports, as it does data paths
  (e.g. `DATA_PATH`). Never duplicate path construction.
- Upstream results reach a script as paths in its inputs (e.g.
  `delay_distributions_results_paths`), which the script reads, never as values loaded in
  `inputs.py`: params are evaluated when the DAG is built, before upstream jobs run, so a
  loaded value would be stale or missing.
- Outputs follow `results/<name>/[<value>/]<output>.csv` and
  `figures/<topic>/[<value>/]<output>.<ext>`, with each figure saved in every format in
  `FIG_EXTENSIONS` (in `inputs.py`), so `fig_paths` maps each figure to a list of paths.
  Snakemake `output:` lists assume this.
- Each script has a single public entry point (`run_analyses` / `make_plots`) at the top; every
  other function is `_private` and sits below it. A helper that is a general reusable task (not
  one analysis's step) and is used across scripts belongs in the package instead. Don't factor
  a one-liner into a helper just because it repeats.
- If paper figures are compiled from per-analysis panels, figure numbers live only in that
  compile script and the Snakefile. Don't put figure or panel numbers in other scripts'
  docstrings or comments; they go stale on any renumbering.

To add an analysis or plot script: write it, add its `get_inputs_<name>()` and path getter to
`inputs.py`, and add its rule to the Snakefile (for an analysis, with the `as_param` param).

## Tests

- `pixi run test` (all tests); `pixi run pytest tests/test_x.py -k name` (a single test).
- `tests/test_public_api.py` asserts the exact set of names in `endoutbreakhet.__all__`. **If
  the exports change, update the package `__init__` and this test in the same change.**
- Test files written with AI assistance start with the line
  `# Note that AI tools were used to generate tests`. Keep or add it when generating or
  substantially modifying a test file.
- Tests can and should be edited to enable code improvements. Don't preserve a test, or contort
  the code, just to avoid touching it; keep the suite sensible and green.

## Commands

| Task | Command |
|------|---------|
| Tests | `pixi run test` |
| Format | `pixi run format` (ruff import sort + format) |
| Lint | `pixi run lint` (ruff check) |
| Type check | `pixi run typecheck` (ty check) |
| Full reproduction | `pixi run snakemake --cores N` |
| Results only | `pixi run snakemake --cores N results` |

Run format, lint and typecheck after code changes.

## Tooling

- The only deliberate ruff config is import sorting (`extend-select = ["I"]`); the `format` task
  runs `ruff check --select I --fix && ruff format`. Otherwise ruff and ty use their defaults on
  purpose, so don't add other config.
- New dependencies go in `pyproject.toml` (conda-forge first, `pypi-dependencies` only if
  needed; dev-only tools in the `dev` feature). Commit the updated `pixi.lock`.
- matplotlib SVG output is non-deterministic (run-varying IDs, float noise), but PNG output is
  byte-identical across runs. So after rerunning a plot script, `git status figures/` shows
  whether a plotting change altered a figure via its PNG; look at the PNG to check the
  change.

## Repo

- GitHub: `will-s-hart/end-outbreak-het-symptoms` (private). Main branch is `main`.
