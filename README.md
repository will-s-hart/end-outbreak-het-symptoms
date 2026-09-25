# end-outbreak-het-symptoms

Python code for analysing the risk of additional cases at the apparent end of an outbreak,
accounting for superspreading (heterogeneity in the number of secondary cases) and for
infectiousness that is linked to symptoms.

This project is at an early stage: the analyses have not been written yet.

## Installation

We recommend using the [pixi](https://pixi.prefix.dev/) package manager. Once pixi is
installed, running

```bash
pixi install --locked
```

creates a local environment (under `.pixi/`) containing the exact dependency versions
recorded in `pixi.lock`. Prefixing a command with `pixi run` executes it in that environment.

## Reproducing the results

Individual analyses are run as modules of the [`scripts`](scripts/) package from the
repository root, e.g.

```bash
pixi run python -m scripts.<name>
```

Alternatively, the full pipeline for all results and figures can be run with
[Snakemake](https://snakemake.readthedocs.io/):

```bash
pixi run snakemake --cores 1 --forceall
```
