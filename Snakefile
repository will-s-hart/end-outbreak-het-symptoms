import glob

from scripts.inputs import (
    COMPARISON_SPECS,
    DATA_PATH,
    MODEL_SPECS,
    get_fig_paths_model_comparison_plots,
    get_results_paths_model_fits,
)

# Globbed rather than listed by hand: every package module is an input to every analysis, and
# an omission here silently stops results being rebuilt when that module changes.
package_paths = sorted(glob.glob("src/endoutbreakhet/*.py"))
shared_input_paths = [*package_paths, "scripts/inputs.py"]


def model_fits_results_paths(*, model_spec_name):
    # The full trace is left out: it's gitignored, so on a fresh clone its absence would force
    # every fit to rerun
    results_paths = get_results_paths_model_fits(model_spec_name=model_spec_name)
    return [path for name, path in results_paths.items() if name != "trace"]


def model_comparison_plots_fig_paths(*, comparison_spec_name):
    fig_paths = get_fig_paths_model_comparison_plots(
        comparison_spec_name=comparison_spec_name
    )
    return [path for paths in fig_paths.values() for path in paths]


# Built before the rules below, since Snakemake resolves a rule's inputs when it is defined
results_paths = [
    model_fits_results_paths(model_spec_name=model_spec_name)
    for model_spec_name in MODEL_SPECS
]
figure_paths = [
    model_comparison_plots_fig_paths(comparison_spec_name=comparison_spec_name)
    for comparison_spec_name in COMPARISON_SPECS
]


rule all:
    input:
        figure_paths,


rule results:
    input:
        results_paths,


# Each fit gets the whole machine (the pipeline parallelises over days itself)
rule model_fits:
    input:
        shared_input_paths,
        "scripts/model_fits.py",
        DATA_PATH,
    output:
        model_fits_results_paths(model_spec_name="{spec}"),
    wildcard_constraints:
        spec="|".join(MODEL_SPECS),
    threads: workflow.cores
    shell:
        "pixi run python -m scripts.model_fits --spec {wildcards.spec}"


# One rule per comparison spec rather than a wildcard, since which figures are made depends
# on the spec
for comparison_spec_name, comparison_spec in COMPARISON_SPECS.items():

    rule:
        name:
            f"model_comparison_plots_{comparison_spec_name}"
        input:
            shared_input_paths,
            "scripts/model_comparison_plots.py",
            DATA_PATH,
            [
                model_fits_results_paths(model_spec_name=model_spec_name)
                for model_spec_name in comparison_spec["model_spec_names"]
            ],
        output:
            model_comparison_plots_fig_paths(comparison_spec_name=comparison_spec_name),
        params:
            spec=comparison_spec_name,
        shell:
            "pixi run python -m scripts.model_comparison_plots --spec {params.spec}"
