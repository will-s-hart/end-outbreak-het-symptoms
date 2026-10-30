import glob

from scripts.inputs import (
    COMPARISON_SPECS,
    DATA_PATH,
    MODEL_SPECS,
    get_fig_paths_delay_distributions_plots,
    get_fig_paths_fit_diagnostics_plots,
    get_fig_paths_model_comparison_plots,
    get_inputs_delay_distributions,
    get_inputs_model_fits,
    get_results_paths_delay_distributions,
    get_results_paths_model_fits,
)
from scripts.snakemake_utils import as_param

# Globbed rather than listed by hand: an omission here silently stops results being rebuilt
# when that module changes.
package_paths = sorted(glob.glob("src/endoutbreakhet/*.py"))
# Analysis rules leave out the plotting module, which their scripts never import
# (tests/test_imports.py checks), and inputs.py, whose values they track through their params
# instead. So editing a figure or a spec reruns only what it changes.
analysis_package_paths = [
    path for path in package_paths if path != "src/endoutbreakhet/plotting.py"
]
shared_input_paths = [*package_paths, "scripts/inputs.py"]

delay_distributions_results_paths = list(
    get_results_paths_delay_distributions().values()
)
delay_distributions_plots_fig_paths = [
    path
    for paths in get_fig_paths_delay_distributions_plots().values()
    for path in paths
]

fit_diagnostics_plots_fig_paths = [
    path
    for paths in get_fig_paths_fit_diagnostics_plots().values()
    for path in paths
]


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
    delay_distributions_results_paths,
    *(
        model_fits_results_paths(model_spec_name=model_spec_name)
        for model_spec_name in MODEL_SPECS
    ),
]
figure_paths = [
    delay_distributions_plots_fig_paths,
    fit_diagnostics_plots_fig_paths,
    *(
        model_comparison_plots_fig_paths(comparison_spec_name=comparison_spec_name)
        for comparison_spec_name in COMPARISON_SPECS
    ),
]


rule all:
    input:
        figure_paths,


rule results:
    input:
        results_paths,


# Seconds of compute, which shouldn't queue for a cluster slot ahead of every fit
localrules:
    delay_distributions,


rule delay_distributions:
    input:
        analysis_package_paths,
        "scripts/delay_distributions.py",
    output:
        delay_distributions_results_paths,
    params:
        inputs=as_param(inputs=get_inputs_delay_distributions()),
    shell:
        "pixi run python -m scripts.delay_distributions"


# Each fit gets the whole machine (the pipeline parallelises over days itself). The width is
# a callable so it is only evaluated per job: workflow.cores raises when --cores is unset, as
# under a cluster executor, where --set-threads replaces it before it would be called.
rule model_fits:
    input:
        analysis_package_paths,
        "scripts/model_fits.py",
        DATA_PATH,
        delay_distributions_results_paths,
    output:
        model_fits_results_paths(model_spec_name="{spec}"),
    wildcard_constraints:
        spec="|".join(MODEL_SPECS),
    params:
        inputs=lambda wildcards: as_param(
            inputs=get_inputs_model_fits(model_spec_name=wildcards.spec)
        ),
    threads: lambda wildcards: workflow.cores
    shell:
        "pixi run python -m scripts.model_fits --spec {wildcards.spec}"


rule delay_distributions_plots:
    input:
        shared_input_paths,
        "scripts/delay_distributions_plots.py",
        delay_distributions_results_paths,
    output:
        delay_distributions_plots_fig_paths,
    shell:
        "pixi run python -m scripts.delay_distributions_plots"


rule fit_diagnostics_plots:
    input:
        shared_input_paths,
        "scripts/fit_diagnostics_plots.py",
        DATA_PATH,
        [
            get_results_paths_model_fits(model_spec_name=model_spec_name)[name]
            for model_spec_name in MODEL_SPECS
            for name in ("diagnostics", "risk", "evidence")
        ],
    output:
        fit_diagnostics_plots_fig_paths,
    shell:
        "pixi run python -m scripts.fit_diagnostics_plots"


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
