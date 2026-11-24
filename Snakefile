import collections
import glob
import shutil

from scripts.inputs import (
    COMPARISON_SPECS,
    DATA_PATH,
    MODEL_SPECS,
    get_fig_paths_delay_distributions_plots,
    get_fig_paths_fit_diagnostics_plots,
    get_fig_paths_model_comparison_plots,
    get_fig_paths_numbered,
    get_fig_paths_overview_plots,
    get_inputs_delay_distributions,
    get_inputs_model_comparison_plots,
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
overview_plots_fig_paths = [
    path for paths in get_fig_paths_overview_plots().values() for path in paths
]
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


def model_comparison_plots_results_paths(*, comparison_spec_name):
    # Just the results the comparison's figures plot, as the script reads them
    results_paths_by_model = get_inputs_model_comparison_plots(
        comparison_spec_name=comparison_spec_name
    )["results_paths_by_model"]
    return [path for paths in results_paths_by_model.values() for path in paths.values()]


def model_comparison_plots_fig_paths(*, comparison_spec_name):
    fig_paths = get_fig_paths_model_comparison_plots(
        comparison_spec_name=comparison_spec_name
    )
    return [path for paths in fig_paths.values() for path in paths]


# Every figure's number (1-5 in the paper, S1-S7 in its supplement, A1 onwards for figures
# in neither), with the figure the numbered copy is made from
numbered_figure_sources = {
    "1": get_fig_paths_overview_plots()["summary"],
    "S1": get_fig_paths_delay_distributions_plots()["summary"],
    **{
        figure_number: get_fig_paths_model_comparison_plots(
            comparison_spec_name=comparison_spec_name
        )[fig_name]
        for figure_number, comparison_spec_name, fig_name in (
            ("2", "serial_interval_based_models_fixed_dispersion", "summary"),
            ("3", "serial_interval_based_models_estimated_dispersion", "summary"),
            ("4", "serial_interval_based_vs_so_fixed_dispersion", "summary"),
            ("5", "serial_interval_based_vs_so_80_pc_reporting", "unreported_risk"),
            ("S2", "serial_interval_based_vs_so_estimated_dispersion", "summary"),
            ("S3", "serial_interval_based_vs_so_80_pc_reporting", "posterior"),
            ("S4", "serial_interval_based_vs_so_60_pc_reporting", "summary"),
            ("S5", "serial_interval_based_vs_so_fixed_dispersion", "risk_sustained"),
            ("S6", "serial_interval_based_vs_so_80_pc_reporting", "risk_cases_reports"),
            ("S7", "poisson_vs_superspreading", "summary"),
        )
    },
    "A1": get_fig_paths_fit_diagnostics_plots()["gate"],
}

# Built before the rules below, since Snakemake resolves a rule's inputs when it is defined
results_paths = [
    delay_distributions_results_paths,
    *(
        model_fits_results_paths(model_spec_name=model_spec_name)
        for model_spec_name in MODEL_SPECS
    ),
]
plot_fig_paths = [
    *overview_plots_fig_paths,
    *delay_distributions_plots_fig_paths,
    *fit_diagnostics_plots_fig_paths,
    *(
        path
        for comparison_spec_name in COMPARISON_SPECS
        for path in model_comparison_plots_fig_paths(
            comparison_spec_name=comparison_spec_name
        )
    ),
]
# Each figure the plot rules make has exactly one number
numbered_source_counts = collections.Counter(
    path for paths in numbered_figure_sources.values() for path in paths
)
numbering_errors = {
    "unnumbered": set(plot_fig_paths) - numbered_source_counts.keys(),
    "not made by a plot rule": numbered_source_counts.keys() - set(plot_fig_paths),
    "numbered more than once": {
        path for path, count in numbered_source_counts.items() if count > 1
    },
}
if any(numbering_errors.values()):
    raise ValueError(
        "Every figure needs exactly one number in numbered_figure_sources: "
        + "; ".join(
            f"{error}: {sorted(map(str, paths))}"
            for error, paths in numbering_errors.items()
            if paths
        )
    )
figure_paths = [
    plot_fig_paths,
    *(
        get_fig_paths_numbered(figure_number=figure_number)
        for figure_number in numbered_figure_sources
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


# One rule for the specs with complete reporting and one for those with under-reporting,
# which take their own sampler settings, so that a cluster profile can size each group's jobs
# (the under-reporting fits take several times as long). Each fit gets the whole machine
# (the pipeline parallelises over days itself). The width is a callable so it is only
# evaluated per job: workflow.cores raises when --cores is unset, as under a cluster
# executor, where --set-threads replaces it before it would be called.
rule model_fits_complete_reporting:
    input:
        analysis_package_paths,
        "scripts/model_fits.py",
        DATA_PATH,
        delay_distributions_results_paths,
    output:
        model_fits_results_paths(model_spec_name="{spec}"),
    wildcard_constraints:
        spec="|".join(
            model_spec_name
            for model_spec_name, model_spec in MODEL_SPECS.items()
            if model_spec["reporting_prob"] == 1
        ),
    params:
        inputs=lambda wildcards: as_param(
            inputs=get_inputs_model_fits(model_spec_name=wildcards.spec)
        ),
    threads: lambda wildcards: workflow.cores
    shell:
        "pixi run python -m scripts.model_fits --spec {wildcards.spec}"


use rule model_fits_complete_reporting as model_fits_under_reporting with:
    wildcard_constraints:
        spec="|".join(
            model_spec_name
            for model_spec_name, model_spec in MODEL_SPECS.items()
            if model_spec["reporting_prob"] < 1
        ),


rule overview_plots:
    input:
        shared_input_paths,
        "scripts/overview_plots.py",
        DATA_PATH,
    output:
        overview_plots_fig_paths,
    shell:
        "pixi run python -m scripts.overview_plots"


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
for comparison_spec_name in COMPARISON_SPECS:

    rule:
        name:
            f"model_comparison_plots_{comparison_spec_name}"
        input:
            shared_input_paths,
            "scripts/model_comparison_plots.py",
            DATA_PATH,
            model_comparison_plots_results_paths(
                comparison_spec_name=comparison_spec_name
            ),
        output:
            model_comparison_plots_fig_paths(comparison_spec_name=comparison_spec_name),
        params:
            spec=comparison_spec_name,
        shell:
            "pixi run python -m scripts.model_comparison_plots --spec {params.spec}"


# One rule per figure, copying it (in each format) to its numbered path
for figure_number, source_paths in numbered_figure_sources.items():

    rule:
        name:
            f"numbered_figure_{figure_number}"
        input:
            source_paths,
        output:
            get_fig_paths_numbered(figure_number=figure_number),
        localrule: True
        run:
            for source_path, path in zip(input, output, strict=True):
                shutil.copyfile(source_path, path)
