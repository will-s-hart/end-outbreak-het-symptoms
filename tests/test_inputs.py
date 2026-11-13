# Note that AI tools were used to generate tests
import pathlib

import pytest

import scripts.inputs
from endoutbreakhet.inference import _build_pymc_model
from scripts.inputs import (
    COMPARISON_SPECS,
    FIG_EXTENSIONS,
    FIT_GATE_THRESHOLDS,
    FIT_KWARGS,
    FIT_KWARGS_UNDER_REPORTING,
    MODEL_SPECS,
    get_fig_paths_numbered,
    get_inputs_delay_distributions,
    get_inputs_delay_distributions_plots,
    get_inputs_fit_diagnostics_plots,
    get_inputs_model_comparison_plots,
    get_inputs_model_fits,
    get_inputs_overview_plots,
    get_results_paths_delay_distributions,
)
from scripts.snakemake_utils import as_param


@pytest.mark.parametrize("model_spec_name", MODEL_SPECS)
def test_model_specs_build(*, model_spec_name, toy_inputs, monkeypatch):
    # The spec's model on toy data, so skip loading the real data. Its sampler settings are
    # nutpie's where every variable is continuous (as fit_model then samples with nutpie),
    # and PyMC's with Metropolis otherwise
    monkeypatch.setattr(scripts.inputs, "_get_shared_inputs", dict)
    inputs = get_inputs_model_fits(model_spec_name=model_spec_name)
    model = _build_pymc_model(
        model_name=inputs["model_name"],
        reported_incidence_vec=toy_inputs["reported_incidence_vec"],
        switch_day=toy_inputs["switch_day"],
        delay_distributions=toy_inputs["delay_distributions"],
        priors=inputs["priors"],
        reporting_prob=inputs["reporting_prob"],
    )
    assert inputs["fit_kwargs"] == (
        FIT_KWARGS_UNDER_REPORTING if model.discrete_value_vars else FIT_KWARGS
    )


@pytest.mark.parametrize("comparison_spec_name", COMPARISON_SPECS)
def test_comparison_specs_valid(*, comparison_spec_name):
    comparison_spec = COMPARISON_SPECS[comparison_spec_name]
    model_specs = [
        MODEL_SPECS[model_spec_name]
        for model_spec_name in comparison_spec["model_spec_names"]
    ]
    # The plots key results by model name, so these must be distinct
    model_names = [model_spec["model_name"] for model_spec in model_specs]
    assert len(set(model_names)) == len(model_names)
    recorded_risk_types = (
        {"cases", "transmission"}
        if "dlo" in model_names
        else {"cases", "transmission", "sustained", "reports"}
    )
    assert comparison_spec["figures"]
    for figure in comparison_spec["figures"].values():
        assert figure and figure.keys() <= {"param_names", "unreported", "risk_types"}
        # Non-empty tuples (a missing trailing comma would leave a bare string)
        for key in figure.keys() & {"param_names", "risk_types"}:
            assert isinstance(figure[key], tuple) and figure[key]
        # R_after's densities last, as their panel has the model probabilities' box
        assert figure.get("param_names", ("rep_no_after",))[-1] == "rep_no_after"
        # Given only to add the panel
        assert figure.get("unreported", True) is True
        # Risk types every model in the comparison records
        assert set(figure.get("risk_types", ())) <= recorded_risk_types


@pytest.mark.parametrize("comparison_spec_name", COMPARISON_SPECS)
def test_get_inputs_model_comparison_plots(*, comparison_spec_name, monkeypatch):
    # Only the paths, priors and figures are checked, so skip loading the data
    monkeypatch.setattr(scripts.inputs, "_get_shared_inputs", dict)
    comparison_spec = COMPARISON_SPECS[comparison_spec_name]
    inputs = get_inputs_model_comparison_plots(
        comparison_spec_name=comparison_spec_name
    )

    # Every figure has each kind of panel, or not
    figures = inputs["figures"]
    assert figures.keys() == comparison_spec["figures"].keys()
    for fig_name, figure in figures.items():
        assert figure == {
            "param_names": (),
            "unreported": False,
            "risk_types": (),
            **comparison_spec["figures"][fig_name],
        }
    # Reads the results the figures plot, where the fits write them
    plotted_result_names = {
        result_name
        for key, result_names in (
            ("param_names", ("posterior", "evidence")),
            ("unreported", ("unreported_incidence",)),
            ("risk_types", ("risk",)),
        )
        if any(figure[key] for figure in figures.values())
        for result_name in result_names
    }
    assert inputs["results_paths_by_model"] == {
        MODEL_SPECS[model_spec_name]["model_name"]: {
            result_name: path
            for result_name, path in get_inputs_model_fits(
                model_spec_name=model_spec_name
            )["results_paths"].items()
            if result_name in plotted_result_names
        }
        for model_spec_name in comparison_spec["model_spec_names"]
    }
    # A prior for each parameter with densities in any figure
    assert set(inputs["priors"]) == {
        param_name
        for figure in figures.values()
        for param_name in figure["param_names"]
    }

    fig_paths = inputs["fig_paths"]
    assert list(fig_paths) == list(comparison_spec["figures"])
    for fig_name, paths in fig_paths.items():
        assert [path.suffix for path in paths] == [f".{ext}" for ext in FIG_EXTENSIONS]
        for path in paths:
            assert path.parent == pathlib.Path("figures/model_comparison") / (
                comparison_spec_name
            )
            assert path.stem == fig_name


def test_get_inputs_model_comparison_plots_raises_for_conflicting_priors(
    *, monkeypatch
):
    monkeypatch.setitem(
        COMPARISON_SPECS,
        "conflicting_priors",
        {
            "model_spec_names": ["sse_fixed_dispersion", "ssi_estimated_dispersion"],
            "figures": {"summary": {"param_names": ("rep_no_before", "dispersion")}},
        },
    )
    with pytest.raises(ValueError, match="different priors for dispersion"):
        get_inputs_model_comparison_plots(comparison_spec_name="conflicting_priors")


def test_get_inputs_delay_distributions():
    inputs = get_inputs_delay_distributions()
    assert inputs["results_paths"] == get_results_paths_delay_distributions()
    assert inputs["cont_params_by_delay"].keys() == inputs["results_paths"].keys()
    as_param(inputs=inputs)


def _assert_summary_fig_paths(*, fig_paths, fig_dir):
    # A single figure, with a path in fig_dir for each format
    assert list(fig_paths) == ["summary"]
    assert fig_paths["summary"] == [
        pathlib.Path(fig_dir) / f"summary.{ext}" for ext in FIG_EXTENSIONS
    ]


def test_get_inputs_overview_plots(*, monkeypatch):
    # The data isn't checked, so skip loading it
    monkeypatch.setattr(scripts.inputs, "_get_shared_inputs", dict)
    inputs = get_inputs_overview_plots()
    # The means the delays are discretised from, with the TOST's making up the serial
    # interval's with the incubation period's
    cont_params_by_delay = get_inputs_delay_distributions()["cont_params_by_delay"]
    delay_means = inputs["delay_means"]
    assert delay_means == {
        delay_name: cont_params["mean"]
        for delay_name, cont_params in cont_params_by_delay.items()
    }
    assert delay_means["tost"] + delay_means["inc_period"] == pytest.approx(
        delay_means["serial_interval"]
    )
    _assert_summary_fig_paths(fig_paths=inputs["fig_paths"], fig_dir="figures/overview")


def test_get_inputs_delay_distributions_plots():
    inputs = get_inputs_delay_distributions_plots()
    # Reads the results where the analysis writes them
    assert (
        inputs["delay_distributions_results_paths"]
        == get_results_paths_delay_distributions()
    )
    _assert_summary_fig_paths(
        fig_paths=inputs["fig_paths"], fig_dir="figures/delay_distributions"
    )


def test_get_fig_paths_numbered():
    assert get_fig_paths_numbered(figure_number="S1") == [
        pathlib.Path(f"figures/figure_S1.{ext}") for ext in FIG_EXTENSIONS
    ]


def test_get_inputs_model_fits_reads_delay_distributions_results(*, monkeypatch):
    monkeypatch.setattr(scripts.inputs, "_get_shared_inputs", dict)
    inputs = get_inputs_model_fits(model_spec_name="sse_fixed_dispersion")
    assert (
        inputs["delay_distributions_results_paths"]
        == get_results_paths_delay_distributions()
    )


def test_get_inputs_fit_diagnostics_plots(*, monkeypatch):
    # Only the paths and values are checked, so skip loading the data
    monkeypatch.setattr(scripts.inputs, "_get_shared_inputs", dict)
    inputs = get_inputs_fit_diagnostics_plots()

    # Reads every spec's results where the fits write them
    assert inputs["results_paths_by_spec"] == {
        model_spec_name: get_inputs_model_fits(model_spec_name=model_spec_name)[
            "results_paths"
        ]
        for model_spec_name in MODEL_SPECS
    }
    assert inputs["reporting_prob_by_spec"] == {
        model_spec_name: model_spec["reporting_prob"]
        for model_spec_name, model_spec in MODEL_SPECS.items()
    }
    # Fixed parameters, which the gate leaves out
    assert inputs["fixed_param_names_by_spec"]["sse_fixed_dispersion"] == ["dispersion"]
    assert inputs["fixed_param_names_by_spec"]["sse_estimated_dispersion"] == []
    assert inputs["fixed_param_names_by_spec"].keys() == MODEL_SPECS.keys()
    assert inputs["thresholds"] == FIT_GATE_THRESHOLDS

    fig_paths = inputs["fig_paths"]
    assert list(fig_paths) == ["gate"]
    assert [path.suffix for path in fig_paths["gate"]] == [
        f".{ext}" for ext in FIG_EXTENSIONS
    ]
    for path in fig_paths["gate"]:
        assert path.parent == pathlib.Path("figures/fit_diagnostics")
        assert path.stem == "gate"


def test_model_fits_rngs_distinct_per_spec_and_independent_of_order(*, monkeypatch):
    # Each spec has a stream of its own (so specs' log evidence errors are independent),
    # keyed by its name rather than its place among the specs
    monkeypatch.setattr(scripts.inputs, "_get_shared_inputs", dict)

    def _first_draws():
        return {
            model_spec_name: get_inputs_model_fits(model_spec_name=model_spec_name)[
                "rng"
            ].random()
            for model_spec_name in MODEL_SPECS
        }

    first_draws = _first_draws()
    assert len(set(first_draws.values())) == len(MODEL_SPECS)
    monkeypatch.setattr(
        scripts.inputs, "MODEL_SPECS", dict(reversed(MODEL_SPECS.items()))
    )
    assert _first_draws() == first_draws


def test_model_fits_params_distinct_per_spec_and_stable():
    # Snakemake compares params by repr
    params = _get_model_fits_params()
    assert len(set(params.values())) == len(MODEL_SPECS)
    assert _get_model_fits_params() == params


def test_editing_a_model_spec_changes_only_its_params(*, monkeypatch):
    params_before = _get_model_fits_params()
    monkeypatch.setitem(
        MODEL_SPECS,
        "sse_60_pc_reporting",
        MODEL_SPECS["sse_60_pc_reporting"] | {"reporting_prob": 0.61},
    )
    params_after = _get_model_fits_params()
    assert {
        model_spec_name
        for model_spec_name in MODEL_SPECS
        if params_after[model_spec_name] != params_before[model_spec_name]
    } == {"sse_60_pc_reporting"}


def _get_model_fits_params():
    return {
        model_spec_name: repr(
            as_param(inputs=get_inputs_model_fits(model_spec_name=model_spec_name))
        )
        for model_spec_name in MODEL_SPECS
    }
