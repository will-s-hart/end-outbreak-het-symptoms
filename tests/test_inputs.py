# Note that AI tools were used to generate tests
import pathlib

import pytest

import scripts.inputs
from endoutbreakhet.inference import _build_pymc_model
from scripts.inputs import (
    COMPARISON_SPECS,
    FIG_EXTENSIONS,
    MODEL_SPECS,
    get_inputs_model_comparison_plots,
    get_inputs_model_fits,
)


@pytest.mark.parametrize("model_spec_name", MODEL_SPECS)
def test_model_specs_build(*, model_spec_name, toy_inputs):
    model_spec = MODEL_SPECS[model_spec_name]
    _build_pymc_model(
        model_name=model_spec["model_name"],
        reported_incidence_vec=toy_inputs["reported_incidence_vec"],
        switch_day=toy_inputs["switch_day"],
        delay_distributions=toy_inputs["delay_distributions"],
        priors=model_spec["priors"],
        reporting_prob=1.0,
    )


@pytest.mark.parametrize("comparison_spec_name", COMPARISON_SPECS)
def test_comparison_specs_valid(*, comparison_spec_name):
    comparison_spec = COMPARISON_SPECS[comparison_spec_name]
    # The plots key results by model name, so these must be distinct
    model_names = [
        MODEL_SPECS[model_spec_name]["model_name"]
        for model_spec_name in comparison_spec["model_spec_names"]
    ]
    assert len(set(model_names)) == len(model_names)
    assert isinstance(comparison_spec["risk_types"], tuple)
    assert set(comparison_spec["risk_types"]) <= {"cases", "transmission"}
    assert isinstance(comparison_spec["param_names"], tuple)


@pytest.mark.parametrize("comparison_spec_name", COMPARISON_SPECS)
def test_get_inputs_model_comparison_plots(*, comparison_spec_name, monkeypatch):
    # Only the paths and priors are checked, so skip loading the data
    monkeypatch.setattr(scripts.inputs, "_get_shared_inputs", dict)
    comparison_spec = COMPARISON_SPECS[comparison_spec_name]
    inputs = get_inputs_model_comparison_plots(
        comparison_spec_name=comparison_spec_name
    )

    # Reads the results where the fits write them
    assert inputs["results_paths_by_model"] == {
        MODEL_SPECS[model_spec_name]["model_name"]: get_inputs_model_fits(
            model_spec_name=model_spec_name
        )["results_paths"]
        for model_spec_name in comparison_spec["model_spec_names"]
    }
    assert set(inputs["priors"]) == set(comparison_spec["param_names"])

    fig_paths = inputs["fig_paths"]
    assert set(fig_paths) == {
        *(f"densities_{param_name}" for param_name in comparison_spec["param_names"]),
        "model_probabilities",
        "risk_incidence_overlay",
    }
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
            "risk_types": ("cases",),
            "param_names": ("rep_no_before", "dispersion"),
        },
    )
    with pytest.raises(ValueError, match="different priors for dispersion"):
        get_inputs_model_comparison_plots(comparison_spec_name="conflicting_priors")
