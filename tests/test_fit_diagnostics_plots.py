# Note that AI tools were used to generate tests
import numpy as np
import pandas as pd
import pytest

from scripts import fit_diagnostics_plots
from scripts.inputs import FIT_GATE_THRESHOLDS, get_inputs_fit_diagnostics_plots

# A variable's diagnostics that pass every threshold, and those of a constant variable
_PASSING_DIAGNOSTICS = {
    "n_nonfinite": 0,
    "n_constant": 0,
    "n_diagnosed": 5,
    "max_rhat": 1.001,
    "min_ess_bulk": 1000.0,
    "min_ess_tail": 900.0,
    "median_rhat": 1.0,
    "median_ess_bulk": 2000.0,
    "median_ess_tail": 1800.0,
}
_UNASSESSED_DIAGNOSTICS = {
    "n_constant": 5,
    "n_diagnosed": 0,
    **{
        name: np.nan
        for name in _PASSING_DIAGNOSTICS
        if name.startswith(("max", "min", "median"))
    },
}


def test_gate():
    # One case per day, on variables a and b and a fixed parameter (constant, but left out
    # of the gate), and risks of cases and transmission
    diagnostics_df = pd.DataFrame(
        [
            {"day": day}
            | _diagnostics_row(
                n_divergences=n_divergences,
                n_max_treedepth=n_max_treedepth,
                overrides_by_variable={
                    "a": a,
                    "b": b,
                    "dispersion": _UNASSESSED_DIAGNOSTICS,
                },
            )
            for day, (n_divergences, n_max_treedepth, a, b) in enumerate(
                [
                    # Everything passes
                    (0, 0, {}, {}),
                    # An R-hat past its threshold
                    (0, 0, {"max_rhat": 1.02}, {}),
                    # A non-finite draw fails every criterion on the variable, and NUTS
                    # didn't run
                    (np.nan, np.nan, {"n_nonfinite": 1}, {}),
                    # A divergence, an infinite ESS and a NaN R-hat
                    (1, 0, {"min_ess_bulk": np.inf}, {"max_rhat": np.nan}),
                    # A constant variable alongside one that passes, which is shown
                    # rather than passed, as it could be stuck in every chain
                    (0, 0, _UNASSESSED_DIAGNOSTICS, {}),
                    # Constant variables only
                    (0, 0, _UNASSESSED_DIAGNOSTICS, _UNASSESSED_DIAGNOSTICS),
                    # Draws at the maximum tree depth, which alone fail the fit
                    (0, 3, {}, {}),
                ]
            )
        ]
    )
    risk_df = pd.DataFrame(
        {
            "day": np.arange(7),
            "risk_cases": 0.5,
            # Within the threshold; past it; zero (constant samples, so no R-hat); a NaN
            # R-hat; a NaN MCSE; an R-hat past its threshold; within the threshold
            "risk_mcse_cases": [0.001, 0.02, 0.0, 0.001, np.nan, 0.001, 0.001],
            "risk_rhat_cases": [1.0, 1.0, np.nan, np.nan, 1.0, 1.02, 1.0],
            "risk_transmission": 0.5,
            "risk_mcse_transmission": [0.02, *[0.001] * 6],
            "risk_rhat_transmission": 1.0,
        }
    )

    gate = fit_diagnostics_plots._gate(
        diagnostics_df=diagnostics_df,
        risk_df=risk_df,
        log_evidence_se=0.05,
        fixed_param_names=["dispersion"],
        thresholds=FIT_GATE_THRESHOLDS,
    )

    expected = {
        "rhat": ["pass", "fail", "fail", "fail", "unassessed", "unassessed", "pass"],
        "ess_bulk": [
            "pass",
            "pass",
            "fail",
            "fail",
            "unassessed",
            "unassessed",
            "pass",
        ],
        "ess_tail": [
            "pass",
            "pass",
            "fail",
            "pass",
            "unassessed",
            "unassessed",
            "pass",
        ],
        "divergences": ["pass", "pass", "unassessed", "fail", "pass", "pass", "pass"],
        "tree_depth": ["pass", "pass", "unassessed", "pass", "pass", "pass", "fail"],
        "fit": ["pass", "fail", "fail", "fail", "unassessed", "unassessed", "fail"],
        "risk": ["fail", "fail", "pass", "fail", "fail", "fail", "pass"],
    }
    for criterion, statuses in expected.items():
        assert list(gate[criterion]) == statuses, criterion
    assert gate["evidence"] == "pass"


@pytest.mark.parametrize(
    ("log_evidence_se", "status"),
    [(0.05, "pass"), (0.2, "fail"), (np.nan, "fail"), (np.inf, "fail")],
)
def test_gate_evidence(*, log_evidence_se, status):
    gate = fit_diagnostics_plots._gate(
        diagnostics_df=pd.DataFrame(
            [
                {"day": 0}
                | _diagnostics_row(
                    n_divergences=0, n_max_treedepth=0, overrides_by_variable={"a": {}}
                )
            ]
        ),
        risk_df=pd.DataFrame(
            {
                "day": [0],
                "risk_cases": 0.5,
                "risk_mcse_cases": 0.0,
                "risk_rhat_cases": 1.0,
            }
        ),
        log_evidence_se=log_evidence_se,
        fixed_param_names=[],
        thresholds=FIT_GATE_THRESHOLDS,
    )
    assert gate["evidence"] == status


def test_format_days():
    assert fit_diagnostics_plots._format_days(days=np.array([3, 4, 5, 8, 10, 11])) == (
        "3-5, 8, 10-11"
    )


@pytest.mark.parametrize("failing", [False, True])
def test_make_plots_writes_figure_and_summary(
    *, failing, tmp_path, monkeypatch, capsys
):
    # Real inputs, with the results and figure redirected into tmp_path and fake results
    # written there, in which fixed parameters are constant. Under-reporting specs'
    # unreported cases are (1 - reporting prob) / reporting prob times the reported cases.
    # If failing, the first spec fails R-hat on two days and its evidence, a variable is
    # constant on two others, and draws reach the maximum tree depth on another
    inputs = get_inputs_fit_diagnostics_plots()
    inputs["results_paths_by_spec"] = {
        model_spec_name: {key: tmp_path / path for key, path in paths.items()}
        for model_spec_name, paths in inputs["results_paths_by_spec"].items()
    }
    inputs["fig_paths"] = {
        fig_name: [tmp_path / path for path in paths]
        for fig_name, paths in inputs["fig_paths"].items()
    }
    days = inputs["risk_days"]
    reported_incidence_vec = inputs["reported_incidence_vec"]
    first_spec_name = next(iter(inputs["results_paths_by_spec"]))
    for model_spec_name, paths in inputs["results_paths_by_spec"].items():
        paths["risk"].parent.mkdir(parents=True)
        reporting_prob = inputs["reporting_prob_by_spec"][model_spec_name]
        diagnostics_df = pd.DataFrame(
            [
                {"day": day}
                | _diagnostics_row(
                    n_divergences=0,
                    n_max_treedepth=0,
                    overrides_by_variable={
                        "a": {},
                        "b": {},
                        **dict.fromkeys(
                            inputs["fixed_param_names_by_spec"][model_spec_name],
                            _UNASSESSED_DIAGNOSTICS,
                        ),
                    },
                )
                for day in days
            ]
        )
        if reporting_prob < 1:
            diagnostics_df["mean_total_unreported"] = [
                (1 - reporting_prob)
                / reporting_prob
                * reported_incidence_vec[1 : day + 1].sum()
                for day in days
            ]
        log_evidence_se = 0.01
        if failing and model_spec_name == first_spec_name:
            diagnostics_df.loc[[3, 4], "max_rhat_a"] = 1.05
            for name, value in _UNASSESSED_DIAGNOSTICS.items():
                diagnostics_df.loc[[7, 8], f"{name}_b"] = value
            diagnostics_df.loc[10, "n_max_treedepth"] = 2
            log_evidence_se = 0.5
        diagnostics_df.to_csv(paths["diagnostics"], index=False)
        pd.DataFrame(
            {
                "day": days,
                "risk_cases": 0.5,
                "risk_mcse_cases": 0.001,
                "risk_rhat_cases": 1.0,
            }
        ).to_csv(paths["risk"], index=False)
        pd.DataFrame(
            {"log_evidence": [-80.0], "log_evidence_se": [log_evidence_se]}
        ).to_csv(paths["evidence"], index=False)
    monkeypatch.setattr(
        fit_diagnostics_plots, "get_inputs_fit_diagnostics_plots", lambda: inputs
    )

    fit_diagnostics_plots.make_plots()

    for path in inputs["fig_paths"]["gate"]:
        assert path.stat().st_size > 0
    output = capsys.readouterr().out
    n_specs = len(inputs["results_paths_by_spec"])
    if failing:
        assert (
            f"1 of {n_specs} specs don't pass the fit gate:\n{first_spec_name}\n"
            in output
        )
        assert f"  rhat fails on 2 of {len(days)} days ({days[3]}-{days[4]})" in output
        assert (
            f"  rhat is unassessed on 2 of {len(days)} days ({days[7]}-{days[8]})"
            in output
        )
        assert f"    a rhat: worst 1.05 on day {days[3]}" in output
        assert f"    b: constant in every draw on days {days[7]}-{days[8]}" in output
        assert f"  tree_depth fails on 1 of {len(days)} days ({days[10]})\n" in output
        assert (
            f"    tree_depth: worst 2 on day {days[10]} (median over days 0)" in output
        )
        assert "  evidence fails: log evidence SE 0.5" in output
    else:
        assert f"All {n_specs} specs pass the fit gate" in output
    for model_spec_name, reporting_prob in inputs["reporting_prob_by_spec"].items():
        if reporting_prob < 1:
            assert f"  {model_spec_name}: 1-1\n" in output


def _diagnostics_row(*, n_divergences, n_max_treedepth, overrides_by_variable):
    # A day's diagnostics of each variable, passing unless overridden
    row = {"n_divergences": n_divergences, "n_max_treedepth": n_max_treedepth}
    for variable_name, overrides in overrides_by_variable.items():
        row |= {
            f"{name}_{variable_name}": value
            for name, value in (_PASSING_DIAGNOSTICS | overrides).items()
        }
    return row
