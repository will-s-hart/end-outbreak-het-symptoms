import argparse

import numpy as np
import pandas as pd

from endoutbreakhet.plotting import plot_fit_gate, save_figure
from scripts.inputs import get_inputs_fit_diagnostics_plots

# Fit criteria on a variable's diagnostic, by the prefix of its columns in the diagnostics,
# which is also its threshold's name
_STAT_CRITERIA = {
    "rhat": "max_rhat",
    "ess_bulk": "min_ess_bulk",
    "ess_tail": "min_ess_tail",
}
_DAY_CRITERIA = (*_STAT_CRITERIA, "divergences", "risk")


def make_plots():
    inputs = get_inputs_fit_diagnostics_plots()

    thresholds = inputs["thresholds"]
    results_by_spec = {
        model_spec_name: {
            "diagnostics_df": pd.read_csv(paths["diagnostics"]),
            "risk_df": pd.read_csv(paths["risk"]),
            "log_evidence_se": pd.read_csv(paths["evidence"])["log_evidence_se"].iloc[
                0
            ],
        }
        for model_spec_name, paths in inputs["results_paths_by_spec"].items()
    }
    fixed_param_names_by_spec = inputs["fixed_param_names_by_spec"]
    gate_by_spec = {
        model_spec_name: _gate(
            **results,
            fixed_param_names=fixed_param_names_by_spec[model_spec_name],
            thresholds=thresholds,
        )
        for model_spec_name, results in results_by_spec.items()
    }
    # Every spec is fitted on the same days
    days = next(iter(results_by_spec.values()))["risk_df"]["day"].to_numpy()
    gate_fig = plot_fit_gate(
        status_by_criterion={
            criterion: np.stack([gate[criterion] for gate in gate_by_spec.values()])
            for criterion in _DAY_CRITERIA
        },
        fit_status=np.stack([gate["fit"] for gate in gate_by_spec.values()]),
        evidence_status=[gate["evidence"] for gate in gate_by_spec.values()],
        spec_names=list(gate_by_spec),
        days=days,
        comparison_day=inputs["comparison_day"],
    )
    save_figure(fig=gate_fig, paths=inputs["fig_paths"]["gate"])

    _print_summary(
        results_by_spec=results_by_spec,
        fixed_param_names_by_spec=fixed_param_names_by_spec,
        gate_by_spec=gate_by_spec,
        thresholds=thresholds,
    )
    _print_unreported_ratios(
        results_by_spec=results_by_spec,
        reporting_prob_by_spec=inputs["reporting_prob_by_spec"],
        reported_incidence_vec=inputs["reported_incidence_vec"],
    )


def _gate(*, diagnostics_df, risk_df, log_evidence_se, fixed_param_names, thresholds):
    # "pass", "fail" or "unassessed" for each criterion on each day (the results' rows), and
    # for the evidence. A fit criterion on the variables' diagnostics (see
    # _variable_statuses) fails if any variable fails it, is unassessed if any is, and
    # passes otherwise, leaving out fixed parameters, which are constant by construction.
    # A variable that's constant in every draw can't be told from one stuck in every chain,
    # so it's shown, rather than passed. The fit fails if any fit criterion does. Risks are
    # gated on every day (see _risk_passes), but where the fit fails their MCSE is only
    # conditional on the exploration achieved
    gate = {
        criterion: _combine(
            statuses=[
                _variable_statuses(
                    diagnostics_df=diagnostics_df,
                    variable_name=variable_name,
                    stat_prefix=stat_prefix,
                    threshold=thresholds[stat_prefix],
                )
                for variable_name in _gated_variable_names(
                    diagnostics_df=diagnostics_df, fixed_param_names=fixed_param_names
                )
            ]
        )
        for criterion, stat_prefix in _STAT_CRITERIA.items()
    }
    n_divergences_vec = diagnostics_df["n_divergences"].to_numpy(dtype=float)
    gate["divergences"] = np.where(
        np.isnan(n_divergences_vec),
        "unassessed",
        np.where(n_divergences_vec <= thresholds["max_n_divergences"], "pass", "fail"),
    )
    gate["fit"] = _combine(
        statuses=[gate[criterion] for criterion in (*_STAT_CRITERIA, "divergences")]
    )
    gate["risk"] = _combine(
        statuses=[
            np.where(
                _risk_passes(
                    risk_df=risk_df, risk_type=risk_type, thresholds=thresholds
                ),
                "pass",
                "fail",
            )
            for risk_type in _risk_types(risk_df=risk_df)
        ]
    )
    gate["evidence"] = (
        "pass"
        if _within(
            values=log_evidence_se,
            threshold=thresholds["max_log_evidence_se"],
            upper=True,
        )
        else "fail"
    )
    return gate


def _gated_variable_names(*, diagnostics_df, fixed_param_names):
    variable_names = [
        column.removeprefix("n_diagnosed_")
        for column in diagnostics_df.columns
        if column.startswith("n_diagnosed_")
    ]
    return [name for name in variable_names if name not in fixed_param_names]


def _variable_statuses(*, diagnostics_df, variable_name, stat_prefix, threshold):
    # On each day, whether a variable fails (an element with a non-finite draw, or a
    # non-finite diagnostic, or one past the threshold, which bounds the worst element's
    # diagnostic from above for a "max" prefix and below for a "min" one), is unassessed
    # (no elements diagnosed) or passes
    return np.where(
        diagnostics_df[f"n_nonfinite_{variable_name}"].to_numpy() > 0,
        "fail",
        np.where(
            diagnostics_df[f"n_diagnosed_{variable_name}"].to_numpy() == 0,
            "unassessed",
            np.where(
                _within(
                    values=diagnostics_df[f"{stat_prefix}_{variable_name}"],
                    threshold=threshold,
                    upper=stat_prefix.startswith("max"),
                ),
                "pass",
                "fail",
            ),
        ),
    )


def _within(*, values, threshold, upper):
    # Whether values are finite and at most (if upper) or at least the threshold
    values = np.asarray(values, dtype=float)
    return np.isfinite(values) & (values <= threshold if upper else values >= threshold)


def _combine(*, statuses):
    # Elementwise over arrays of statuses: "fail" if any fails, else "unassessed" if any
    # is, else "pass"
    stacked = np.stack(statuses)
    return np.where(
        np.any(stacked == "fail", axis=0),
        "fail",
        np.where(np.any(stacked == "unassessed", axis=0), "unassessed", "pass"),
    )


def _risk_types(*, risk_df):
    return [
        column.removeprefix("risk_mcse_")
        for column in risk_df.columns
        if column.startswith("risk_mcse_")
    ]


def _risk_passes(*, risk_df, risk_type, thresholds):
    # Whether a risk's MCSE is finite and within its threshold on each day, and so is its
    # R-hat, unless the MCSE is zero (every sample the same, so no R-hat)
    mcse_vec = risk_df[f"risk_mcse_{risk_type}"].to_numpy(dtype=float)
    return _within(
        values=mcse_vec, threshold=thresholds["max_risk_mcse"], upper=True
    ) & (
        (mcse_vec == 0)
        | _within(
            values=risk_df[f"risk_rhat_{risk_type}"],
            threshold=thresholds["max_rhat"],
            upper=True,
        )
    )


def _print_summary(
    *, results_by_spec, fixed_param_names_by_spec, gate_by_spec, thresholds
):
    # Every spec that doesn't pass, with what fails or is unassessed
    flagged_spec_names = [
        model_spec_name
        for model_spec_name, gate in gate_by_spec.items()
        if gate["evidence"] == "fail"
        or any(np.any(gate[criterion] != "pass") for criterion in _DAY_CRITERIA)
    ]
    if not flagged_spec_names:
        print(f"All {len(gate_by_spec)} specs pass the fit gate")
        return
    print(
        f"{len(flagged_spec_names)} of {len(gate_by_spec)} specs don't pass the fit gate:"
    )
    for model_spec_name in flagged_spec_names:
        print(model_spec_name)
        _print_spec_problems(
            **results_by_spec[model_spec_name],
            fixed_param_names=fixed_param_names_by_spec[model_spec_name],
            gate=gate_by_spec[model_spec_name],
            thresholds=thresholds,
        )


def _print_spec_problems(
    *, diagnostics_df, risk_df, log_evidence_se, fixed_param_names, gate, thresholds
):
    # The days each criterion fails or is unassessed on, then why, with the worst values
    days = risk_df["day"].to_numpy()
    for criterion in _DAY_CRITERIA:
        for status, verb in (("fail", "fails"), ("unassessed", "is unassessed")):
            status_idx = gate[criterion] == status
            if np.any(status_idx):
                print(
                    f"  {criterion} {verb} on {status_idx.sum()} of {len(days)} days "
                    f"({_format_days(days=days[status_idx])})"
                )

    for variable_name in _gated_variable_names(
        diagnostics_df=diagnostics_df, fixed_param_names=fixed_param_names
    ):
        nonfinite_idx = diagnostics_df[f"n_nonfinite_{variable_name}"].to_numpy() > 0
        if np.any(nonfinite_idx):
            print(
                f"    {variable_name}: non-finite draws on days "
                f"{_format_days(days=days[nonfinite_idx])}"
            )
        constant_idx = ~nonfinite_idx & (
            diagnostics_df[f"n_diagnosed_{variable_name}"].to_numpy() == 0
        )
        if np.any(constant_idx):
            print(
                f"    {variable_name}: constant in every draw on days "
                f"{_format_days(days=days[constant_idx])}"
            )
        for criterion, stat_prefix in _STAT_CRITERIA.items():
            fail_idx = ~nonfinite_idx & (
                _variable_statuses(
                    diagnostics_df=diagnostics_df,
                    variable_name=variable_name,
                    stat_prefix=stat_prefix,
                    threshold=thresholds[stat_prefix],
                )
                == "fail"
            )
            if not np.any(fail_idx):
                continue
            # Worst day: a non-finite diagnostic, else the one furthest past the threshold
            stat_vec = diagnostics_df[f"{stat_prefix}_{variable_name}"].to_numpy()
            badness_vec = np.where(
                np.isfinite(stat_vec),
                stat_vec if stat_prefix.startswith("max") else -stat_vec,
                np.inf,
            )
            worst = np.flatnonzero(fail_idx)[np.argmax(badness_vec[fail_idx])]
            median = diagnostics_df[f"median_{criterion}_{variable_name}"].iloc[worst]
            print(
                f"    {variable_name} {criterion}: worst {stat_vec[worst]:.4g} on day "
                f"{days[worst]} (median over elements {median:.4g})"
            )

    if np.any(gate["divergences"] == "fail"):
        n_divergences_vec = diagnostics_df["n_divergences"].to_numpy()
        worst = np.nanargmax(n_divergences_vec)
        print(
            f"    divergences: worst {n_divergences_vec[worst]:.0f} on day "
            f"{days[worst]} (median over days {np.nanmedian(n_divergences_vec):.0f})"
        )

    for risk_type in _risk_types(risk_df=risk_df):
        fail_idx = ~_risk_passes(
            risk_df=risk_df, risk_type=risk_type, thresholds=thresholds
        )
        if np.any(fail_idx):
            print(
                f"    risk_{risk_type} on days {_format_days(days=days[fail_idx])}: "
                "worst MCSE "
                f"{np.max(risk_df[f'risk_mcse_{risk_type}'].to_numpy()[fail_idx]):.4g}, "
                "worst R-hat "
                f"{np.max(risk_df[f'risk_rhat_{risk_type}'].to_numpy()[fail_idx]):.4g}"
            )

    if gate["evidence"] == "fail":
        print(f"  evidence fails: log evidence SE {log_evidence_se:.4g}")


def _format_days(*, days):
    # Runs of consecutive days as ranges, e.g. "3-5, 8"
    runs = np.split(days, np.flatnonzero(np.diff(days) != 1) + 1)
    return ", ".join(
        str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}" for run in runs
    )


def _print_unreported_ratios(
    *, results_by_spec, reporting_prob_by_spec, reported_incidence_vec
):
    # For each under-reporting spec, the range over days of the posterior mean total of
    # unreported cases relative to (1 - reporting_prob) / reporting_prob times the reported
    # cases after day 0. That factor relates the expected numbers of unreported and
    # reported cases, but the ratio has no posterior target, so it's descriptive only
    under_reporting_probs = {
        model_spec_name: reporting_prob
        for model_spec_name, reporting_prob in reporting_prob_by_spec.items()
        if reporting_prob < 1
    }
    if not under_reporting_probs:
        return
    print(
        "Posterior mean unreported cases relative to (1 - reporting prob) / reporting "
        "prob times reported cases after day 0, range over days:"
    )
    for model_spec_name, reporting_prob in under_reporting_probs.items():
        diagnostics_df = results_by_spec[model_spec_name]["diagnostics_df"]
        # Onsets after the data are zero, as in the fits
        reported_total_vec = np.array(
            [reported_incidence_vec[1 : day + 1].sum() for day in diagnostics_df["day"]]
        )
        ratio_vec = diagnostics_df["mean_total_unreported"].to_numpy() / (
            (1 - reporting_prob) / reporting_prob * reported_total_vec
        )
        print(f"  {model_spec_name}: {np.min(ratio_vec):.3g}-{np.max(ratio_vec):.3g}")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=(
            "Plot which fits, risks and evidence pass the fit gate's thresholds, and "
            "summarise what doesn't."
        )
    ).parse_args()

    make_plots()
