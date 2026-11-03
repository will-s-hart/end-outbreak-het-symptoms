import argparse

import pandas as pd

from endoutbreakhet import compute_model_probabilities
from endoutbreakhet.plotting import (
    make_panel_figure,
    plot_densities,
    plot_model_probabilities,
    plot_model_probabilities_inset,
    plot_risk_incidence_overlay,
    plot_unreported_incidence,
    save_figure,
    set_plot_style,
)
from scripts.inputs import COMPARISON_SPECS, get_inputs_model_comparison_plots


def make_plots(*, comparison_spec_name):
    set_plot_style()
    inputs = get_inputs_model_comparison_plots(
        comparison_spec_name=comparison_spec_name
    )

    # Each result the figures plot, by model
    df_by_model_by_result = {}
    for model_name, paths in inputs["results_paths_by_model"].items():
        for result_name, path in paths.items():
            df_by_model_by_result.setdefault(result_name, {})[model_name] = pd.read_csv(
                path
            )
    vline_days = [inputs["ert_days"]["arrival"], inputs["ert_days"]["departure"]]
    for fig_name, figure in inputs["figures"].items():
        fig = _make_figure(
            **figure,
            df_by_model_by_result=df_by_model_by_result,
            priors=inputs["priors"],
            unreported_plot_end_date=inputs["unreported_plot_end_date"],
            incidence_kwargs={
                "reported_incidence_vec": inputs["reported_incidence_vec"],
                "first_case_date": inputs["first_case_date"],
                "vline_days": vline_days,
            },
        )
        save_figure(fig=fig, paths=inputs["fig_paths"][fig_name])


def _make_figure(
    *,
    param_names,
    unreported,
    risk_types,
    df_by_model_by_result,
    priors,
    unreported_plot_end_date,
    incidence_kwargs,
):
    # A column per density panel (but at least two), with the densities and model
    # probabilities a column wide and the unreported cases and risks filling the rest of
    # their row. R_after's densities leave the upper right of their panel empty, so model
    # probabilities following them go in a box there instead, labelled as a panel of its own
    probabilities_inset = param_names[-1:] == ("rep_no_after",)
    n_narrow = len(param_names) + (bool(param_names) and not probabilities_inset)
    n_wide = unreported + bool(risk_types)
    n_labels = n_narrow + n_wide + probabilities_inset
    labels = (
        [f"{chr(ord('A') + idx)}." for idx in range(n_labels)]
        if n_labels > 1
        else [None]
    )
    inset_label = labels.pop(len(param_names)) if probabilities_inset else None
    fig, axes = make_panel_figure(
        panel_widths=[1] * n_narrow + [None] * n_wide,
        n_columns=max(2, len(param_names)),
        labels=labels,
    )

    for idx, (ax, param_name) in enumerate(
        zip(axes[: len(param_names)], param_names, strict=True)
    ):
        plot_densities(
            ax=ax,
            posterior_df_by_model=df_by_model_by_result["posterior"],
            priors=priors,
            param_name=param_name,
            log_x=(param_name == "dispersion"),
            legend=(idx == 0),
        )
    other_axes = iter(axes[len(param_names) :])
    if param_names:
        model_probability_by_model = compute_model_probabilities(
            log_evidence_by_model={
                model_name: evidence_df["log_evidence"].iloc[0]
                for model_name, evidence_df in df_by_model_by_result["evidence"].items()
            },
            prior_probability_by_model=None,
        )
        if probabilities_inset:
            plot_model_probabilities_inset(
                ax=axes[len(param_names) - 1],
                model_probability_by_model=model_probability_by_model,
                label=inset_label,
            )
        else:
            plot_model_probabilities(
                ax=next(other_axes),
                model_probability_by_model=model_probability_by_model,
            )
    if unreported:
        plot_unreported_incidence(
            ax=next(other_axes),
            unreported_pmf_df_by_model=df_by_model_by_result["unreported_incidence"],
            end_date=unreported_plot_end_date,
            **incidence_kwargs,
        )
    if risk_types:
        plot_risk_incidence_overlay(
            ax=next(other_axes),
            risk_df_by_model=df_by_model_by_result["risk"],
            risk_types=risk_types,
            **incidence_kwargs,
        )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate model comparison plots for a specified comparison spec."
    )
    parser.add_argument("--spec", required=True, choices=COMPARISON_SPECS)
    args = parser.parse_args()

    make_plots(comparison_spec_name=args.spec)
