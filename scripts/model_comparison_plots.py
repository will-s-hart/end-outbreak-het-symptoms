import argparse

import pandas as pd

from endoutbreakhet import compute_model_probabilities
from endoutbreakhet.plotting import (
    make_panel_figure,
    plot_densities,
    plot_model_probabilities_inset,
    plot_risk_incidence_overlay,
    plot_unreported_incidence,
    save_figure,
    set_plot_style,
)
from scripts.inputs import COMPARISON_SPECS, get_inputs_model_comparison_plots

# Three density panels in a row are small, so their legend's line samples are shorter, to
# keep it clear of the densities
_THREE_DENSITY_LEGEND_KWARGS = {"handlelength": 1.2}


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
    # The densities share a row, and the unreported cases and risks each fill a row of
    # their own. R_after's densities leave the upper right of their panel empty, so they
    # come last, with the model probabilities in a box there, labelled as a panel of its
    # own. Three densities are in smaller panels, with a box to fit
    if param_names and param_names[-1] != "rep_no_after":
        raise ValueError(
            "The model probabilities go in a box on R_after's densities, so param_names "
            f"must end with rep_no_after: {param_names}"
        )
    n_densities = len(param_names)
    n_wide = unreported + bool(risk_types)
    n_labels = n_densities + bool(param_names) + n_wide
    labels = (
        [f"{chr(ord('A') + idx)}." for idx in range(n_labels)]
        if n_labels > 1
        else [None]
    )
    inset_label = labels.pop(n_densities) if param_names else None
    compact = n_densities > 2
    fig, axes = make_panel_figure(
        panels_per_row=[n_densities] * bool(param_names) + [1] * n_wide, labels=labels
    )

    # The legend goes on R_before's densities if there are any, as k's leave it no room and
    # R_after's have the box
    legend_idx = (
        param_names.index("rep_no_before") if "rep_no_before" in param_names else 0
    )
    for idx, (ax, param_name) in enumerate(
        zip(axes[:n_densities], param_names, strict=True)
    ):
        plot_densities(
            ax=ax,
            posterior_df_by_model=df_by_model_by_result["posterior"],
            priors=priors,
            param_name=param_name,
            log_x=(param_name == "dispersion"),
            legend=(idx == legend_idx),
            legend_kwargs=_THREE_DENSITY_LEGEND_KWARGS if compact else None,
        )
    other_axes = iter(axes[n_densities:])
    if param_names:
        model_probability_by_model = compute_model_probabilities(
            log_evidence_by_model={
                model_name: evidence_df["log_evidence"].iloc[0]
                for model_name, evidence_df in df_by_model_by_result["evidence"].items()
            },
            prior_probability_by_model=None,
        )
        plot_model_probabilities_inset(
            ax=axes[n_densities - 1],
            model_probability_by_model=model_probability_by_model,
            label=inset_label,
            compact=compact,
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
