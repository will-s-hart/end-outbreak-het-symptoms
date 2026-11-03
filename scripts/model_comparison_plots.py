import argparse

import pandas as pd

from endoutbreakhet import compute_model_probabilities
from endoutbreakhet.plotting import (
    make_panel_figure,
    plot_densities,
    plot_model_probabilities,
    plot_risk_incidence_overlay,
    save_figure,
    set_plot_style,
)
from scripts.inputs import COMPARISON_SPECS, get_inputs_model_comparison_plots


def make_plots(*, comparison_spec_name):
    set_plot_style()
    inputs = get_inputs_model_comparison_plots(
        comparison_spec_name=comparison_spec_name
    )

    results_paths_by_model = inputs["results_paths_by_model"]
    fig_paths = inputs["fig_paths"]
    param_names = inputs["param_names"]
    posterior_df_by_model = {
        model_name: pd.read_csv(paths["posterior"])
        for model_name, paths in results_paths_by_model.items()
    }
    log_evidence_by_model = {
        model_name: pd.read_csv(paths["evidence"])["log_evidence"].iloc[0]
        for model_name, paths in results_paths_by_model.items()
    }
    risk_overlay_kwargs = {
        "risk_df_by_model": {
            model_name: pd.read_csv(paths["risk"])
            for model_name, paths in results_paths_by_model.items()
        },
        "reported_incidence_vec": inputs["reported_incidence_vec"],
        "first_case_date": inputs["first_case_date"],
        "vline_days": [inputs["ert_days"]["arrival"], inputs["ert_days"]["departure"]],
    }

    # The risks fill the rest of the last row
    summary_fig, summary_axes = make_panel_figure(
        panel_widths=[1] * (len(param_names) + 1) + [None]
    )
    *densities_axes, model_probabilities_ax, risk_ax = summary_axes
    for ax, param_name in zip(densities_axes, param_names, strict=True):
        plot_densities(
            ax=ax,
            posterior_df_by_model=posterior_df_by_model,
            priors=inputs["priors"],
            param_name=param_name,
            log_x=(param_name == "dispersion"),
            legend=(param_name == "rep_no_before"),
        )
    plot_model_probabilities(
        ax=model_probabilities_ax,
        model_probability_by_model=compute_model_probabilities(
            log_evidence_by_model=log_evidence_by_model,
            prior_probability_by_model=None,
        ),
    )
    plot_risk_incidence_overlay(
        ax=risk_ax, risk_types=inputs["risk_types"], **risk_overlay_kwargs
    )
    save_figure(fig=summary_fig, paths=fig_paths["summary"])
    # Each extra risk panel in a figure of its own
    for risk_types in inputs["extra_risk_panels"]:
        risk_fig, (risk_ax,) = make_panel_figure(panel_widths=[2])
        plot_risk_incidence_overlay(
            ax=risk_ax, risk_types=risk_types, **risk_overlay_kwargs
        )
        save_figure(fig=risk_fig, paths=fig_paths[f"risk_{'_'.join(risk_types)}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate model comparison plots for a specified comparison spec."
    )
    parser.add_argument("--spec", required=True, choices=COMPARISON_SPECS)
    args = parser.parse_args()

    make_plots(comparison_spec_name=args.spec)
