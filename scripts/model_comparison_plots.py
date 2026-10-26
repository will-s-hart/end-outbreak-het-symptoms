import argparse

import pandas as pd

from endoutbreakhet import compute_model_probabilities
from endoutbreakhet.plotting import (
    plot_densities,
    plot_model_probabilities,
    plot_risk_incidence_overlay,
    save_figure,
)
from scripts.inputs import COMPARISON_SPECS, get_inputs_model_comparison_plots


def make_plots(*, comparison_spec_name):
    inputs = get_inputs_model_comparison_plots(
        comparison_spec_name=comparison_spec_name
    )

    results_paths_by_model = inputs["results_paths_by_model"]
    fig_paths = inputs["fig_paths"]
    # KDEs of the posterior density of each parameter, with its prior density
    posterior_df_by_model = {
        model_name: pd.read_csv(paths["posterior"])
        for model_name, paths in results_paths_by_model.items()
    }
    for param_name in inputs["param_names"]:
        densities_fig = plot_densities(
            posterior_df_by_model=posterior_df_by_model,
            priors=inputs["priors"],
            param_name=param_name,
            log_x=(param_name == "dispersion"),
            legend=(param_name == "rep_no_before"),
        )
        save_figure(fig=densities_fig, paths=fig_paths[f"densities_{param_name}"])
    # Pie chart of posterior model probabilities
    log_evidence_by_model = {
        model_name: pd.read_csv(paths["evidence"])["log_evidence"].iloc[0]
        for model_name, paths in results_paths_by_model.items()
    }
    model_probabilities_fig = plot_model_probabilities(
        model_probability_by_model=compute_model_probabilities(
            log_evidence_by_model=log_evidence_by_model,
            prior_probability_by_model=None,
        )
    )
    save_figure(fig=model_probabilities_fig, paths=fig_paths["model_probabilities"])
    # Overlay of risk and incidence curves
    risk_df_by_model = {
        model_name: pd.read_csv(paths["risk"])
        for model_name, paths in results_paths_by_model.items()
    }
    risk_incidence_overlay_fig = plot_risk_incidence_overlay(
        risk_df_by_model=risk_df_by_model,
        reported_incidence_vec=inputs["reported_incidence_vec"],
        first_case_date=inputs["first_case_date"],
        risk_types=inputs["risk_types"],
        vline_days=[inputs["ert_days"]["arrival"], inputs["ert_days"]["departure"]],
    )
    save_figure(
        fig=risk_incidence_overlay_fig, paths=fig_paths["risk_incidence_overlay"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate model comparison plots for a specified comparison spec."
    )
    parser.add_argument("--spec", required=True, choices=COMPARISON_SPECS)
    args = parser.parse_args()

    make_plots(comparison_spec_name=args.spec)
