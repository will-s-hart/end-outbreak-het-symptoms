import pathlib
from typing import Any

import numpy as np
import pandas as pd

from endoutbreakhet import discretise_cori, gamma_dist_from_mean_sd

PRIORS_POISSON = {
    "rep_no_before": {"dist": "lognormal", "median": 1.0, "quantile_025": 0.2},
    "rep_no_after": {"dist": "lognormal", "median": 1.0, "quantile_025": 0.2},
}
PRIORS_FIXED_DISPERSION = {
    **PRIORS_POISSON,
    "dispersion": {"dist": "fixed", "value": 0.18},
}
PRIORS_ESTIMATED_DISPERSION = {
    **PRIORS_POISSON,
    "dispersion": {"dist": "lognormal", "median": 0.18, "quantile_025": 0.018},
}
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "poisson": {
        "model_name": "poisson",
        "priors": PRIORS_POISSON,
    },
    "dlo_fixed_dispersion": {
        "model_name": "dlo",
        "priors": PRIORS_FIXED_DISPERSION,
    },
    "dlo_estimated_dispersion": {
        "model_name": "dlo",
        "priors": PRIORS_ESTIMATED_DISPERSION,
    },
    "sse_fixed_dispersion": {
        "model_name": "sse",
        "priors": PRIORS_FIXED_DISPERSION,
    },
    "sse_estimated_dispersion": {
        "model_name": "sse",
        "priors": PRIORS_ESTIMATED_DISPERSION,
    },
    "ssi_fixed_dispersion": {
        "model_name": "ssi",
        "priors": PRIORS_FIXED_DISPERSION,
    },
    "ssi_estimated_dispersion": {
        "model_name": "ssi",
        "priors": PRIORS_ESTIMATED_DISPERSION,
    },
    "poisson_so": {
        "model_name": "poisson_so",
        "priors": PRIORS_POISSON,
    },
    "sse_so_fixed_dispersion": {
        "model_name": "sse_so",
        "priors": PRIORS_FIXED_DISPERSION,
    },
    "sse_so_estimated_dispersion": {
        "model_name": "sse_so",
        "priors": PRIORS_ESTIMATED_DISPERSION,
    },
    "ssi_so_fixed_dispersion": {
        "model_name": "ssi_so",
        "priors": PRIORS_FIXED_DISPERSION,
    },
    "ssi_so_estimated_dispersion": {
        "model_name": "ssi_so",
        "priors": PRIORS_ESTIMATED_DISPERSION,
    },
}
FIT_KWARGS = {
    "chains": 4,
    "draws": 1000,
    "tune": 1000,
    "thin": 1,
    "target_accept": 0.9,
}
N_EVIDENCE_PROPOSAL_SAMPLES = 10_000
FIG_EXTENSIONS = ("svg", "png")
DATA_PATH = pathlib.Path("data/equateur_2018_onsets.csv")

COMPARISON_SPECS = {
    "naive_models_fixed_dispersion": {
        "model_spec_names": [
            "dlo_fixed_dispersion",
            "sse_fixed_dispersion",
            "ssi_fixed_dispersion",
        ],
        "risk_types": ("cases",),
        "param_names": ("rep_no_before", "rep_no_after"),
    },
    "naive_models_estimated_dispersion": {
        "model_spec_names": [
            "dlo_estimated_dispersion",
            "sse_estimated_dispersion",
            "ssi_estimated_dispersion",
        ],
        "risk_types": ("cases",),
        "param_names": ("rep_no_before", "rep_no_after", "dispersion"),
    },
    "naive_vs_so_fixed_dispersion": {
        "model_spec_names": [
            "sse_fixed_dispersion",
            "ssi_fixed_dispersion",
            "sse_so_fixed_dispersion",
            "ssi_so_fixed_dispersion",
        ],
        "risk_types": ("cases", "transmission"),
        "param_names": ("rep_no_before", "rep_no_after"),
    },
    "naive_vs_so_estimated_dispersion": {
        "model_spec_names": [
            "sse_estimated_dispersion",
            "ssi_estimated_dispersion",
            "sse_so_estimated_dispersion",
            "ssi_so_estimated_dispersion",
        ],
        "risk_types": ("cases", "transmission"),
        "param_names": ("rep_no_before", "rep_no_after", "dispersion"),
    },
    "poisson_vs_superspreading": {
        "model_spec_names": [
            "poisson",
            "poisson_so",
            "ssi_fixed_dispersion",
            "ssi_so_fixed_dispersion",
        ],
        "risk_types": ("cases", "transmission"),
        "param_names": ("rep_no_before", "rep_no_after"),
    },
}


def get_inputs_model_fits(*, model_spec_name):
    model_spec = MODEL_SPECS[model_spec_name]
    return {
        **_get_shared_inputs(),
        "model_name": model_spec["model_name"],
        "priors": model_spec["priors"],
        "fit_kwargs": FIT_KWARGS,
        "n_evidence_proposal_samples": N_EVIDENCE_PROPOSAL_SAMPLES,
        "rng": np.random.default_rng(seed=42),
        "results_paths": get_results_paths_model_fits(model_spec_name=model_spec_name),
    }


def get_inputs_model_comparison_plots(*, comparison_spec_name):
    comparison_spec = COMPARISON_SPECS[comparison_spec_name]
    model_specs = {
        model_spec_name: MODEL_SPECS[model_spec_name]
        for model_spec_name in comparison_spec["model_spec_names"]
    }

    # Keyed by model name, which the plots use to style each model
    results_paths_by_model = {
        model_spec["model_name"]: get_results_paths_model_fits(
            model_spec_name=model_spec_name
        )
        for model_spec_name, model_spec in model_specs.items()
    }
    # Each density plot shows one prior, so the specs must agree on it
    priors = {}
    for param_name in comparison_spec["param_names"]:
        param_priors = [
            model_spec["priors"][param_name]
            for model_spec in model_specs.values()
            if param_name in model_spec["priors"]
        ]
        if any(prior != param_priors[0] for prior in param_priors):
            raise ValueError(
                f"Model specs in {comparison_spec_name} have different priors for "
                f"{param_name}"
            )
        priors[param_name] = param_priors[0]

    return {
        **_get_shared_inputs(),
        "results_paths_by_model": results_paths_by_model,
        "priors": priors,
        "param_names": comparison_spec["param_names"],
        "risk_types": comparison_spec["risk_types"],
        "fig_paths": get_fig_paths_model_comparison_plots(
            comparison_spec_name=comparison_spec_name
        ),
    }


def get_results_paths_model_fits(*, model_spec_name):
    results_dir = pathlib.Path("results/model_fits") / model_spec_name
    return {
        "risk": results_dir / "risk.csv",
        "evidence": results_dir / "evidence.csv",
        "posterior": results_dir / "posterior.csv",
        "trace": results_dir / "trace.nc",
    }


def get_fig_paths_model_comparison_plots(*, comparison_spec_name):
    # One path per file format for each figure
    fig_dir = pathlib.Path("figures/model_comparison") / comparison_spec_name
    fig_names = [
        *(
            f"densities_{param_name}"
            for param_name in COMPARISON_SPECS[comparison_spec_name]["param_names"]
        ),
        "model_probabilities",
        "risk_incidence_overlay",
    ]
    return {
        fig_name: [fig_dir / f"{fig_name}.{ext}" for ext in FIG_EXTENSIONS]
        for fig_name in fig_names
    }


def _get_shared_inputs():
    df_data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    reported_incidence_vec = df_data["incidence"].to_numpy()

    first_case_date = df_data["date"].min()
    ert_arrival_date = pd.to_datetime("2018-05-8")
    ert_departure_date = pd.to_datetime("2018-07-24")
    ert_arrival_day = (ert_arrival_date - first_case_date).days
    ert_departure_day = (ert_departure_date - first_case_date).days
    ert_days = {
        "arrival": ert_arrival_day,
        "departure": ert_departure_day,
    }

    switch_day = ert_arrival_day
    comparison_day = ert_departure_day
    risk_days = np.arange(switch_day, ert_departure_day + 20)

    serial_interval_cont_params = {"mean": 15.3, "sd": 9.3, "max": 150}
    inc_period_cont_params = {"mean": 11.4, "sd": 8.1, "max": 150}
    tost_cont_params = {
        "mean": serial_interval_cont_params["mean"] - inc_period_cont_params["mean"],
        "sd": (
            serial_interval_cont_params["sd"] ** 2 - inc_period_cont_params["sd"] ** 2
        )
        ** 0.5,
        "max": 150,
    }
    delay_distributions = {
        "serial_interval": discretise_cori(
            dist_cont=gamma_dist_from_mean_sd(
                mean=serial_interval_cont_params["mean"],
                sd=serial_interval_cont_params["sd"],
            ),
            max_val=serial_interval_cont_params["max"],
        ),
        "inc_period": discretise_cori(
            dist_cont=gamma_dist_from_mean_sd(
                mean=inc_period_cont_params["mean"], sd=inc_period_cont_params["sd"]
            ),
            max_val=inc_period_cont_params["max"],
        ),
        "tost": discretise_cori(
            dist_cont=gamma_dist_from_mean_sd(
                mean=tost_cont_params["mean"], sd=tost_cont_params["sd"]
            ),
            max_val=tost_cont_params["max"],
            allow_zero=True,
        ),
    }

    return {
        "reported_incidence_vec": reported_incidence_vec,
        "first_case_date": first_case_date,
        "ert_days": ert_days,
        "switch_day": switch_day,
        "comparison_day": comparison_day,
        "risk_days": risk_days,
        "delay_distributions": delay_distributions,
    }
