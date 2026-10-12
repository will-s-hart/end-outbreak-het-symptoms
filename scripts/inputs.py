import pathlib

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
MODEL_SPECS = {
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
N_EVIDENCE_PROPOSAL_SAMPLES = 1000


def get_inputs_model_fits(*, spec_name):
    spec = MODEL_SPECS[spec_name]

    df_data = pd.read_csv("data/equateur_2018_onsets.csv", parse_dates=["date"])
    incidence_vec = df_data["incidence"].to_numpy()

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

    results_dir = pathlib.Path("results/model_fits") / spec_name
    results_paths = {
        "risk": results_dir / "risk.csv",
        "evidence": results_dir / "evidence.csv",
        "posterior": results_dir / "posterior.csv",
        "trace": results_dir / "trace.nc",
    }

    inputs = {
        "incidence_vec": incidence_vec,
        "ert_days": ert_days,
        "switch_day": switch_day,
        "comparison_day": comparison_day,
        "risk_days": risk_days,
        "delay_distributions": delay_distributions,
        "model_name": spec["model_name"],
        "priors": spec["priors"],
        "fit_kwargs": FIT_KWARGS,
        "n_evidence_proposal_samples": N_EVIDENCE_PROPOSAL_SAMPLES,
        "rng": np.random.default_rng(seed=42),
        "results_paths": results_paths,
    }
    return inputs
