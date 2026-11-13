import pathlib
from typing import Any

import numpy as np
import pandas as pd

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
# Sampler settings for the specs with every case reported, which nutpie samples. The high
# target acceptance stops NUTS diverging where the dispersion is estimated
FIT_KWARGS = {
    "chains": 4,
    "draws": 4000,
    "tune": 1000,
    "thin": 1,
    "target_accept": 0.999,
}
# For the under-reporting specs, which PyMC samples with NUTS for the continuous variables
# and Metropolis for the unreported counts. Metropolis mixes slowly, hence the extra draws,
# thinned to as many as the other specs keep at little cost, as the counts' draws are highly
# correlated. NUTS diverged occasionally at lower target acceptances
FIT_KWARGS_UNDER_REPORTING = FIT_KWARGS | {
    "draws": 12000,
    "thin": 3,
    "target_accept": 0.99,
}
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "poisson": {
        "model_name": "poisson",
        "priors": PRIORS_POISSON,
        "reporting_prob": 1.0,
        "fit_kwargs": FIT_KWARGS,
    },
    "poisson_so": {
        "model_name": "poisson_so",
        "priors": PRIORS_POISSON,
        "reporting_prob": 1.0,
        "fit_kwargs": FIT_KWARGS,
    },
    **{
        f"{model_name}_{dispersion_type}_dispersion": {
            "model_name": model_name,
            "priors": priors,
            "reporting_prob": 1.0,
            "fit_kwargs": FIT_KWARGS,
        }
        for model_name in ("sse", "ssi", "ssd", "sse_so", "ssi_so")
        for dispersion_type, priors in (
            ("fixed", PRIORS_FIXED_DISPERSION),
            ("estimated", PRIORS_ESTIMATED_DISPERSION),
        )
    },
    **{
        f"{model_name}_{round(100 * reporting_prob)}_pc_reporting": {
            "model_name": model_name,
            "priors": PRIORS_FIXED_DISPERSION,
            "reporting_prob": reporting_prob,
            "fit_kwargs": FIT_KWARGS_UNDER_REPORTING,
        }
        for model_name in ("sse", "ssi", "sse_so", "ssi_so")
        for reporting_prob in (0.6, 0.8)
    },
}
# Cheap, being independent draws made on the comparison day only, and fewer added
# appreciably to the under-reporting specs' log evidence SE
N_EVIDENCE_PROPOSAL_SAMPLES = 100_000
# Applied to the saved diagnostics by fit_diagnostics_plots only, so editing one never reruns
# a fit
FIT_GATE_THRESHOLDS = {
    "max_rhat": 1.01,
    # Over all chains, not per chain
    "min_ess_bulk": 400,
    "min_ess_tail": 400,
    # Any divergence means exploration may be biased, so loosen this only for rare
    # divergences whose cause is understood, recording it here
    "max_n_divergences": 0,
    # Draws whose trajectory hit NUTS's maximum tree depth: still valid transitions, but a
    # sign the step size is small for the posterior (likeliest at a high target_accept),
    # costing efficiency
    "max_n_max_treedepth": 0,
    # Absolute: one percentage point
    "max_risk_mcse": 0.01,
    # Bounds the standard error of every posterior model probability by half this, to first
    # order, whatever the comparison
    "max_log_evidence_se": 0.1,
}
FIG_EXTENSIONS = ("svg", "png")
DATA_PATH = pathlib.Path("data/equateur_2018_onsets.csv")

# Each comparison's figures, by name, each with any of these panels, in this order: the prior
# and posterior densities of each of param_names, which end with rep_no_after, whose panel
# has the model probabilities in a box; the unreported cases each day on the comparison day,
# if unreported; and the risks of risk_types
COMPARISON_SPECS: dict[str, dict[str, Any]] = {
    "naive_models_fixed_dispersion": {
        "model_spec_names": [
            "sse_fixed_dispersion",
            "ssi_fixed_dispersion",
            "ssd_fixed_dispersion",
        ],
        "figures": {
            "summary": {
                "param_names": ("rep_no_before", "rep_no_after"),
                "risk_types": ("cases",),
            },
        },
    },
    "naive_models_estimated_dispersion": {
        "model_spec_names": [
            "sse_estimated_dispersion",
            "ssi_estimated_dispersion",
            "ssd_estimated_dispersion",
        ],
        "figures": {
            "summary": {
                "param_names": ("dispersion", "rep_no_before", "rep_no_after"),
                "risk_types": ("cases",),
            },
        },
    },
    "naive_vs_so_fixed_dispersion": {
        "model_spec_names": [
            "sse_fixed_dispersion",
            "ssi_fixed_dispersion",
            "sse_so_fixed_dispersion",
            "ssi_so_fixed_dispersion",
        ],
        "figures": {
            "summary": {
                "param_names": ("rep_no_before", "rep_no_after"),
                "risk_types": ("cases", "transmission"),
            },
            "risk_sustained": {"risk_types": ("sustained",)},
        },
    },
    "naive_vs_so_estimated_dispersion": {
        "model_spec_names": [
            "sse_estimated_dispersion",
            "ssi_estimated_dispersion",
            "sse_so_estimated_dispersion",
            "ssi_so_estimated_dispersion",
        ],
        "figures": {
            "summary": {
                "param_names": ("dispersion", "rep_no_before", "rep_no_after"),
                "risk_types": ("cases", "transmission"),
            },
        },
    },
    "poisson_vs_superspreading": {
        "model_spec_names": [
            "poisson",
            "poisson_so",
            "ssi_fixed_dispersion",
            "ssi_so_fixed_dispersion",
        ],
        "figures": {
            "summary": {
                "param_names": ("rep_no_before", "rep_no_after"),
                "risk_types": ("cases", "transmission"),
            },
        },
    },
    "naive_vs_so_60_pc_reporting": {
        "model_spec_names": [
            "sse_60_pc_reporting",
            "ssi_60_pc_reporting",
            "sse_so_60_pc_reporting",
            "ssi_so_60_pc_reporting",
        ],
        "figures": {
            "summary": {
                "param_names": ("rep_no_before", "rep_no_after"),
                "unreported": True,
                "risk_types": ("cases", "transmission"),
            },
        },
    },
    "naive_vs_so_80_pc_reporting": {
        "model_spec_names": [
            "sse_80_pc_reporting",
            "ssi_80_pc_reporting",
            "sse_so_80_pc_reporting",
            "ssi_so_80_pc_reporting",
        ],
        "figures": {
            "unreported_risk": {
                "unreported": True,
                "risk_types": ("cases", "transmission"),
            },
            "posterior": {
                "param_names": ("rep_no_before", "rep_no_after"),
            },
            "risk_cases_reports": {
                "risk_types": ("cases", "reports"),
            },
        },
    },
}


def get_inputs_delay_distributions():
    serial_interval_cont_params = {
        "mean": 15.3,
        "sd": 9.3,
        "max": 120,
    }
    inc_period_cont_params = {"mean": 11.4, "sd": 8.1, "max": 120}
    tost_cont_params = {
        "mean": serial_interval_cont_params["mean"] - inc_period_cont_params["mean"],
        "sd": (
            serial_interval_cont_params["sd"] ** 2 - inc_period_cont_params["sd"] ** 2
        )
        ** 0.5,
        "max": 120,
    }
    return {
        "cont_params_by_delay": {
            "serial_interval": serial_interval_cont_params,
            "inc_period": inc_period_cont_params,
            "tost": tost_cont_params,
        },
        "results_paths": get_results_paths_delay_distributions(),
    }


def get_inputs_model_fits(*, model_spec_name):
    model_spec = MODEL_SPECS[model_spec_name]
    return {
        **_get_shared_inputs(),
        # Paths rather than the distributions themselves, so the fits depend on the
        # delay_distributions results as files
        "delay_distributions_results_paths": get_results_paths_delay_distributions(),
        "model_name": model_spec["model_name"],
        "priors": model_spec["priors"],
        "reporting_prob": model_spec["reporting_prob"],
        "fit_kwargs": model_spec["fit_kwargs"],
        "n_evidence_proposal_samples": N_EVIDENCE_PROPOSAL_SAMPLES,
        # A stream of its own for each spec (keyed by name, so reordering the specs changes
        # nothing), so specs' log evidence errors are independent
        "rng": np.random.default_rng(seed=[42, *model_spec_name.encode()]),
        "results_paths": get_results_paths_model_fits(model_spec_name=model_spec_name),
    }


def get_inputs_overview_plots():
    cont_params_by_delay = get_inputs_delay_distributions()["cont_params_by_delay"]
    return {
        **_get_shared_inputs(),
        # To which the schematic of the delays is drawn to scale
        "delay_means": {
            delay_name: cont_params["mean"]
            for delay_name, cont_params in cont_params_by_delay.items()
        },
        "fig_paths": get_fig_paths_overview_plots(),
    }


def get_inputs_delay_distributions_plots():
    return {
        "delay_distributions_results_paths": get_results_paths_delay_distributions(),
        # Shared by the panels so they can be compared, and past the 99th percentile of
        # every delay
        "max_plot_lag": 50,
        "fig_paths": get_fig_paths_delay_distributions_plots(),
    }


def get_inputs_model_comparison_plots(*, comparison_spec_name):
    comparison_spec = COMPARISON_SPECS[comparison_spec_name]
    model_specs = {
        model_spec_name: MODEL_SPECS[model_spec_name]
        for model_spec_name in comparison_spec["model_spec_names"]
    }
    figures: dict[str, dict[str, Any]] = {
        fig_name: {"param_names": (), "unreported": False, "risk_types": (), **figure}
        for fig_name, figure in comparison_spec["figures"].items()
    }

    # Only the results the figures plot, keyed by model name, which the plots use to style
    # each model
    result_names = set()
    for figure in figures.values():
        if figure["param_names"]:
            result_names |= {"posterior", "evidence"}
        if figure["unreported"]:
            result_names.add("unreported_incidence")
        if figure["risk_types"]:
            result_names.add("risk")
    results_paths_by_model = {
        model_spec["model_name"]: {
            result_name: path
            for result_name, path in get_results_paths_model_fits(
                model_spec_name=model_spec_name
            ).items()
            if result_name in result_names
        }
        for model_spec_name, model_spec in model_specs.items()
    }
    # Each density plot shows one prior, so the specs must agree on it
    priors = {}
    for param_name in dict.fromkeys(
        param_name
        for figure in figures.values()
        for param_name in figure["param_names"]
    ):
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
        "figures": figures,
        # A month past the last reported case, by when every model's unreported cases are
        # near zero
        "unreported_plot_end_date": pd.Timestamp("2018-07-01"),
        "fig_paths": get_fig_paths_model_comparison_plots(
            comparison_spec_name=comparison_spec_name
        ),
    }


def get_inputs_fit_diagnostics_plots():
    return {
        **_get_shared_inputs(),
        "results_paths_by_spec": {
            model_spec_name: get_results_paths_model_fits(
                model_spec_name=model_spec_name
            )
            for model_spec_name in MODEL_SPECS
        },
        "reporting_prob_by_spec": {
            model_spec_name: model_spec["reporting_prob"]
            for model_spec_name, model_spec in MODEL_SPECS.items()
        },
        # Constant by construction, so left out of the gate
        "fixed_param_names_by_spec": {
            model_spec_name: [
                param_name
                for param_name, prior in model_spec["priors"].items()
                if prior.get("dist") == "fixed"
            ]
            for model_spec_name, model_spec in MODEL_SPECS.items()
        },
        "thresholds": FIT_GATE_THRESHOLDS,
        "fig_paths": get_fig_paths_fit_diagnostics_plots(),
    }


def get_results_paths_delay_distributions():
    results_dir = pathlib.Path("results/delay_distributions")
    return {
        delay_name: results_dir / f"{delay_name}.csv"
        for delay_name in ("serial_interval", "inc_period", "tost")
    }


def get_results_paths_model_fits(*, model_spec_name):
    results_dir = pathlib.Path("results/model_fits") / model_spec_name
    return {
        "risk": results_dir / "risk.csv",
        "diagnostics": results_dir / "diagnostics.csv",
        "evidence": results_dir / "evidence.csv",
        "posterior": results_dir / "posterior.csv",
        # Each day's posterior pmf of the unreported cases
        "unreported_incidence": results_dir / "unreported_incidence.csv",
        "trace": results_dir / "trace.nc",
    }


def get_fig_paths_overview_plots():
    # One path per file format for the figure
    fig_dir = pathlib.Path("figures/overview")
    return {"summary": [fig_dir / f"summary.{ext}" for ext in FIG_EXTENSIONS]}


def get_fig_paths_delay_distributions_plots():
    # One path per file format for the figure
    fig_dir = pathlib.Path("figures/delay_distributions")
    return {"summary": [fig_dir / f"summary.{ext}" for ext in FIG_EXTENSIONS]}


def get_fig_paths_model_comparison_plots(*, comparison_spec_name):
    # One path per file format for each figure
    fig_dir = pathlib.Path("figures/model_comparison") / comparison_spec_name
    return {
        fig_name: [fig_dir / f"{fig_name}.{ext}" for ext in FIG_EXTENSIONS]
        for fig_name in COMPARISON_SPECS[comparison_spec_name]["figures"]
    }


def get_fig_paths_fit_diagnostics_plots():
    # One path per file format for the figure
    fig_dir = pathlib.Path("figures/fit_diagnostics")
    return {"gate": [fig_dir / f"gate.{ext}" for ext in FIG_EXTENSIONS]}


def get_fig_paths_numbered(*, figure_number):
    # One path per file format for the numbered copy of a figure
    return [
        pathlib.Path("figures") / f"figure_{figure_number}.{ext}"
        for ext in FIG_EXTENSIONS
    ]


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

    return {
        "reported_incidence_vec": reported_incidence_vec,
        "first_case_date": first_case_date,
        "ert_days": ert_days,
        "switch_day": switch_day,
        "comparison_day": comparison_day,
        "risk_days": risk_days,
    }
