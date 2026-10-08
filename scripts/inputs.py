import pandas as pd

from endoutbreakhet.utils import discretise_cori, gamma_dist_from_mean_sd


def get_inputs():
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
    final_analysis_day = ert_departure_day + 20

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

    priors = {
        "rep_no_before": {"dist": "lognormal", "median": 1.0, "quantile_025": 0.2},
        "rep_no_after": {"dist": "lognormal", "median": 1.0, "quantile_025": 0.2},
        "dispersion": {"dist": "lognormal", "median": 0.18, "quantile_025": 0.018},
    }

    inputs = {
        "incidence_vec": incidence_vec,
        "ert_days": ert_days,
        "switch_day": switch_day,
        "final_analysis_day": final_analysis_day,
        "delay_distributions": delay_distributions,
        "priors": priors,
    }
    return inputs
