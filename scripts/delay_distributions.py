import argparse

from endoutbreakhet import (
    DELAY_FIRST_LAGS,
    DelayDistribution,
    discretise_cori,
    gamma_dist_from_mean_sd,
)
from scripts.inputs import get_inputs_delay_distributions


def run_analyses():
    inputs = get_inputs_delay_distributions()
    for delay_name, cont_params in inputs["cont_params_by_delay"].items():
        first_lag = DELAY_FIRST_LAGS[delay_name]
        prob_vec = discretise_cori(
            dist_cont=gamma_dist_from_mean_sd(
                mean=cont_params["mean"], sd=cont_params["sd"]
            ),
            max_val=cont_params["max"],
            allow_zero=first_lag == 0,
        )
        path = inputs["results_paths"][delay_name]
        path.parent.mkdir(parents=True, exist_ok=True)
        DelayDistribution(prob=prob_vec, first_lag=first_lag).to_df().to_csv(
            path, index=False
        )


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=(
            "Discretise the serial interval, incubation period and TOST distributions."
        )
    ).parse_args()

    run_analyses()
