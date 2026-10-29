import argparse

import numpy as np
import pandas as pd

from endoutbreakhet import discretise_cori, gamma_dist_from_mean_sd
from scripts.inputs import get_inputs_delay_distributions


def run_analyses():
    inputs = get_inputs_delay_distributions()
    for delay_name, cont_params in inputs["cont_params_by_delay"].items():
        prob_vec = discretise_cori(
            dist_cont=gamma_dist_from_mean_sd(
                mean=cont_params["mean"], sd=cont_params["sd"]
            ),
            max_val=cont_params["max"],
            allow_zero=cont_params["allow_zero"],
        )
        first_lag = 0 if cont_params["allow_zero"] else 1
        path = inputs["results_paths"][delay_name]
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"lag": np.arange(first_lag, first_lag + len(prob_vec)), "prob": prob_vec}
        ).to_csv(path, index=False)


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=(
            "Discretise the serial interval, incubation period and TOST distributions."
        )
    ).parse_args()

    run_analyses()
