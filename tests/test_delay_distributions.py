# Note that AI tools were used to generate tests
import numpy as np
import pandas as pd
import pytest

from endoutbreakhet import discretise_cori, gamma_dist_from_mean_sd
from scripts import delay_distributions
from scripts.inputs import get_inputs_delay_distributions


def test_run_analyses_writes_discretised_delays(*, tmp_path, monkeypatch):
    # Real inputs, with the results redirected into tmp_path
    inputs = get_inputs_delay_distributions()
    inputs["results_paths"] = {
        delay_name: tmp_path / path
        for delay_name, path in inputs["results_paths"].items()
    }
    monkeypatch.setattr(
        delay_distributions, "get_inputs_delay_distributions", lambda: inputs
    )

    delay_distributions.run_analyses()

    first_lag_by_delay = {}
    for delay_name, cont_params in inputs["cont_params_by_delay"].items():
        df = pd.read_csv(
            inputs["results_paths"][delay_name], float_precision="round_trip"
        )
        first_lag = df["lag"].iloc[0]
        first_lag_by_delay[delay_name] = first_lag
        assert df["lag"].tolist() == list(range(first_lag, first_lag + len(df)))
        assert df["prob"].sum() == pytest.approx(1)
        # Read back exactly, as the fits read it
        np.testing.assert_array_equal(
            df["prob"].to_numpy(),
            discretise_cori(
                dist_cont=gamma_dist_from_mean_sd(
                    mean=cont_params["mean"], sd=cont_params["sd"]
                ),
                max_val=cont_params["max"],
                allow_zero=cont_params["allow_zero"],
            ),
        )
    assert first_lag_by_delay == {"serial_interval": 1, "inc_period": 1, "tost": 0}
