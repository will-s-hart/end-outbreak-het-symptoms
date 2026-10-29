# Note that AI tools were used to generate tests
import numpy as np
import pandas as pd
import pytest

from scripts import delay_distributions_plots
from scripts.inputs import get_inputs_delay_distributions_plots


def test_make_plots_writes_figures(*, tmp_path, monkeypatch):
    # Real inputs, with the results and figures redirected into tmp_path and toy delay
    # distributions written there
    inputs = get_inputs_delay_distributions_plots()
    inputs["delay_distributions_results_paths"] = {
        delay_name: tmp_path / path
        for delay_name, path in inputs["delay_distributions_results_paths"].items()
    }
    inputs["fig_paths"] = {
        fig_name: [tmp_path / path for path in paths]
        for fig_name, paths in inputs["fig_paths"].items()
    }
    first_lag_by_delay = {"serial_interval": 1, "inc_period": 1, "tost": 0}
    for delay_name, path in inputs["delay_distributions_results_paths"].items():
        path.parent.mkdir(parents=True, exist_ok=True)
        first_lag = first_lag_by_delay[delay_name]
        pd.DataFrame(
            {"lag": np.arange(first_lag, first_lag + 4), "prob": [0.1, 0.4, 0.3, 0.2]}
        ).to_csv(path, index=False)
    monkeypatch.setattr(
        delay_distributions_plots,
        "get_inputs_delay_distributions_plots",
        lambda: inputs,
    )

    delay_distributions_plots.make_plots()

    for paths in inputs["fig_paths"].values():
        for path in paths:
            assert path.stat().st_size > 0


def test_implied_serial_interval_df():
    # TOST of 0 or 1 days and incubation period of 1 or 2 days, each equally likely, so
    # the serial interval is 1, 2 or 3 days with probabilities 1/4, 1/2, 1/4
    implied_df = delay_distributions_plots._implied_serial_interval_df(
        tost_df=pd.DataFrame({"lag": [0, 1], "prob": [0.5, 0.5]}),
        inc_period_df=pd.DataFrame({"lag": [1, 2], "prob": [0.5, 0.5]}),
    )
    assert implied_df["lag"].tolist() == [1, 2, 3]
    np.testing.assert_allclose(implied_df["prob"], [0.25, 0.5, 0.25])


def test_implied_serial_interval_matches_naive_moments():
    # The TOST's mean and variance are those of the serial interval less those of the
    # incubation period, so the implied serial interval should nearly match the naive one
    delay_df_by_name = {
        delay_name: pd.read_csv(path)
        for delay_name, path in get_inputs_delay_distributions_plots()[
            "delay_distributions_results_paths"
        ].items()
    }
    implied_df = delay_distributions_plots._implied_serial_interval_df(
        tost_df=delay_df_by_name["tost"], inc_period_df=delay_df_by_name["inc_period"]
    )
    assert implied_df["prob"].sum() == pytest.approx(1)
    for moment in (1, 2):
        assert (
            implied_df["lag"] ** moment * implied_df["prob"]
        ).sum() == pytest.approx(
            (
                delay_df_by_name["serial_interval"]["lag"] ** moment
                * delay_df_by_name["serial_interval"]["prob"]
            ).sum(),
            rel=1e-3,
        )
