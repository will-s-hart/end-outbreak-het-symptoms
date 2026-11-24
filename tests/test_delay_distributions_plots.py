# Note that AI tools were used to generate tests
import numpy as np
import pandas as pd
import pytest

from endoutbreakhet import DELAY_FIRST_LAGS, DelayDistribution
from scripts import delay_distributions_plots
from scripts.inputs import get_inputs_delay_distributions_plots


def test_make_plots_writes_figures(*, tmp_path, monkeypatch):
    # Real inputs, with the results and figure redirected into tmp_path and toy delay
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
    for delay_name, path in inputs["delay_distributions_results_paths"].items():
        path.parent.mkdir(parents=True, exist_ok=True)
        DelayDistribution(
            prob=np.array([0.1, 0.4, 0.3, 0.2]), first_lag=DELAY_FIRST_LAGS[delay_name]
        ).to_df().to_csv(path, index=False)
    monkeypatch.setattr(
        delay_distributions_plots,
        "get_inputs_delay_distributions_plots",
        lambda: inputs,
    )

    delay_distributions_plots.make_plots()

    for paths in inputs["fig_paths"].values():
        for path in paths:
            assert path.stat().st_size > 0


def test_implied_serial_interval():
    # TOST of 0 or 1 days and incubation period of 1 or 2 days, each equally likely, so
    # the serial interval is 1, 2 or 3 days with probabilities 1/4, 1/2, 1/4
    implied = delay_distributions_plots._implied_serial_interval(
        tost=DelayDistribution(prob=np.array([0.5, 0.5]), first_lag=0),
        inc_period=DelayDistribution(prob=np.array([0.5, 0.5]), first_lag=1),
    )
    np.testing.assert_array_equal(implied.lags, [1, 2, 3])
    np.testing.assert_allclose(implied.prob, [0.25, 0.5, 0.25])


def test_implied_serial_interval_matches_input_moments():
    # The TOST's mean and variance are those of the serial interval less those of the
    # incubation period, so the implied serial interval should nearly match the input one
    delay_by_name = {
        delay_name: DelayDistribution.from_df(df=pd.read_csv(path))
        for delay_name, path in get_inputs_delay_distributions_plots()[
            "delay_distributions_results_paths"
        ].items()
    }
    implied = delay_distributions_plots._implied_serial_interval(
        tost=delay_by_name["tost"], inc_period=delay_by_name["inc_period"]
    )
    serial_interval = delay_by_name["serial_interval"]
    for moment in (1, 2):
        assert np.sum(implied.lags**moment * implied.prob) == pytest.approx(
            np.sum(serial_interval.lags**moment * serial_interval.prob), rel=1e-3
        )
