# Note that AI tools were used to generate tests
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from endoutbreakhet import DELAY_FIRST_LAGS, DelayDistribution, compute_fit_diagnostics
from scripts import model_fits
from scripts.inputs import MODEL_SPECS, get_inputs_model_fits


def test_run_analyses_passes_delays_and_reporting_prob_and_writes_diagnostics(
    *, tmp_path, monkeypatch
):
    # Real inputs, with toy delay distributions and the results redirected into tmp_path
    model_spec_name = "sse_60_pc_reporting"
    inputs = get_inputs_model_fits(model_spec_name=model_spec_name)
    inputs["results_paths"] = {
        name: tmp_path / path for name, path in inputs["results_paths"].items()
    }
    # Probabilities with full-precision digits, which only a round-trip read gets exactly
    rng = np.random.default_rng(1)
    delay_distributions = {
        delay_name: DelayDistribution(
            prob=rng.dirichlet(np.ones(n_lags)),
            first_lag=DELAY_FIRST_LAGS[delay_name],
        )
        for delay_name, n_lags in (
            ("serial_interval", 6),
            ("inc_period", 5),
            ("tost", 4),
        )
    }
    inputs["delay_distributions_results_paths"] = {}
    for delay_name, delay in delay_distributions.items():
        path = tmp_path / f"{delay_name}.csv"
        delay.to_df().to_csv(path, index=False)
        inputs["delay_distributions_results_paths"][delay_name] = path
    monkeypatch.setattr(
        model_fits, "get_inputs_model_fits", lambda *, model_spec_name: inputs
    )
    pipeline_kwargs = {}
    # Diagnostics of two days' fits, with a latent constant on the first (so its stats are
    # NaN) but not the second, and divergences counted on the second only
    diagnostics_df = pd.DataFrame(
        [
            {"day": day}
            | compute_fit_diagnostics(
                trace=xr.DataTree.from_dict(
                    {
                        "posterior": xr.Dataset(
                            {
                                "rep_no_before": (
                                    ("chain", "draw"),
                                    rng.normal(size=(2, 50)),
                                ),
                                "u": (("chain", "draw", "latent_day"), u_values),
                            }
                        ),
                        "sample_stats": xr.Dataset(attrs=sample_stats_attrs),
                    }
                )
            )
            for day, u_values, sample_stats_attrs in [
                (0, np.ones((2, 50, 3)), {}),
                (1, rng.uniform(size=(2, 50, 3)), {"n_divergences": 2}),
            ]
        ]
    )

    def _run_pipeline(**kwargs):
        pipeline_kwargs.update(kwargs)
        posterior = xr.Dataset({"rep_no_before": (("chain", "draw"), np.ones((1, 2)))})
        return {
            "risk_df": pd.DataFrame({"day": [0]}),
            "diagnostics_df": diagnostics_df,
            "evidence": {"log_evidence": 0.0},
            "trace": xr.DataTree.from_dict({"posterior": posterior}),
        }

    monkeypatch.setattr(model_fits, "run_pipeline", _run_pipeline)

    model_fits.run_analyses(model_spec_name=model_spec_name)

    assert pipeline_kwargs["delay_distributions"].keys() == delay_distributions.keys()
    for delay_name, delay in delay_distributions.items():
        actual = pipeline_kwargs["delay_distributions"][delay_name]
        assert isinstance(actual, DelayDistribution)
        assert actual.first_lag == delay.first_lag
        np.testing.assert_array_equal(actual.prob, delay.prob)
    assert (
        pipeline_kwargs["reporting_prob"]
        == MODEL_SPECS[model_spec_name]["reporting_prob"]
    )
    # The diagnostics come back from their CSV as they went in
    assert diagnostics_df["min_ess_bulk_u"].isna().tolist() == [True, False]
    pd.testing.assert_frame_equal(
        pd.read_csv(inputs["results_paths"]["diagnostics"]),
        diagnostics_df,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    ("lags", "message"),
    [
        ([0, 1], "serial_interval must start at lag 1"),
        ([1, 3], "consecutive"),
        ([2, 1], "consecutive"),
    ],
)
def test_run_analyses_rejects_invalid_csv_lags(*, lags, message, tmp_path, monkeypatch):
    inputs = get_inputs_model_fits(model_spec_name="poisson")
    path = tmp_path / "serial_interval.csv"
    pd.DataFrame({"lag": lags, "prob": [0.3, 0.7]}).to_csv(path, index=False)
    inputs["delay_distributions_results_paths"] = {"serial_interval": path}
    monkeypatch.setattr(
        model_fits, "get_inputs_model_fits", lambda *, model_spec_name: inputs
    )
    with pytest.raises(ValueError, match=message):
        model_fits.run_analyses(model_spec_name="poisson")
