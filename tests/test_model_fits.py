# Note that AI tools were used to generate tests
import numpy as np
import pandas as pd
import xarray as xr

from scripts import model_fits
from scripts.inputs import MODEL_SPECS, get_inputs_model_fits


def test_run_analyses_passes_delays_and_reporting_prob(*, tmp_path, monkeypatch):
    # Real inputs, with toy delay distributions and the results redirected into tmp_path
    model_spec_name = "sse_60_pc_reporting"
    inputs = get_inputs_model_fits(model_spec_name=model_spec_name)
    inputs["results_paths"] = {
        name: tmp_path / path for name, path in inputs["results_paths"].items()
    }
    # Probabilities with full-precision digits, which only a round-trip read gets exactly
    rng = np.random.default_rng(1)
    delay_distributions = {
        delay_name: rng.dirichlet(np.ones(n_lags))
        for delay_name, n_lags in (
            ("serial_interval", 6),
            ("inc_period", 5),
            ("tost", 4),
        )
    }
    inputs["delay_distributions_results_paths"] = {}
    for delay_name, prob_vec in delay_distributions.items():
        path = tmp_path / f"{delay_name}.csv"
        pd.DataFrame({"lag": np.arange(len(prob_vec)), "prob": prob_vec}).to_csv(
            path, index=False
        )
        inputs["delay_distributions_results_paths"][delay_name] = path
    monkeypatch.setattr(
        model_fits, "get_inputs_model_fits", lambda *, model_spec_name: inputs
    )
    pipeline_kwargs = {}

    def _run_pipeline(**kwargs):
        pipeline_kwargs.update(kwargs)
        posterior = xr.Dataset({"rep_no_before": (("chain", "draw"), np.ones((1, 2)))})
        return {
            "risk_df": pd.DataFrame({"day": [0]}),
            "evidence": {"log_evidence": 0.0},
            "trace": xr.DataTree.from_dict({"posterior": posterior}),
        }

    monkeypatch.setattr(model_fits, "run_pipeline", _run_pipeline)

    model_fits.run_analyses(model_spec_name=model_spec_name)

    assert pipeline_kwargs["delay_distributions"].keys() == delay_distributions.keys()
    for delay_name, prob_vec in delay_distributions.items():
        np.testing.assert_array_equal(
            pipeline_kwargs["delay_distributions"][delay_name], prob_vec
        )
    assert (
        pipeline_kwargs["reporting_prob"]
        == MODEL_SPECS[model_spec_name]["reporting_prob"]
    )
