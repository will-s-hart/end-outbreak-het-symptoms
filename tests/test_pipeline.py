# Note that AI tools were used to generate tests
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from endoutbreakhet import pipeline

# Seeded with the year of the first Pipe Masters
_SEED = 1971


@pytest.fixture
def pipeline_inputs(*, toy_inputs):
    # Toy data run to day 17. The comparison day (10) is not a risk day, and the risk days
    # are first, second and third reef: third reef (day 19) only breaks once the flat spell
    # after the data is padded in
    return toy_inputs | {
        "model_name": "poisson",
        "comparison_day": 10,
        "risk_days": np.array([13, 16, 19]),
        "fit_kwargs": {
            "chains": 2,
            "tune": 50,
            "draws": 50,
            "compute_convergence_checks": False,
        },
        "n_evidence_proposal_samples": 200,
        "rng": np.random.default_rng(_SEED),
    }


def test_fits_data_through_each_day(*, pipeline_inputs, monkeypatch):
    fits = []
    evidence_calls = []

    def _paddle_out(*, reported_incidence_vec, rng, **kwargs):
        # Stub fit that paddles out but never gets barrelled, just noting the set it saw
        fits.append({"reported_incidence_vec": reported_incidence_vec, "rng": rng})
        return types.SimpleNamespace(n_days=len(reported_incidence_vec))

    def _fake_risk(*, trace, risk_type, **kwargs):
        return trace.n_days + {"cases": 0.25, "transmission": 0.5}[risk_type]

    def _fake_log_evidence(*, trace, **kwargs):
        evidence_calls.append(trace)
        return {"log_evidence": -trace.n_days}

    monkeypatch.setattr(pipeline, "fit_model", _paddle_out)
    monkeypatch.setattr(pipeline, "compute_risk", _fake_risk)
    monkeypatch.setattr(pipeline, "compute_log_evidence", _fake_log_evidence)
    outputs = pipeline.run_pipeline(**pipeline_inputs, n_jobs=1)

    days = [10, 13, 16, 19]
    padded_incidence_vec = np.concatenate(
        [pipeline_inputs["reported_incidence_vec"], [0, 0]]
    )
    assert len(fits) == len(days)
    for fit, day in zip(fits, days, strict=True):
        np.testing.assert_array_equal(
            fit["reported_incidence_vec"], padded_incidence_vec[: day + 1]
        )
        assert (
            fit["reported_incidence_vec"].dtype
            == pipeline_inputs["reported_incidence_vec"].dtype
        )
    # Each day rides its own stream of random numbers
    assert len({fit["rng"].random() for fit in fits}) == len(days)

    pd.testing.assert_frame_equal(
        outputs["risk_df"],
        pd.DataFrame(
            {
                "day": days,
                "risk_cases": [day + 1.25 for day in days],
                "risk_transmission": [day + 1.5 for day in days],
            }
        ),
        check_dtype=False,
    )
    # Like Gerry Lopez at Pipe, the evidence is taken once, on the comparison day only
    assert len(evidence_calls) == 1
    assert evidence_calls[0] is outputs["trace"]
    assert outputs["trace"].n_days == 11
    assert outputs["evidence"] == {"log_evidence": -11}


def test_parallel_matches_serial(*, pipeline_inputs):
    # Pipe and Backdoor break left and right off the same peak: from the same seed, serial
    # and parallel runs must ride the identical wave
    outputs_serial = pipeline.run_pipeline(
        **(pipeline_inputs | {"rng": np.random.default_rng(_SEED)}), n_jobs=1
    )
    outputs_parallel = pipeline.run_pipeline(
        **(pipeline_inputs | {"rng": np.random.default_rng(_SEED)}), n_jobs=2
    )

    pd.testing.assert_frame_equal(
        outputs_serial["risk_df"], outputs_parallel["risk_df"]
    )
    assert outputs_serial["evidence"] == outputs_parallel["evidence"]
    xr.testing.assert_equal(
        outputs_serial["trace"].posterior.to_dataset(),
        outputs_parallel["trace"].posterior.to_dataset(),
    )
    risks = outputs_serial["risk_df"][["risk_cases", "risk_transmission"]].to_numpy()
    assert np.all((risks >= 0) & (risks <= 1))
