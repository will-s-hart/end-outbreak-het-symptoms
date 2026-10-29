# Note that AI tools were used to generate tests
import numpy as np
import pytest

from endoutbreakhet import (
    DelayDistribution,
    compute_log_evidence,
    compute_risk,
    fit_model,
    run_pipeline,
)


@pytest.mark.parametrize(
    "entry_point", [fit_model, compute_log_evidence, compute_risk, run_pipeline]
)
@pytest.mark.parametrize(
    ("delay_name", "first_lag"),
    [("serial_interval", 0), ("inc_period", 0), ("tost", 1)],
)
def test_model_entry_points_reject_unsupported_delay_lag(
    *, entry_point, delay_name, first_lag, toy_inputs
):
    delays = toy_inputs["delay_distributions"]
    delays[delay_name] = DelayDistribution(
        prob=delays[delay_name].prob, first_lag=first_lag
    )
    rng = np.random.default_rng(0)
    extra_kwargs = {
        fit_model: {"priors": toy_inputs["priors"], "rng": rng},
        compute_log_evidence: {
            "priors": toy_inputs["priors"],
            "rng": rng,
            "trace": None,
            "n_proposal_samples": 10,
        },
        compute_risk: {"trace": None},
        run_pipeline: {
            "priors": toy_inputs["priors"],
            "rng": rng,
            "comparison_day": 0,
            "risk_days": np.array([0]),
            "fit_kwargs": {},
            "n_evidence_proposal_samples": 10,
        },
    }
    with pytest.raises(ValueError, match=f"{delay_name} must start at lag"):
        entry_point(
            model_name="poisson" if delay_name == "serial_interval" else "poisson_so",
            reported_incidence_vec=toy_inputs["reported_incidence_vec"],
            switch_day=toy_inputs["switch_day"],
            delay_distributions=delays,
            **extra_kwargs[entry_point],
        )
