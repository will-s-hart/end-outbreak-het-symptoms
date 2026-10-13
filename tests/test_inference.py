# Note that AI tools were used to generate tests
import numpy as np
import pymc as pm
import pytest
import scipy.special
import scipy.stats

from endoutbreakhet.inference import _build_pymc_model, fit_model
from endoutbreakhet.utils import parse_lognormal_prior

# Toy data: 9 days with cases, and 14 days with nonzero expected total infectiousness
# (TOST lags 0-2)
_N_LATENTS = {"ssi": 9, "ssi_so": 9, "sse_so": 14}


def test_free_rvs(*, model_name, toy_inputs):
    model = _build_pymc_model(model_name=model_name, **toy_inputs)
    expected_names = {"rep_no_before", "rep_no_after"}
    if model_name not in ("poisson", "poisson_so"):
        expected_names |= {"dispersion"}
    if model_name in _N_LATENTS:
        expected_names |= {"u"}
        assert model.initial_point()["u_interval__"].shape == (_N_LATENTS[model_name],)
    assert {rv.name for rv in model.free_RVs} == expected_names


def test_fixed_prior(*, model_name, toy_inputs_fixed_dispersion):
    # A fixed parameter is a deterministic, not a free variable
    model = _build_pymc_model(model_name=model_name, **toy_inputs_fixed_dispersion)
    assert "dispersion" not in {rv.name for rv in model.free_RVs}
    if model_name not in ("poisson", "poisson_so"):
        assert "dispersion" in {var.name for var in model.deterministics}
        assert model["dispersion"].eval() == 0.18


def test_observed_days(*, model_name, toy_inputs):
    # Day 0 is imported; force of infection/disease is zero from day 16 (naive) or 17 (SO)
    model = _build_pymc_model(model_name=model_name, **toy_inputs)
    last_day = 16 if model_name.endswith("_so") else 15
    np.testing.assert_array_equal(
        model.rvs_to_values[model["likelihood"]].data,
        toy_inputs["incidence_vec"][1 : last_day + 1],
    )


def test_unknown_model_raises(*, toy_inputs):
    with pytest.raises(ValueError, match="Unknown model name"):
        _build_pymc_model(model_name="nonexistent", **toy_inputs)


def test_unsupported_prior_raises(*, model_name, toy_inputs):
    priors = toy_inputs["priors"] | {
        "rep_no_after": {"dist": "gamma", "median": 1.0, "quantile_025": 0.2}
    }
    with pytest.raises(ValueError, match="Unsupported prior"):
        _build_pymc_model(model_name=model_name, **(toy_inputs | {"priors": priors}))


def test_poisson_logp(*, toy_inputs):
    # Independent hand computation of the Poisson model's log density in PyMC's sampling
    # space (log R), where the lognormal prior on R is a normal prior on log R
    incidence_vec = toy_inputs["incidence_vec"]
    n_days = len(incidence_vec)
    rep_no_before, rep_no_after = 1.3, 0.6
    total_inf_vec = np.convolve(
        incidence_vec,
        np.concatenate([[0.0], toy_inputs["delay_distributions"]["serial_interval"]]),
    )[:n_days]
    rep_no_vec = np.where(
        np.arange(n_days) < toy_inputs["switch_day"], rep_no_before, rep_no_after
    )
    observed_days = slice(1, 16)
    expected = scipy.stats.poisson.logpmf(
        incidence_vec[observed_days],
        rep_no_vec[observed_days] * total_inf_vec[observed_days],
    ).sum()
    for name, value in [
        ("rep_no_before", rep_no_before),
        ("rep_no_after", rep_no_after),
    ]:
        prior_params = parse_lognormal_prior(name=name, priors=toy_inputs["priors"])
        expected += scipy.stats.norm.logpdf(
            np.log(value), loc=prior_params["mu"], scale=prior_params["sigma"]
        )

    model = _build_pymc_model(model_name="poisson", **toy_inputs)
    point = {
        "rep_no_before_log__": np.log(rep_no_before),
        "rep_no_after_log__": np.log(rep_no_after),
    }
    assert model.compile_logp()(point) == pytest.approx(expected)


def test_gamma_via_uniform(*, toy_inputs):
    model = _build_pymc_model(model_name="ssi", **toy_inputs)
    incidence_vec = toy_inputs["incidence_vec"]
    mean_vec = incidence_vec[incidence_vec > 0]
    dispersion = 0.3
    u = np.random.default_rng(0).uniform(size=len(mean_vec))

    (cohort_inf_case_days,) = model.replace_rvs_by_values(
        [model["cohort_inf_case_days"]]
    )
    values = cohort_inf_case_days.eval(
        {
            model.rvs_to_values[model["dispersion"]]: np.log(dispersion),
            model.rvs_to_values[model["u"]]: scipy.special.logit(u),
        }
    )
    np.testing.assert_allclose(
        values,
        scipy.stats.gamma.ppf(u, a=dispersion * mean_vec, scale=1 / dispersion),
        rtol=1e-8,
    )


@pytest.mark.parametrize(("model_name", "step"), [("poisson", None), ("ssi", pm.Slice)])
def test_fit_model(*, model_name, step, toy_inputs):
    trace = fit_model(
        model_name=model_name,
        **toy_inputs,
        rng=np.random.default_rng(0),
        step=step,
        thin=2,
        chains=1,
        tune=20,
        draws=20,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert {"rep_no_before", "rep_no_after"} <= set(trace.posterior.data_vars)
    np.testing.assert_array_equal(trace.posterior.draw, np.arange(10))
