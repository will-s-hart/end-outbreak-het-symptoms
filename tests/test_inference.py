# Note that AI tools were used to generate tests
import numpy as np
import pymc as pm
import pytest
import scipy.special
import scipy.stats

from endoutbreakhet import DelayDistribution
from endoutbreakhet.inference import _build_pymc_model, fit_model
from endoutbreakhet.risk import compute_risk
from endoutbreakhet.utils import parse_lognormal_prior

# Toy data: 9 days with cases, and 14 days with nonzero expected total infectiousness
# (TOST lags 0-2). With under-reporting, there are latents on all 18 days
_N_LATENTS = {"ssi": 9, "ssi_so": 9, "sse_so": 14}


@pytest.mark.parametrize("reporting_prob", [1.0, 0.5])
def test_free_rvs(*, model_name, reporting_prob, toy_inputs):
    model = _build_pymc_model(
        model_name=model_name, **toy_inputs, reporting_prob=reporting_prob
    )
    n_days = len(toy_inputs["reported_incidence_vec"])
    expected_names = {"rep_no_before", "rep_no_after"}
    if model_name not in ("poisson", "poisson_so"):
        expected_names |= {"dispersion"}
    if reporting_prob < 1:
        # Unreported cases on every day but day 0
        expected_names |= {"unreported_incidence"}
        assert model.initial_point()["unreported_incidence"].shape == (n_days - 1,)
    if model_name in _N_LATENTS:
        expected_names |= {"u"}
        n_latents = _N_LATENTS[model_name] if reporting_prob == 1 else n_days
        assert model.initial_point()["u_interval__"].shape == (n_latents,)
    assert {rv.name for rv in model.free_RVs} == expected_names


def test_fixed_prior(*, model_name, toy_inputs_fixed_dispersion):
    # A fixed parameter is a deterministic, not a free variable
    model = _build_pymc_model(
        model_name=model_name, **toy_inputs_fixed_dispersion, reporting_prob=1.0
    )
    assert "dispersion" not in {rv.name for rv in model.free_RVs}
    if model_name not in ("poisson", "poisson_so"):
        assert "dispersion" in {var.name for var in model.deterministics}
        assert model["dispersion"].eval() == 0.18


@pytest.mark.parametrize("reporting_prob", [1.0, 0.5])
def test_observed_days(*, model_name, reporting_prob, toy_inputs):
    # Day 0 is imported. If all cases are reported, force of infection/disease is zero from
    # day 16 (naive) or 17 (SO), so later days are left out. Unreported cases could make it
    # nonzero on any day
    model = _build_pymc_model(
        model_name=model_name, **toy_inputs, reporting_prob=reporting_prob
    )
    reported_incidence_vec = toy_inputs["reported_incidence_vec"]
    if reporting_prob < 1:
        last_day = len(reported_incidence_vec) - 1
    else:
        last_day = 16 if model_name.endswith("_so") else 15
    likelihood_days = np.arange(1, last_day + 1)
    np.testing.assert_array_equal(model.coords["likelihood_day"], likelihood_days)
    np.testing.assert_array_equal(
        model.rvs_to_values[model["likelihood"]].data,
        reported_incidence_vec[likelihood_days],
    )


def test_unknown_model_raises(*, toy_inputs):
    with pytest.raises(ValueError, match="Unknown model name"):
        _build_pymc_model(model_name="nonexistent", **toy_inputs, reporting_prob=1.0)


@pytest.mark.parametrize("reporting_prob", [0.0, 1.5])
def test_invalid_reporting_prob_raises(*, reporting_prob, toy_inputs):
    with pytest.raises(ValueError, match="Reporting probability"):
        _build_pymc_model(
            model_name="poisson", **toy_inputs, reporting_prob=reporting_prob
        )


def test_unsupported_prior_raises(*, model_name, toy_inputs):
    priors = toy_inputs["priors"] | {
        "rep_no_after": {"dist": "gamma", "median": 1.0, "quantile_025": 0.2}
    }
    with pytest.raises(ValueError, match="Unsupported prior"):
        _build_pymc_model(
            model_name=model_name,
            **(toy_inputs | {"priors": priors}),
            reporting_prob=1.0,
        )


def test_poisson_logp(*, toy_inputs):
    # Independent hand computation of the Poisson model's log density in PyMC's sampling
    # space (log R)
    reported_incidence_vec = toy_inputs["reported_incidence_vec"]
    param_values = {"rep_no_before": 1.3, "rep_no_after": 0.6}
    _, expected_incidence_vec = _expected_incidence(
        model_name="poisson",
        total_incidence_vec=reported_incidence_vec,
        param_values=param_values,
        u_vec=None,
        inputs=toy_inputs,
    )
    observed_days = slice(1, 16)
    expected = scipy.stats.poisson.logpmf(
        reported_incidence_vec[observed_days], expected_incidence_vec[observed_days]
    ).sum() + _log_prior(param_values=param_values, priors=toy_inputs["priors"])

    model = _build_pymc_model(model_name="poisson", **toy_inputs, reporting_prob=1.0)
    point = {f"{name}_log__": np.log(value) for name, value in param_values.items()}
    assert model.compile_logp()(point) == pytest.approx(expected)


def test_under_reporting_logp(*, model_name, toy_inputs_under_reporting):
    # Independent hand computation of the log density from the notes' thinned form: given
    # the history (including earlier unreported cases and any gamma latents), the reported
    # and unreported counts on each day are independent Poisson, or marginal and
    # conditional negative binomial. The model instead uses the total incidence and
    # binomial reporting. The unreported case on day 12 makes force of infection/disease
    # nonzero up to day 16 (naive) or 17 (SO), where it is zero from day 16 or 17 from
    # reported cases alone. In the naive models it is still zero on day 17 (so there are no
    # cases)
    inputs = toy_inputs_under_reporting
    reported_incidence_vec = inputs["reported_incidence_vec"]
    n_days = len(reported_incidence_vec)
    reporting_prob = inputs["reporting_prob"]
    param_values = {"rep_no_before": 1.3, "rep_no_after": 0.6}
    if model_name not in ("poisson", "poisson_so"):
        param_values["dispersion"] = 0.3
    unreported_incidence = np.zeros(n_days - 1, dtype=np.int64)
    unreported_incidence[[2, 6, 11]] = [1, 2, 1]
    total_incidence_vec = reported_incidence_vec + np.concatenate(
        [[0], unreported_incidence]
    )
    # Uniforms underlying the gamma latents (if any), which are on every day
    u_vec = np.random.default_rng(0).uniform(size=n_days)

    total_inf_vec, expected_incidence_vec = _expected_incidence(
        model_name=model_name,
        total_incidence_vec=total_incidence_vec,
        param_values=param_values,
        u_vec=u_vec,
        inputs=inputs,
    )
    days = np.arange(1, 18 if model_name.endswith("_so") else 17)
    np.testing.assert_array_equal(np.flatnonzero(expected_incidence_vec > 0), days)
    reported = reported_incidence_vec[days]
    unreported = total_incidence_vec[days] - reported
    mean = expected_incidence_vec[days]
    if model_name not in ("dlo", "sse"):
        expected = scipy.stats.poisson.logpmf(
            reported, reporting_prob * mean
        ) + scipy.stats.poisson.logpmf(unreported, (1 - reporting_prob) * mean)
    else:
        dispersion = param_values["dispersion"] * (
            1 if model_name == "dlo" else total_inf_vec[days]
        )
        expected = _nbinom_logpmf(
            k=reported, mean=reporting_prob * mean, dispersion=dispersion
        ) + _nbinom_logpmf(
            k=unreported,
            mean=(1 - reporting_prob)
            * mean
            * (dispersion + reported)
            / (reporting_prob * mean + dispersion),
            dispersion=dispersion + reported,
        )
    expected = expected.sum() + _log_prior(
        param_values=param_values, priors=inputs["priors"]
    )

    model = _build_pymc_model(model_name=model_name, **inputs)
    point = {f"{name}_log__": np.log(value) for name, value in param_values.items()}
    point["unreported_incidence"] = unreported_incidence
    if model_name in _N_LATENTS:
        # In PyMC's sampling space (logit u), where the uniform density is u * (1 - u)
        point["u_interval__"] = scipy.special.logit(u_vec)
        expected += np.sum(np.log(u_vec * (1 - u_vec)))
    assert model.compile_logp()(point) == pytest.approx(expected)


def test_under_reporting_bridges_gap(*, model_name, toy_inputs):
    # Force of infection/disease from the day-0 case is zero from day 5 (naive) or 6 (SO),
    # so the reported case on day 7 needs unreported cases in between, and the data are
    # impossible if all cases are reported. Fitting starts from a point with nonzero
    # density, with an unreported case on each day before day 7
    inputs = toy_inputs | {
        "reported_incidence_vec": np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    }
    with pytest.raises(ValueError, match="Local incidence"):
        _build_pymc_model(model_name=model_name, **inputs, reporting_prob=1.0)

    model = _build_pymc_model(model_name=model_name, **inputs, reporting_prob=0.5)
    point = model.initial_point()
    np.testing.assert_array_equal(
        point["unreported_incidence"], [1, 1, 1, 1, 1, 1, 0, 0, 0]
    )
    assert np.isfinite(model.compile_logp()(point))


@pytest.mark.parametrize("reporting_prob", [1.0, 0.5])
def test_impossible_incidence_raises(*, model_name, reporting_prob, toy_inputs):
    # Without a day-0 case, no local cases are possible, reported or not
    inputs = toy_inputs | {"reported_incidence_vec": np.array([0, 0, 1, 0, 0, 0, 0, 0])}
    with pytest.raises(ValueError, match="Local incidence"):
        _build_pymc_model(
            model_name=model_name, **inputs, reporting_prob=reporting_prob
        )


def test_unreported_case_without_force_impossible(
    *, model_name, toy_inputs_under_reporting
):
    # At the initial point (with unreported cases only before the last reported case, on
    # day 11), force of infection/disease is zero on day 17
    model = _build_pymc_model(model_name=model_name, **toy_inputs_under_reporting)
    logp_func = model.compile_logp()
    point = model.initial_point()
    assert np.isfinite(logp_func(point))
    point["unreported_incidence"][-1] = 1
    assert logp_func(point) == -np.inf


def test_under_reporting_gradient_finite(*, model_name, toy_inputs_under_reporting):
    # Days with zero force of infection/disease, and gamma latents with zero mean (the last
    # days, at the initial point), must not give NaN gradients. The gamma latents have no
    # gradient with respect to their shape, so the dispersion is left out in models with
    # them
    model = _build_pymc_model(model_name=model_name, **toy_inputs_under_reporting)
    excluded_names = {"unreported_incidence"}
    if model_name in _N_LATENTS:
        excluded_names |= {"dispersion"}
    rvs = [rv for rv in model.free_RVs if rv.name not in excluded_names]
    assert np.all(np.isfinite(model.compile_dlogp(vars=rvs)(model.initial_point())))


@pytest.mark.parametrize("reporting_prob", [1.0, 0.5])
def test_nonfinite_expected_incidence_impossible(*, reporting_prob):
    # With logit(u) = 40 (u rounds to 1), total infectiousness on day 2 is infinite. It
    # only affects day 3 (after the data), but its products with zero incubation period
    # probabilities make the expected incidence NaN on days 1 and 2. That must have zero
    # density, rather than count as zero expected incidence
    model = _build_pymc_model(
        model_name="sse_so",
        reported_incidence_vec=np.array([2, 0, 0]),
        switch_day=3,
        delay_distributions={
            "tost": DelayDistribution(prob=np.array([0.5, 0, 0.5]), first_lag=0),
            "inc_period": DelayDistribution(prob=np.array([1.0]), first_lag=1),
        },
        priors={
            "rep_no_before": {"dist": "fixed", "value": 1.0},
            "rep_no_after": {"dist": "fixed", "value": 1.0},
            "dispersion": {"dist": "fixed", "value": 0.3},
        },
        reporting_prob=reporting_prob,
    )
    point = model.initial_point()
    point["u_interval__"][-1] = 40
    assert model.compile_logp()(point) == -np.inf


@pytest.mark.parametrize("reporting_prob", [1.0, 0.5])
def test_gamma_via_uniform(*, reporting_prob, toy_inputs):
    # Latents on every day, zero where there are no cases. Only the days with cases have a
    # uniform if all cases are reported, else every day does (here with an unreported case
    # on day 12)
    model = _build_pymc_model(
        model_name="ssi", **toy_inputs, reporting_prob=reporting_prob
    )
    reported_incidence_vec = toy_inputs["reported_incidence_vec"]
    n_days = len(reported_incidence_vec)
    dispersion = 0.3
    values = {model.rvs_to_values[model["dispersion"]]: np.log(dispersion)}
    if reporting_prob == 1:
        mean_vec = reported_incidence_vec
        latent_days = np.flatnonzero(reported_incidence_vec)
    else:
        unreported_incidence = np.zeros(n_days - 1, dtype=np.int64)
        unreported_incidence[11] = 1
        values[model.rvs_to_values[model["unreported_incidence"]]] = (
            unreported_incidence
        )
        mean_vec = reported_incidence_vec + np.concatenate([[0], unreported_incidence])
        latent_days = np.arange(n_days)
    np.testing.assert_array_equal(model.coords["day"], np.arange(n_days))
    np.testing.assert_array_equal(model.coords["latent_day"], latent_days)
    u = np.random.default_rng(0).uniform(size=len(latent_days))
    values[model.rvs_to_values[model["u"]]] = scipy.special.logit(u)

    (cohort_inf_vec,) = model.replace_rvs_by_values([model["cohort_inf"]])
    expected = np.zeros(n_days)
    nonzero_idx = mean_vec[latent_days] > 0
    nonzero_days = latent_days[nonzero_idx]
    expected[nonzero_days] = scipy.stats.gamma.ppf(
        u[nonzero_idx], a=dispersion * mean_vec[nonzero_days], scale=1 / dispersion
    )
    np.testing.assert_allclose(cohort_inf_vec.eval(values), expected, rtol=1e-8)


def _slice_and_untuned_metropolis():
    # Explicit steps: slice sampling for the continuous variables, and Metropolis without
    # tuning (so its scaling stays at 1) for the unreported cases
    model = pm.modelcontext(None)
    return [
        pm.Slice(vars=model.continuous_value_vars),
        pm.Metropolis(vars=[model["unreported_incidence"]], tune=False),
    ]


@pytest.mark.parametrize(
    ("model_name", "step", "reporting_prob"),
    [
        ("poisson", None, 1.0),
        ("ssi", pm.Slice, 1.0),
        # Metropolis steps for the unreported cases, with NUTS or slice sampling
        ("poisson", None, 0.5),
        ("ssi", pm.Slice, 0.5),
        ("ssi", _slice_and_untuned_metropolis, 0.5),
    ],
)
def test_fit_model(*, model_name, step, reporting_prob, toy_inputs):
    trace = fit_model(
        model_name=model_name,
        **toy_inputs,
        rng=np.random.default_rng(0),
        reporting_prob=reporting_prob,
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
    # Day coordinates reach the trace
    assert trace.observed_data["likelihood"].dims == ("likelihood_day",)
    if model_name == "ssi":
        assert trace.posterior["u"].dims == ("chain", "draw", "latent_day")
        assert trace.posterior["cohort_inf"].dims == ("chain", "draw", "day")
        np.testing.assert_array_equal(
            trace.posterior["day"], np.arange(len(toy_inputs["reported_incidence_vec"]))
        )
    if reporting_prob < 1:
        # Metropolis (the only step here with a scaling) samples the unreported cases, with
        # any explicit steps used as given
        assert "scaling" in trace.sample_stats
        if step is _slice_and_untuned_metropolis:
            assert np.all(trace.sample_stats["scaling"] == 1)
        unreported_incidence = trace.posterior["unreported_incidence"]
        assert unreported_incidence.dims == ("chain", "draw", "local_day")
        assert np.issubdtype(unreported_incidence.dtype, np.integer)
        assert np.all(unreported_incidence.values >= 0)
        # Total incidence on every day, with the day-0 cases all reported
        total_incidence = trace.posterior["total_incidence"]
        assert total_incidence.dims == ("chain", "draw", "day")
        np.testing.assert_array_equal(
            total_incidence.values,
            toy_inputs["reported_incidence_vec"]
            + np.pad(unreported_incidence.values, ((0, 0), (0, 0), (1, 0))),
        )

    # The trace has what risk needs, giving risks ordered as in the notes
    risks = {
        risk_type: compute_risk(
            model_name=model_name,
            reported_incidence_vec=toy_inputs["reported_incidence_vec"],
            switch_day=toy_inputs["switch_day"],
            delay_distributions=toy_inputs["delay_distributions"],
            trace=trace,
            reporting_prob=reporting_prob,
            risk_type=risk_type,
        )
        for risk_type in ("cases", "transmission", "sustained", "reports")
    }
    assert 0 <= risks["sustained"] <= risks["transmission"] <= risks["cases"] <= 1
    assert risks["sustained"] <= risks["reports"] <= risks["cases"]


@pytest.mark.parametrize(
    "step",
    [
        # A factory rather than a class, so the step takes every variable
        lambda: pm.Slice(),
        lambda: [pm.Slice(vars=[pm.modelcontext(None)["unreported_incidence"]])],
        lambda: pm.step_methods.CompoundStep(
            [pm.Slice(vars=[pm.modelcontext(None)["unreported_incidence"]])]
        ),
    ],
)
def test_incompatible_step_raises(*, step, toy_inputs_under_reporting):
    # Slice sampling truncates its proposals for the unreported cases to integers, which
    # biases the posterior
    with pytest.raises(ValueError, match="unreported_incidence"):
        fit_model(
            model_name="poisson",
            **toy_inputs_under_reporting,
            rng=np.random.default_rng(0),
            step=step,
        )


def _expected_incidence(
    *, model_name, total_incidence_vec, param_values, u_vec, inputs
):
    # Total infectiousness and expected incidence in each model by direct convolution, with
    # any gamma latents the quantiles at u_vec (zero where their shape is)
    n_days = len(total_incidence_vec)
    delay_distributions = inputs["delay_distributions"]

    def _convolve(*, x_vec, delay_name, first_lag):
        dist_vec = np.concatenate(
            [np.zeros(first_lag), delay_distributions[delay_name].prob]
        )
        return np.convolve(x_vec, dist_vec)[:n_days]

    def _gamma_latents(*, mean_vec):
        dispersion = param_values["dispersion"]
        latent_vec = np.zeros(n_days)
        nonzero_idx = mean_vec > 0
        latent_vec[nonzero_idx] = scipy.stats.gamma.ppf(
            u_vec[nonzero_idx],
            a=dispersion * mean_vec[nonzero_idx],
            scale=1 / dispersion,
        )
        return latent_vec

    so_model = model_name.endswith("_so")
    delay_name, first_lag = ("tost", 0) if so_model else ("serial_interval", 1)
    if model_name in ("ssi", "ssi_so"):
        total_inf_vec = _convolve(
            x_vec=_gamma_latents(mean_vec=total_incidence_vec),
            delay_name=delay_name,
            first_lag=first_lag,
        )
    else:
        total_inf_vec = _convolve(
            x_vec=total_incidence_vec, delay_name=delay_name, first_lag=first_lag
        )
    if model_name == "sse_so":
        total_inf_vec = _gamma_latents(mean_vec=total_inf_vec)

    rep_no_vec = np.where(
        np.arange(n_days) < inputs["switch_day"],
        param_values["rep_no_before"],
        param_values["rep_no_after"],
    )
    expected_incidence_vec = rep_no_vec * total_inf_vec
    if so_model:
        expected_incidence_vec = _convolve(
            x_vec=expected_incidence_vec, delay_name="inc_period", first_lag=1
        )
    return total_inf_vec, expected_incidence_vec


def _log_prior(*, param_values, priors):
    # In PyMC's sampling space (log params), where a lognormal prior on a parameter is a
    # normal prior on its log
    log_prior = 0.0
    for name, value in param_values.items():
        prior_params = parse_lognormal_prior(name=name, priors=priors)
        log_prior += scipy.stats.norm.logpdf(
            np.log(value), loc=prior_params["mu"], scale=prior_params["sigma"]
        )
    return log_prior


def _nbinom_logpmf(*, k, mean, dispersion):
    # Negative binomial with variance mean + mean^2 / dispersion
    return scipy.stats.nbinom.logpmf(
        k, n=dispersion, p=dispersion / (dispersion + mean)
    )
