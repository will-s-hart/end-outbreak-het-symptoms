# Note that AI tools were used to generate tests
import itertools

import arviz_stats
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.special
import scipy.stats
import xarray as xr

from endoutbreakhet import DelayDistribution
from endoutbreakhet.inference import (
    _build_pymc_model,
    _dispersed_starts,
    _gammaincinv,
    compute_fit_diagnostics,
    fit_model,
)
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
    # day 16 (serial interval-based) or 17 (SO), so later days are left out. Unreported
    # cases could make it nonzero on any day
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
    # nonzero up to day 16 (serial interval-based) or 17 (SO), where it is zero from day 16
    # or 17 from reported cases alone. In the serial interval-based models it is still
    # zero on day 17 (so there are no cases)
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
    if model_name not in ("ssd", "sse"):
        expected = scipy.stats.poisson.logpmf(
            reported, reporting_prob * mean
        ) + scipy.stats.poisson.logpmf(unreported, (1 - reporting_prob) * mean)
    else:
        dispersion = param_values["dispersion"] * (
            1 if model_name == "ssd" else total_inf_vec[days]
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
    # Force of infection/disease from the day-0 case is zero from day 5 (serial
    # interval-based) or 6 (SO), so the reported case on day 7 needs unreported cases in
    # between, and the data are
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
    # days, at the initial point), must not give NaN gradients, the dispersion's included
    model = _build_pymc_model(model_name=model_name, **toy_inputs_under_reporting)
    rvs = [rv for rv in model.free_RVs if rv.name != "unreported_incidence"]
    assert np.all(np.isfinite(model.compile_dlogp(vars=rvs)(model.initial_point())))


@pytest.mark.parametrize("reporting_prob", [1.0, 0.5])
def test_nuts_samples_every_continuous_variable(
    *, model_name, reporting_prob, toy_inputs
):
    # Every continuous variable has a gradient (the gamma latents have one in their shape),
    # so pm.sample gives NUTS all of them, and Metropolis any unreported cases
    model = _build_pymc_model(
        model_name=model_name, **toy_inputs, reporting_prob=reporting_prob
    )
    with model:
        _, selected_steps = pm.sampling.mcmc.assign_step_methods(model)
    expected_steps = {pm.NUTS: set(model.continuous_value_vars)}
    if reporting_prob < 1:
        expected_steps[pm.Metropolis] = {
            model.rvs_to_values[model["unreported_incidence"]]
        }
    assert {
        method: set(value_vars) for method, value_vars in selected_steps.items()
    } == expected_steps


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


def test_gammaincinv_gradients():
    # The inverse of the regularised lower incomplete gamma function, as scipy's, with
    # gradients in the probability (the reciprocal of the gamma density) and in the shape,
    # where it's a normal float. The latter is checked against Richardson-extrapolated central
    # differences of its log in the log shape, to the accuracy of PyTensor's gradient of P.
    # Both are zero where it underflows
    k_vec, x_vec = (
        values.ravel()
        for values in np.meshgrid(
            np.geomspace(0.01, 100, 15),
            np.concatenate(
                [np.geomspace(1e-8, 0.5, 8), 1 - np.geomspace(1e-8, 0.5, 8)[::-1]]
            ),
        )
    )
    k, x = pt.dvectors("k", "x")
    z = _gammaincinv(k, x)
    assert isinstance(z, pt.TensorVariable)
    dz_dk = pytensor.grad(z.sum(), wrt=k)
    dz_dx = pytensor.grad(z.sum(), wrt=x)
    assert isinstance(dz_dk, pt.TensorVariable) and isinstance(dz_dx, pt.TensorVariable)
    z_vec, dz_dk_vec, dz_dx_vec = pytensor.function([k, x], [z, dz_dk, dz_dx])(
        k_vec, x_vec
    )

    np.testing.assert_array_equal(z_vec, scipy.special.gammaincinv(k_vec, x_vec))
    normal_idx = z_vec >= np.finfo(float).tiny
    np.testing.assert_allclose(
        dz_dx_vec[normal_idx],
        1 / scipy.stats.gamma.pdf(z_vec, a=k_vec)[normal_idx],
        rtol=1e-12,
    )

    def _dlogz_dlogk(*, step):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                np.log(scipy.special.gammaincinv(k_vec * np.exp(step), x_vec))
                - np.log(scipy.special.gammaincinv(k_vec * np.exp(-step), x_vec))
            ) / (2 * step)

    dlogz_dlogk_vec = (4 * _dlogz_dlogk(step=1e-3) - _dlogz_dlogk(step=2e-3)) / 3
    np.testing.assert_allclose(
        dz_dk_vec[normal_idx],
        (dlogz_dlogk_vec * z_vec / k_vec)[normal_idx],
        rtol=1e-4,
    )
    underflow_idx = z_vec == 0
    assert np.any(underflow_idx)
    np.testing.assert_array_equal(dz_dk_vec[underflow_idx], 0)
    np.testing.assert_array_equal(dz_dx_vec[underflow_idx], 0)


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
        ("ssi", None, 1.0),
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
        )["risk"]
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


@pytest.mark.parametrize(
    "kwarg", ["initvals", "init", "jitter_max_retries", "nuts_sampler"]
)
def test_fit_model_rejects_kwargs_it_sets(*, kwarg, toy_inputs):
    # fit_model sets the chains' starts and sampler itself
    with pytest.raises(ValueError, match=kwarg):
        fit_model(
            model_name="poisson",
            **toy_inputs,
            rng=np.random.default_rng(0),
            **{kwarg: None},
        )


@pytest.mark.parametrize(
    ("count_name", "stat_name", "kwargs"),
    [
        # A huge step size without tuning makes every draw of PyMC's NUTS diverge
        (
            "n_divergences",
            "diverging",
            {"step": lambda: pm.NUTS(step_scale=20), "tune": 0},
        ),
        # Trees of depth 1 at most leave most of nutpie's draws at the maximum depth
        (
            "n_max_treedepth",
            "maxdepth_reached",
            {"tune": 10, "nuts": {"max_treedepth": 1}},
        ),
    ],
)
def test_fit_model_counts_before_thinning(*, count_name, stat_name, kwargs, toy_inputs):
    # Thinning drops half of the counted draws from the sample stats, but not from the count
    traces = {
        thin: fit_model(
            model_name="poisson",
            **toy_inputs,
            rng=np.random.default_rng(0),
            thin=thin,
            chains=2,
            cores=1,
            draws=20,
            progressbar=False,
            compute_convergence_checks=False,
            **kwargs,
        )
        for thin in (1, 2)
    }
    count = int(traces[1].sample_stats[stat_name].sum())
    assert count > int(traces[2].sample_stats[stat_name].sum())
    for trace in traces.values():
        assert trace.sample_stats.attrs[count_name] == count


@pytest.mark.parametrize(
    ("reporting_prob", "step", "stat_name"),
    [
        (1.0, None, "maxdepth_reached"),
        # PyMC, with Metropolis for the unreported cases, or with the step given
        (0.5, None, "reached_max_treedepth"),
        (1.0, pm.NUTS, "reached_max_treedepth"),
    ],
)
def test_fit_model_samples_with_nutpie_where_it_can(
    *, reporting_prob, step, stat_name, toy_inputs
):
    # Told apart by their names for the stat of draws at the maximum tree depth, both
    # counted
    trace = fit_model(
        model_name="ssi",
        **toy_inputs,
        rng=np.random.default_rng(0),
        reporting_prob=reporting_prob,
        step=step,
        chains=2,
        cores=1,
        tune=10,
        draws=10,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert stat_name in trace.sample_stats
    assert {"n_divergences", "n_max_treedepth"} <= set(trace.sample_stats.attrs)


def test_fit_model_under_reporting_with_several_chains(*, toy_inputs_under_reporting):
    # NUTS for the continuous variables (so divergences are counted) and Metropolis for the
    # unreported cases, each chain from its own start
    trace = fit_model(
        model_name="ssi",
        **toy_inputs_under_reporting,
        rng=np.random.default_rng(0),
        chains=3,
        cores=1,
        tune=10,
        draws=10,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert trace.posterior.sizes["chain"] == 3
    assert "n_divergences" in trace.sample_stats.attrs


def test_dispersed_starts(*, model_name, toy_inputs_under_reporting):
    # Data followed by a long run of days without reports. Every chain's start has finite
    # density and its own jittered continuous variables, and the unreported cases are at
    # their initial values plus the chain's index on each day with reported cases, so
    # they're at their initial values (zero) on the days after the data
    n_extra_days = 40
    inputs = toy_inputs_under_reporting | {
        "reported_incidence_vec": np.pad(
            toy_inputs_under_reporting["reported_incidence_vec"], (0, n_extra_days)
        )
    }
    model = _build_pymc_model(model_name=model_name, **inputs)
    starts = _dispersed_starts(
        model=model,
        reported_incidence_vec=inputs["reported_incidence_vec"],
        chains=3,
        rng=np.random.default_rng(0),
    )

    logp_func = model.compile_logp()
    assert all(np.isfinite(logp_func(start)) for start in starts)
    for var in model.continuous_value_vars:
        for start_a, start_b in itertools.combinations(starts, 2):
            assert np.all(start_a[var.name] != start_b[var.name])
    initial_unreported_incidence = model.initial_point()["unreported_incidence"]
    np.testing.assert_array_equal(initial_unreported_incidence[-n_extra_days:], 0)
    for chain, start in enumerate(starts):
        np.testing.assert_array_equal(
            start["unreported_incidence"],
            initial_unreported_incidence
            + chain * (inputs["reported_incidence_vec"][1:] > 0),
        )


def test_dispersed_starts_redraws_nonfinite_density(*, toy_inputs, monkeypatch):
    # Starts with NaN or infinite log density are redrawn, until one is finite
    model = _build_pymc_model(model_name="poisson", **toy_inputs, reporting_prob=1.0)
    logps = iter([np.nan, -np.inf, np.inf, -3.0])
    monkeypatch.setattr(
        model, "compile_logp", lambda **kwargs: lambda point: next(logps)
    )
    _dispersed_starts(
        model=model,
        reported_incidence_vec=toy_inputs["reported_incidence_vec"],
        chains=1,
        rng=np.random.default_rng(0),
    )
    assert next(logps, None) is None


def test_dispersed_starts_raises_when_none_finite(*, toy_inputs, monkeypatch):
    # Naming the chain, with the log density of each term at the last start
    model = _build_pymc_model(model_name="poisson", **toy_inputs, reporting_prob=1.0)
    monkeypatch.setattr(model, "compile_logp", lambda **kwargs: lambda point: -np.inf)
    with pytest.raises(RuntimeError, match=r"chain 0 in 10 attempts.*rep_no_before"):
        _dispersed_starts(
            model=model,
            reported_incidence_vec=toy_inputs["reported_incidence_vec"],
            chains=2,
            rng=np.random.default_rng(0),
        )


def test_fit_diagnostics_classify_elements():
    # Of four elements, one mixes well, one is constant and two have a non-finite draw (an
    # infinite one would still give R-hat near 1), so only the first is diagnosed
    rng = np.random.default_rng(0)
    values = rng.normal(size=(4, 500, 4))
    values[..., 1] = 2.0
    values[0, 7, 2] = np.inf
    values[3, 9, 3] = np.nan
    diagnostics = compute_fit_diagnostics(
        trace=_synthetic_trace(posterior={"x": (("chain", "draw", "x_dim"), values)})
    )
    assert diagnostics["n_nonfinite_x"] == 2
    assert diagnostics["n_constant_x"] == 1
    assert diagnostics["n_diagnosed_x"] == 1
    mixed_values = values[..., 0]
    expected = {
        "rhat_x": arviz_stats.rhat(mixed_values),
        "ess_bulk_x": arviz_stats.ess(mixed_values),
        "ess_tail_x": arviz_stats.ess(mixed_values, method="tail", prob=(0.05, 0.95)),
    }
    for name, value in expected.items():
        worst = "max" if name.startswith("rhat") else "min"
        assert diagnostics[f"{worst}_{name}"] == pytest.approx(value)
        assert diagnostics[f"median_{name}"] == pytest.approx(value)
    assert diagnostics["max_rhat_x"] < 1.01


def test_fit_diagnostics_worst_and_median_over_elements():
    # A well-mixed element and one whose chains are apart: the worst R-hat is the second's,
    # past 1.01, and the median is between the two
    rng = np.random.default_rng(0)
    values = rng.normal(size=(4, 500, 2))
    values[..., 1] += np.arange(4)[:, None]
    diagnostics = compute_fit_diagnostics(
        trace=_synthetic_trace(posterior={"x": (("chain", "draw", "x_dim"), values)})
    )
    rhat_vec = arviz_stats.rhat(values)
    assert diagnostics["max_rhat_x"] == pytest.approx(np.max(rhat_vec))
    assert diagnostics["median_rhat_x"] == pytest.approx(np.median(rhat_vec))
    assert diagnostics["max_rhat_x"] > 1.01
    assert diagnostics["min_ess_bulk_x"] == pytest.approx(
        np.min(arviz_stats.ess(values))
    )


def test_fit_diagnostics_chains_stuck_apart():
    # Each chain stuck at its own value
    values = np.repeat(np.arange(4.0)[:, None], 500, axis=1)
    diagnostics = compute_fit_diagnostics(
        trace=_synthetic_trace(posterior={"x": (("chain", "draw"), values)})
    )
    assert diagnostics["n_diagnosed_x"] == 1
    assert not diagnostics["max_rhat_x"] <= 1.01


def test_fit_diagnostics_constant_variable():
    # A fixed parameter is constant, so it has no elements to diagnose and NaN statistics
    diagnostics = compute_fit_diagnostics(
        trace=_synthetic_trace(
            posterior={"dispersion": (("chain", "draw"), np.full((4, 100), 0.18))}
        )
    )
    assert diagnostics["n_constant_dispersion"] == 1
    assert diagnostics["n_diagnosed_dispersion"] == 0
    for stat_name in ("rhat", "ess_bulk", "ess_tail"):
        worst = "max" if stat_name == "rhat" else "min"
        assert np.isnan(diagnostics[f"{worst}_{stat_name}_dispersion"])
        assert np.isnan(diagnostics[f"median_{stat_name}_dispersion"])


@pytest.mark.parametrize("separated", [False, True])
def test_fit_diagnostics_sparse_counts(*, separated):
    # Rare unreported cases (in 2% of draws) mix well when spread over the chains, but not
    # when all in one chain
    rng = np.random.default_rng(0)
    counts = np.zeros((4, 1000), dtype=np.int64)
    if separated:
        counts[0, rng.choice(1000, size=80, replace=False)] = 1
    else:
        counts.reshape(-1)[rng.choice(4000, size=80, replace=False)] = 1
    diagnostics = compute_fit_diagnostics(
        trace=_synthetic_trace(
            posterior={
                "unreported_incidence": (
                    ("chain", "draw", "local_day"),
                    counts[..., None],
                )
            }
        )
    )
    assert diagnostics["n_diagnosed_unreported_incidence"] == 1
    passes = (
        diagnostics["max_rhat_unreported_incidence"] <= 1.01
        and diagnostics["min_ess_bulk_unreported_incidence"] >= 400
        and diagnostics["min_ess_tail_unreported_incidence"] >= 400
    )
    assert passes != separated
    assert diagnostics["mean_total_unreported"] == pytest.approx(0.02)


@pytest.mark.parametrize("counts", [{}, {"n_divergences": 3, "n_max_treedepth": 5}])
def test_fit_diagnostics_counts(*, counts):
    # The numbers of divergences and of draws at the maximum tree depth, NaN without a
    # count (NUTS didn't run)
    rng = np.random.default_rng(0)
    diagnostics = compute_fit_diagnostics(
        trace=_synthetic_trace(
            posterior={"x": (("chain", "draw"), rng.normal(size=(2, 100)))},
            sample_stats_attrs=counts,
        )
    )
    for count_name in ("n_divergences", "n_max_treedepth"):
        if counts:
            assert diagnostics[count_name] == counts[count_name]
        else:
            assert np.isnan(diagnostics[count_name])
    assert "mean_total_unreported" not in diagnostics


def test_fit_diagnostics_of_fit(*, toy_inputs_under_reporting):
    # Every posterior variable of a real fit, deterministics included, with counts of
    # elements that cover them all
    trace = fit_model(
        model_name="ssi",
        **toy_inputs_under_reporting,
        rng=np.random.default_rng(0),
        chains=2,
        cores=1,
        tune=20,
        draws=20,
        progressbar=False,
        compute_convergence_checks=False,
    )
    diagnostics = compute_fit_diagnostics(trace=trace)
    for name, values in trace.posterior.data_vars.items():
        n_elements = values.size // (values.sizes["chain"] * values.sizes["draw"])
        assert (
            diagnostics[f"n_nonfinite_{name}"]
            + diagnostics[f"n_constant_{name}"]
            + diagnostics[f"n_diagnosed_{name}"]
            == n_elements
        )
    assert {"total_incidence", "cohort_inf", "unreported_incidence"} <= {
        name.removeprefix("n_diagnosed_")
        for name in diagnostics
        if name.startswith("n_diagnosed_")
    }
    assert diagnostics["n_divergences"] >= 0
    assert diagnostics["n_max_treedepth"] >= 0
    assert diagnostics["mean_total_unreported"] == pytest.approx(
        float(trace.posterior["unreported_incidence"].sum(dim="local_day").mean())
    )


def _synthetic_trace(*, posterior, sample_stats_attrs=None):
    # A trace with the posterior variables given as (dims, values)
    return xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(posterior),
            "sample_stats": xr.Dataset(attrs=sample_stats_attrs or {}),
        }
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
