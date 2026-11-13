# Note that AI tools were used to generate tests
import types

import numpy as np
import pytest
import scipy.integrate
import scipy.special
import scipy.stats
import xarray as xr

from endoutbreakhet import DelayDistribution, evidence
from endoutbreakhet.inference import _build_pymc_model, fit_model
from endoutbreakhet.utils import parse_lognormal_prior

_PARAM_NAMES = ["rep_no_before", "rep_no_after", "dispersion"]


@pytest.mark.parametrize(
    ("fixed_dispersion", "reporting_prob"),
    [(False, 1.0), (True, 1.0), (False, 0.5), (True, 0.5)],
)
def test_log_density_matches_pymc(
    *,
    model_name,
    fixed_dispersion,
    reporting_prob,
    toy_inputs,
    toy_inputs_fixed_dispersion,
):
    # Numpy log prior + log likelihood in the bridge sampling layout (log params,
    # z = Phi^-1(u), unreported case counts) against the PyMC model's log density in its
    # sampling parameterisation (log params, logit(u), unreported case counts), corrected
    # by the log Jacobian of logit(u) -> z
    inputs = toy_inputs_fixed_dispersion if fixed_dispersion else toy_inputs
    model = _build_pymc_model(
        model_name=model_name, **inputs, reporting_prob=reporting_prob
    )
    free_rv_names = {rv.name for rv in model.free_RVs}
    estimated_param_names = [name for name in _PARAM_NAMES if name in free_rv_names]
    fixed_params = {
        name: inputs["priors"][name]["value"]
        for name in _PARAM_NAMES
        if name in model.named_vars and name not in free_rv_names
    }
    # Days with gamma latents and unreported case counts, as the model has them
    latent_days = (
        np.array(model.coords["latent_day"]) if "u" in model.named_vars else None
    )
    n_latents = 0 if latent_days is None else len(latent_days)
    unreported_days = (
        np.array(model.coords["local_day"])
        if "unreported_incidence" in model.named_vars
        else None
    )
    rng = np.random.default_rng(0)
    samples = _prior_samples(
        estimated_param_names=estimated_param_names,
        n_latents=n_latents,
        priors=inputs["priors"],
        n_samples=5,
        rng=rng,
    )
    if unreported_days is not None:
        # Unreported cases up to day 12, so force of infection/disease is zero on the
        # last days, as it must be for there to be no cases there
        counts = np.zeros((len(samples), len(unreported_days)), dtype=np.int64)
        counts[:, :12] = rng.poisson(0.5, size=(len(samples), 12))
        samples = np.concatenate([samples, counts], axis=-1)

    log_prior_samples = evidence._log_prior_samples(
        samples=samples,
        estimated_param_names=estimated_param_names,
        latent_days=latent_days,
        priors=inputs["priors"],
    )
    log_likelihood_samples = getattr(evidence, f"_{model_name}_log_likelihood_samples")(
        reported_incidence_vec=inputs["reported_incidence_vec"],
        switch_day=inputs["switch_day"],
        delay_distributions=inputs["delay_distributions"],
        reporting_prob=reporting_prob,
        **evidence._unpack_samples(
            samples=samples,
            estimated_param_names=estimated_param_names,
            fixed_params=fixed_params,
            latent_days=latent_days,
            unreported_days=unreported_days,
            reported_incidence_vec=inputs["reported_incidence_vec"],
        ),
    )

    logp_fn = model.compile_logp()
    for sample, log_density in zip(samples, log_prior_samples + log_likelihood_samples):
        point = {
            f"{name}_log__": sample[i] for i, name in enumerate(estimated_param_names)
        }
        n_params = len(estimated_param_names)
        z = sample[n_params : n_params + n_latents]
        log_jacobian = 0.0
        if n_latents:
            log_cdf, log_sf = scipy.stats.norm.logcdf(z), scipy.stats.norm.logsf(z)
            point["u_interval__"] = log_cdf - log_sf
            log_jacobian = np.sum(scipy.stats.norm.logpdf(z) - log_cdf - log_sf)
        if unreported_days is not None:
            point["unreported_incidence"] = sample[n_params + n_latents :].astype(
                np.int64
            )
        assert np.isfinite(log_density)
        assert log_density == pytest.approx(logp_fn(point) + log_jacobian, rel=1e-9)


@pytest.mark.parametrize(
    ("estimated_param_names", "n_latents", "fixed_dispersion", "under_reporting"),
    [
        (["rep_no_before", "rep_no_after"], 0, False, False),
        (_PARAM_NAMES, 4, False, False),
        (["rep_no_before", "rep_no_after"], 4, True, False),
        (["rep_no_before", "rep_no_after"], 0, False, True),
        (_PARAM_NAMES, 4, False, True),
    ],
)
def test_sample_layout_packing_and_unpacking(
    *,
    estimated_param_names,
    n_latents,
    fixed_dispersion,
    under_reporting,
    toy_inputs,
    toy_inputs_fixed_dispersion,
):
    rng = np.random.default_rng(0)
    n_chains, n_draws = 2, 3
    reported_incidence_vec = toy_inputs["reported_incidence_vec"]
    n_days = len(reported_incidence_vec)
    data_vars: dict[str, tuple] = {
        name: (("chain", "draw"), rng.lognormal(size=(n_chains, n_draws)))
        for name in estimated_param_names
    }
    if fixed_dispersion:
        # Fixed parameters are in the trace as deterministics
        data_vars["dispersion"] = (
            ("chain", "draw"),
            np.full((n_chains, n_draws), 0.18),
        )
    coords = {}
    if n_latents:
        # Latents on some of the days
        coords["latent_day"] = np.sort(rng.choice(10, size=n_latents, replace=False))
        data_vars["u"] = (
            ("chain", "draw", "latent_day"),
            rng.uniform(size=(n_chains, n_draws, n_latents)),
        )
        # Deterministics in the trace are not used
        data_vars["cohort_inf"] = (
            ("chain", "draw", "day"),
            rng.gamma(shape=1.0, size=(n_chains, n_draws, n_days)),
        )
    if under_reporting:
        # Unreported cases on every day but day 0
        coords["local_day"] = np.arange(1, n_days)
        data_vars["unreported_incidence"] = (
            ("chain", "draw", "local_day"),
            rng.poisson(1.0, size=(n_chains, n_draws, n_days - 1)),
        )
    trace = types.SimpleNamespace(posterior=xr.Dataset(data_vars, coords=coords))
    inputs = toy_inputs_fixed_dispersion if fixed_dispersion else toy_inputs

    (
        returned_estimated_param_names,
        fixed_params,
        latent_days,
        unreported_days,
    ) = evidence._sample_layout(trace=trace, priors=inputs["priors"])
    assert returned_estimated_param_names == estimated_param_names
    assert fixed_params == ({"dispersion": 0.18} if fixed_dispersion else {})
    if n_latents:
        np.testing.assert_array_equal(latent_days, coords["latent_day"])
    else:
        assert latent_days is None
    if under_reporting:
        np.testing.assert_array_equal(unreported_days, coords["local_day"])
    else:
        assert unreported_days is None

    samples = evidence._posterior_samples(
        trace=trace,
        estimated_param_names=estimated_param_names,
        latent_days=latent_days,
        unreported_days=unreported_days,
    )
    n_unreported = n_days - 1 if under_reporting else 0
    n_dims = len(estimated_param_names) + n_latents + n_unreported
    assert samples.shape == (n_chains, n_draws, n_dims)

    unpacked_samples = evidence._unpack_samples(
        samples=samples.reshape(n_chains * n_draws, n_dims),
        estimated_param_names=estimated_param_names,
        fixed_params=fixed_params,
        latent_days=latent_days,
        unreported_days=unreported_days,
        reported_incidence_vec=reported_incidence_vec,
    )
    for name in [*estimated_param_names, *fixed_params]:
        np.testing.assert_allclose(
            unpacked_samples[f"{name}_samples"], data_vars[name][1].ravel()
        )
    if n_latents:
        np.testing.assert_allclose(
            scipy.stats.norm.cdf(unpacked_samples["latent_standard_normal_samples"]),
            data_vars["u"][1].reshape(n_chains * n_draws, n_latents),
        )
        np.testing.assert_array_equal(
            unpacked_samples["latent_days"], coords["latent_day"]
        )
    else:
        assert "latent_standard_normal_samples" not in unpacked_samples
        assert "latent_days" not in unpacked_samples
    # Total incidence on every day, with the day-0 cases all reported
    if under_reporting:
        expected_total_incidence_vec_samples = reported_incidence_vec + np.pad(
            data_vars["unreported_incidence"][1].reshape(n_chains * n_draws, -1),
            ((0, 0), (1, 0)),
        )
    else:
        expected_total_incidence_vec_samples = reported_incidence_vec[None, :]
    np.testing.assert_array_equal(
        unpacked_samples["total_incidence_vec_samples"],
        expected_total_incidence_vec_samples,
    )


def test_gamma_vec_samples():
    # Gamma samples with the given means and rates (so shape rate * mean) on the latent
    # days, and zero on the others and where the mean is zero (day 3 in the second sample)
    rng = np.random.default_rng(0)
    mean_vec_samples = rng.uniform(1.0, 3.0, size=(4, 5))
    mean_vec_samples[:, [1, 4]] = 0.0
    mean_vec_samples[1, 3] = 0.0
    latent_days = np.array([0, 2, 3])
    rate_samples = rng.lognormal(size=4)
    z = rng.normal(size=(4, 3))
    expected = np.zeros((4, 5))
    expected[:, latent_days] = scipy.stats.gamma.ppf(
        scipy.stats.norm.cdf(z),
        a=rate_samples[:, None] * mean_vec_samples[:, latent_days],
        scale=1 / rate_samples[:, None],
    )
    expected[1, 3] = 0.0
    np.testing.assert_allclose(
        evidence._gamma_vec_samples(
            mean_vec_samples=mean_vec_samples,
            rate_samples=rate_samples,
            latent_days=latent_days,
            latent_standard_normal_samples=z,
        ),
        expected,
        rtol=1e-10,
    )


@pytest.mark.parametrize("shape", [0.05, 50.0])
def test_gamma_vec_samples_tails(*, shape):
    # One latent day, with mean 2 shared by every sample, so rate shape / 2
    z = np.linspace(-9, 9, 181)
    values = evidence._gamma_vec_samples(
        mean_vec_samples=np.array([[2.0]]),
        rate_samples=np.full(len(z), shape / 2),
        latent_days=np.array([0]),
        latent_standard_normal_samples=z[:, None],
    )[:, 0]
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0)
    assert np.all(np.diff(values) >= 0)
    # Naive quantile at Phi(9) = 1 (in floating point) is infinite
    assert np.isinf(
        scipy.stats.gamma.ppf(scipy.stats.norm.cdf(9.0), a=shape, scale=2.0 / shape)
    )


def test_unknown_model_raises(*, toy_inputs):
    with pytest.raises(ValueError, match="Unknown model name"):
        evidence.compute_log_evidence(
            model_name="nonexistent",
            **toy_inputs,
            trace=None,
            n_proposal_samples=10,
            rng=np.random.default_rng(0),
        )


@pytest.mark.parametrize("reporting_prob", [0.0, 1.5, np.nan])
def test_invalid_reporting_prob_raises(*, reporting_prob, toy_inputs):
    with pytest.raises(ValueError, match="Reporting probability"):
        evidence.compute_log_evidence(
            model_name="poisson",
            **toy_inputs,
            trace=None,
            n_proposal_samples=10,
            rng=np.random.default_rng(0),
            reporting_prob=reporting_prob,
        )


@pytest.mark.parametrize("reporting_prob", [1.0, 0.5])
def test_reporting_prob_mismatch_raises(*, reporting_prob, toy_inputs):
    # Unreported incidence is in the trace if and only if reporting is incomplete
    rng = np.random.default_rng(0)
    data_vars: dict[str, tuple] = {
        name: (("chain", "draw"), rng.lognormal(size=(1, 20)))
        for name in ["rep_no_before", "rep_no_after"]
    }
    if reporting_prob == 1:
        data_vars["unreported_incidence"] = (
            ("chain", "draw", "local_day"),
            np.zeros((1, 20, 17), dtype=np.int64),
        )
    with pytest.raises(ValueError, match="doesn't match reporting probability"):
        evidence.compute_log_evidence(
            model_name="poisson",
            **toy_inputs,
            trace=types.SimpleNamespace(posterior=xr.Dataset(data_vars)),
            n_proposal_samples=10,
            rng=rng,
            reporting_prob=reporting_prob,
        )


def test_unsupported_prior_raises(*, toy_inputs):
    priors = toy_inputs["priors"] | {
        "dispersion": {"dist": "gamma", "median": 1.0, "quantile_025": 0.2}
    }
    with pytest.raises(ValueError, match="Unsupported prior"):
        evidence._log_prior_samples(
            samples=np.zeros((2, 3)),
            estimated_param_names=_PARAM_NAMES,
            latent_days=None,
            priors=priors,
        )


def test_model_trace_mismatch_raises(*, toy_inputs):
    # Poisson trace (no dispersion) passed with an SSE model
    rng = np.random.default_rng(0)
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), rng.lognormal(size=(1, 20)))
            for name in ["rep_no_before", "rep_no_after"]
        }
    )
    with pytest.raises(TypeError, match="dispersion_samples"):
        evidence.compute_log_evidence(
            model_name="sse",
            **toy_inputs,
            trace=types.SimpleNamespace(posterior=posterior),
            n_proposal_samples=10,
            rng=rng,
        )


@pytest.mark.parametrize(
    ("reported_incidence_vec", "reporting_prob"),
    [([2, 1], 0.5), ([2, 1, 0], 0.2), ([2, 1, 0], 0.5), ([2, 1, 0], 0.95)],
)
def test_bridge_sampling_matches_analytical_under_reporting(
    *, reported_incidence_vec, reporting_prob
):
    # Poisson model with fixed reproduction numbers, a serial interval of exactly one day
    # and the switch on day 2, from exact posterior draws of the unreported counts. By
    # Poisson thinning, day 1's reported and unreported counts Y_1 and U_1 are independent
    # Poisson, so without day 2, U_1's posterior is its prior (and the proposal is
    # one-dimensional). With day 2, P(Y_2 = 0 | U_1) = exp(-rho R_after (Y_1 + U_1)) tilts
    # U_1's posterior to another Poisson, and U_2 is Poisson given U_1
    reported_incidence_vec = np.array(reported_incidence_vec)
    rep_no_before, rep_no_after = 0.8, 0.4
    rng = np.random.default_rng(0)
    n_chains, n_draws = 4, 1000
    day_1_unreported_mean = (
        (1 - reporting_prob) * rep_no_before * reported_incidence_vec[0]
    )
    log_evidence = scipy.stats.poisson.logpmf(
        reported_incidence_vec[1],
        reporting_prob * rep_no_before * reported_incidence_vec[0],
    )
    if len(reported_incidence_vec) == 2:
        unreported_incidence = rng.poisson(
            day_1_unreported_mean, size=(n_chains, n_draws, 1)
        )
    else:
        log_tilt = -reporting_prob * rep_no_after
        day_1_unreported = rng.poisson(
            day_1_unreported_mean * np.exp(log_tilt), size=(n_chains, n_draws)
        )
        day_2_unreported = rng.poisson(
            (1 - reporting_prob)
            * rep_no_after
            * (reported_incidence_vec[1] + day_1_unreported)
        )
        unreported_incidence = np.stack([day_1_unreported, day_2_unreported], axis=-1)
        # Mean of P(Y_2 = 0 | U_1) over U_1's prior
        log_evidence += reported_incidence_vec[1] * log_tilt + (
            day_1_unreported_mean * np.expm1(log_tilt)
        )
    # Fixed parameters are in the trace as deterministics
    posterior = xr.Dataset(
        {
            "rep_no_before": (
                ("chain", "draw"),
                np.full((n_chains, n_draws), rep_no_before),
            ),
            "rep_no_after": (
                ("chain", "draw"),
                np.full((n_chains, n_draws), rep_no_after),
            ),
            "unreported_incidence": (
                ("chain", "draw", "local_day"),
                unreported_incidence,
            ),
        },
        coords={"local_day": np.arange(1, len(reported_incidence_vec))},
    )

    result = evidence.compute_log_evidence(
        model_name="poisson",
        reported_incidence_vec=reported_incidence_vec,
        switch_day=2,
        delay_distributions={
            "serial_interval": DelayDistribution(prob=np.array([1.0]), first_lag=1)
        },
        priors={
            "rep_no_before": {"dist": "fixed", "value": rep_no_before},
            "rep_no_after": {"dist": "fixed", "value": rep_no_after},
        },
        trace=types.SimpleNamespace(posterior=posterior),
        n_proposal_samples=2000,
        rng=rng,
        reporting_prob=reporting_prob,
    )
    assert abs(result["log_evidence"] - log_evidence) < 4 * result["log_evidence_se"]


def test_bridge_sampling_constant_counts():
    # Local cases are infected at lag 3, after the end of the series, so the unreported
    # counts are always zero and the reported counts after day 0 have probability 1. With
    # fixed parameters, every dimension is constant
    n_chains, n_draws = 4, 1000
    posterior = xr.Dataset(
        {
            "rep_no_before": (("chain", "draw"), np.full((n_chains, n_draws), 0.8)),
            "rep_no_after": (("chain", "draw"), np.full((n_chains, n_draws), 0.4)),
            "unreported_incidence": (
                ("chain", "draw", "local_day"),
                np.zeros((n_chains, n_draws, 2), dtype=np.int64),
            ),
        },
        coords={"local_day": [1, 2]},
    )
    result = evidence.compute_log_evidence(
        model_name="poisson",
        reported_incidence_vec=np.array([2, 0, 0]),
        switch_day=2,
        delay_distributions={
            "serial_interval": DelayDistribution(
                prob=np.array([0.0, 0.0, 1.0]), first_lag=1
            )
        },
        priors={
            "rep_no_before": {"dist": "fixed", "value": 0.8},
            "rep_no_after": {"dist": "fixed", "value": 0.4},
        },
        trace=types.SimpleNamespace(posterior=posterior),
        n_proposal_samples=2000,
        rng=np.random.default_rng(0),
        reporting_prob=0.5,
    )
    assert abs(result["log_evidence"]) < 4 * result["log_evidence_se"]


@pytest.mark.parametrize("model_name", ["poisson", "ssd"])
def test_bridge_sampling_matches_quadrature(*, model_name, toy_inputs_fixed_dispersion):
    # For the Poisson model, and the SSD model with fixed dispersion, the likelihood
    # factorises over days before and after the switch, and the priors are independent, so
    # the evidence is a product of 1-D integrals over log R and the posteriors of R before
    # and after the switch are independent (sampled exactly by inverse transform on a grid)
    inputs = toy_inputs_fixed_dispersion
    dispersion = inputs["priors"]["dispersion"]["value"]
    reported_incidence_vec = inputs["reported_incidence_vec"]
    n_days = len(reported_incidence_vec)
    total_inf_vec = np.convolve(
        reported_incidence_vec,
        np.concatenate([[0.0], inputs["delay_distributions"]["serial_interval"].prob]),
    )[:n_days]
    days = np.arange(n_days)
    rng = np.random.default_rng(0)

    log_evidence = 0.0
    posterior = {}
    for name, period_idx in [
        ("rep_no_before", (total_inf_vec > 0) & (days < inputs["switch_day"])),
        ("rep_no_after", (total_inf_vec > 0) & (days >= inputs["switch_day"])),
    ]:
        prior_params = parse_lognormal_prior(name=name, priors=inputs["priors"])
        log_rep_no_grid = prior_params["mu"] + prior_params["sigma"] * np.linspace(
            -8, 8, 20001
        )
        mean = np.exp(log_rep_no_grid)[:, None] * total_inf_vec[period_idx]
        if model_name == "poisson":
            log_likelihood = scipy.stats.poisson.logpmf(
                reported_incidence_vec[period_idx], mean
            )
        else:
            log_likelihood = scipy.stats.nbinom.logpmf(
                reported_incidence_vec[period_idx],
                n=dispersion,
                p=dispersion / (dispersion + mean),
            )
        log_unnormalised_density = scipy.stats.norm.logpdf(
            log_rep_no_grid, loc=prior_params["mu"], scale=prior_params["sigma"]
        ) + log_likelihood.sum(axis=-1)
        log_shift = log_unnormalised_density.max()
        density = np.exp(log_unnormalised_density - log_shift)
        log_evidence += log_shift + np.log(
            scipy.integrate.trapezoid(density, log_rep_no_grid)
        )
        cdf = scipy.integrate.cumulative_trapezoid(density, log_rep_no_grid, initial=0)
        posterior[name] = (
            ("chain", "draw"),
            np.exp(
                np.interp(rng.uniform(size=(4, 1000)), cdf / cdf[-1], log_rep_no_grid)
            ),
        )
    if model_name == "ssd":
        # Fixed parameters are in the trace as deterministics
        posterior["dispersion"] = (("chain", "draw"), np.full((4, 1000), dispersion))

    result = evidence.compute_log_evidence(
        model_name=model_name,
        **inputs,
        trace=types.SimpleNamespace(posterior=xr.Dataset(posterior)),
        n_proposal_samples=2000,
        rng=rng,
    )
    assert abs(result["log_evidence"] - log_evidence) < 4 * result["log_evidence_se"]


def test_bridge_sampling_ssi_matches_prior_sampling(*, toy_inputs):
    rng = np.random.default_rng(0)
    trace = fit_model(
        model_name="ssi",
        **toy_inputs,
        rng=rng,
        chains=4,
        tune=500,
        draws=1000,
        progressbar=False,
        compute_convergence_checks=False,
    )
    result = evidence.compute_log_evidence(
        model_name="ssi",
        **toy_inputs,
        trace=trace,
        n_proposal_samples=2000,
        rng=rng,
    )

    # Reference estimate by prior sampling, with delta-method standard error
    n_samples = 200_000
    unpacked_samples = evidence._unpack_samples(
        samples=_prior_samples(
            estimated_param_names=_PARAM_NAMES,
            n_latents=trace.posterior["u"].shape[-1],
            priors=toy_inputs["priors"],
            n_samples=n_samples,
            rng=rng,
        ),
        estimated_param_names=_PARAM_NAMES,
        fixed_params={},
        latent_days=trace.posterior["u"]["latent_day"].values,
        unreported_days=None,
        reported_incidence_vec=toy_inputs["reported_incidence_vec"],
    )
    log_likelihood_samples = evidence._ssi_log_likelihood_samples(
        reported_incidence_vec=toy_inputs["reported_incidence_vec"],
        switch_day=toy_inputs["switch_day"],
        delay_distributions=toy_inputs["delay_distributions"],
        reporting_prob=1.0,
        **unpacked_samples,
    )
    reference_log_evidence, reference_se = _log_mean_exp_and_se(
        log_values=log_likelihood_samples
    )

    assert abs(result["log_evidence"] - reference_log_evidence) < 4 * np.hypot(
        result["log_evidence_se"], reference_se
    )


@pytest.mark.parametrize("model_name", ["poisson", "ssi"])
def test_bridge_sampling_matches_forward_simulation(*, model_name, toy_inputs):
    # Under-reporting on a short series. Reference estimate by simulating total incidence
    # forward from the day-0 cases with parameters (and gamma latents) drawn from the
    # prior, and averaging the probability of the reported incidence given the total
    inputs = toy_inputs | {
        "reported_incidence_vec": np.array([2, 1, 1, 2, 0, 1, 0, 0]),
        "switch_day": 3,
    }
    reporting_prob = 0.5
    rng = np.random.default_rng(0)
    trace = fit_model(
        model_name=model_name,
        **inputs,
        rng=rng,
        reporting_prob=reporting_prob,
        chains=4,
        # One core, so the chains run in this process: forking it, which earlier tests'
        # numeric libraries may have made multi-threaded, would risk deadlock
        cores=1,
        tune=500,
        draws=1000,
        progressbar=False,
        compute_convergence_checks=False,
    )
    result = evidence.compute_log_evidence(
        model_name=model_name,
        **inputs,
        trace=trace,
        n_proposal_samples=2000,
        rng=rng,
        reporting_prob=reporting_prob,
    )

    reported_incidence_vec = inputs["reported_incidence_vec"]
    n_days = len(reported_incidence_vec)
    serial_interval = inputs["delay_distributions"]["serial_interval"].prob
    n_samples = 1_000_000
    param_samples = {}
    for name in _PARAM_NAMES:
        prior_params = parse_lognormal_prior(name=name, priors=inputs["priors"])
        param_samples[name] = rng.lognormal(
            mean=prior_params["mu"], sigma=prior_params["sigma"], size=n_samples
        )
    total_incidence_vec_samples = np.zeros((n_samples, n_days), dtype=np.int64)
    total_incidence_vec_samples[:, 0] = reported_incidence_vec[0]
    # Each day's cases' total infectiousness: their number, or a gamma latent in SSI
    cohort_inf_vec_samples = np.zeros((n_samples, n_days))
    for t in range(n_days):
        if t > 0:
            lags = np.arange(1, min(t, len(serial_interval)) + 1)
            rep_no_samples = param_samples[
                "rep_no_before" if t < inputs["switch_day"] else "rep_no_after"
            ]
            total_incidence_vec_samples[:, t] = rng.poisson(
                rep_no_samples
                * (cohort_inf_vec_samples[:, t - lags] @ serial_interval[lags - 1])
            )
        if model_name == "poisson":
            cohort_inf_vec_samples[:, t] = total_incidence_vec_samples[:, t]
        else:
            cohort_inf_vec_samples[:, t] = rng.gamma(
                shape=param_samples["dispersion"] * total_incidence_vec_samples[:, t],
                scale=1 / param_samples["dispersion"],
            )
    reference_log_evidence, reference_se = _log_mean_exp_and_se(
        log_values=scipy.stats.binom.logpmf(
            reported_incidence_vec[1:],
            total_incidence_vec_samples[:, 1:],
            reporting_prob,
        ).sum(axis=-1)
    )

    assert abs(result["log_evidence"] - reference_log_evidence) < 4 * np.hypot(
        result["log_evidence_se"], reference_se
    )


@pytest.mark.parametrize(
    ("log_evidence_by_model", "prior_probability_by_model", "expected"),
    [
        ({"sse": -10.0, "ssi": -10.0}, None, {"sse": 0.5, "ssi": 0.5}),
        ({"sse": 0.0, "ssi": np.log(3)}, None, {"sse": 0.25, "ssi": 0.75}),
        # Prior probabilities need not be normalised
        (
            {"sse": -10.0, "ssi": -10.0},
            {"sse": 1.0, "ssi": 3.0},
            {"sse": 0.25, "ssi": 0.75},
        ),
        # Large log evidences don't overflow
        ({"sse": -1e5, "ssi": -1e5 + np.log(3)}, None, {"sse": 0.25, "ssi": 0.75}),
    ],
)
def test_compute_model_probabilities(
    *, log_evidence_by_model, prior_probability_by_model, expected
):
    model_probability_by_model = evidence.compute_model_probabilities(
        log_evidence_by_model=log_evidence_by_model,
        prior_probability_by_model=prior_probability_by_model,
    )
    assert model_probability_by_model == pytest.approx(expected)
    assert list(model_probability_by_model) == list(log_evidence_by_model)


def _prior_samples(*, estimated_param_names, n_latents, priors, n_samples, rng):
    # Prior samples in the bridge sampling layout (log params, then z)
    columns = []
    for name in estimated_param_names:
        prior_params = parse_lognormal_prior(name=name, priors=priors)
        columns.append(
            rng.normal(
                loc=prior_params["mu"],
                scale=prior_params["sigma"],
                size=(n_samples, 1),
            )
        )
    columns.append(rng.normal(size=(n_samples, n_latents)))
    return np.concatenate(columns, axis=-1)


def _log_mean_exp_and_se(*, log_values):
    # Log of the mean of exp(log_values), with the delta-method standard error from the
    # effective sample size of the weights exp(log_values)
    log_sum = scipy.special.logsumexp(log_values)
    ess = np.exp(2 * log_sum - scipy.special.logsumexp(2 * log_values))
    return log_sum - np.log(len(log_values)), np.sqrt(1 / ess - 1 / len(log_values))
