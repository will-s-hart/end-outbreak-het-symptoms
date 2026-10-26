# Note that AI tools were used to generate tests
import types

import numpy as np
import pymc as pm
import pytest
import scipy.integrate
import scipy.special
import scipy.stats
import xarray as xr

from endoutbreakhet import evidence
from endoutbreakhet.inference import _build_pymc_model, fit_model
from endoutbreakhet.utils import parse_lognormal_prior

_PARAM_NAMES = ["rep_no_before", "rep_no_after", "dispersion"]


@pytest.mark.parametrize("fixed_dispersion", [False, True])
def test_log_density_matches_pymc(
    *, model_name, fixed_dispersion, toy_inputs, toy_inputs_fixed_dispersion
):
    # Numpy log prior + log likelihood in the bridge sampling parameterisation (log params,
    # z = Phi^-1(u)) against the PyMC model's log density in its sampling parameterisation
    # (log params, logit(u)), corrected by the log Jacobian of logit(u) -> z
    inputs = toy_inputs_fixed_dispersion if fixed_dispersion else toy_inputs
    model = _build_pymc_model(model_name=model_name, **inputs, reporting_prob=1.0)
    free_rv_names = {rv.name for rv in model.free_RVs}
    estimated_param_names = [name for name in _PARAM_NAMES if name in free_rv_names]
    fixed_params = {
        name: inputs["priors"][name]["value"]
        for name in _PARAM_NAMES
        if name in model.named_vars and name not in free_rv_names
    }
    # Days with gamma latents, as the model has them
    latent_days = (
        np.array(model.coords["latent_day"]) if "u" in model.named_vars else None
    )
    n_latents = 0 if latent_days is None else len(latent_days)
    unconstrained_samples = _prior_unconstrained_samples(
        estimated_param_names=estimated_param_names,
        n_latents=n_latents,
        priors=inputs["priors"],
        n_samples=5,
        rng=np.random.default_rng(0),
    )

    log_prior_samples = evidence._log_prior_samples(
        unconstrained_samples=unconstrained_samples,
        estimated_param_names=estimated_param_names,
        latent_days=latent_days,
        priors=inputs["priors"],
    )
    log_likelihood_samples = getattr(evidence, f"_{model_name}_log_likelihood_samples")(
        reported_incidence_vec=inputs["reported_incidence_vec"],
        switch_day=inputs["switch_day"],
        delay_distributions=inputs["delay_distributions"],
        **evidence._param_and_latent_samples(
            unconstrained_samples=unconstrained_samples,
            estimated_param_names=estimated_param_names,
            fixed_params=fixed_params,
            latent_days=latent_days,
        ),
    )

    logp_fn = model.compile_logp()
    for sample, log_density in zip(
        unconstrained_samples, log_prior_samples + log_likelihood_samples
    ):
        point = {
            f"{name}_log__": sample[i] for i, name in enumerate(estimated_param_names)
        }
        z = sample[len(estimated_param_names) :]
        log_jacobian = 0.0
        if n_latents:
            log_cdf, log_sf = scipy.stats.norm.logcdf(z), scipy.stats.norm.logsf(z)
            point["u_interval__"] = log_cdf - log_sf
            log_jacobian = np.sum(scipy.stats.norm.logpdf(z) - log_cdf - log_sf)
        assert log_density == pytest.approx(logp_fn(point) + log_jacobian, rel=1e-9)


@pytest.mark.parametrize(
    ("estimated_param_names", "n_latents", "fixed_dispersion"),
    [
        (["rep_no_before", "rep_no_after"], 0, False),
        (_PARAM_NAMES, 4, False),
        (["rep_no_before", "rep_no_after"], 4, True),
    ],
)
def test_param_layout_and_unconstrained_posterior_samples(
    *,
    estimated_param_names,
    n_latents,
    fixed_dispersion,
    toy_inputs,
    toy_inputs_fixed_dispersion,
):
    rng = np.random.default_rng(0)
    n_chains, n_draws = 2, 3
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
            rng.gamma(shape=1.0, size=(n_chains, n_draws, 10)),
        )
    trace = types.SimpleNamespace(posterior=xr.Dataset(data_vars, coords=coords))
    inputs = toy_inputs_fixed_dispersion if fixed_dispersion else toy_inputs

    returned_estimated_param_names, fixed_params, latent_days = evidence._param_layout(
        trace=trace, priors=inputs["priors"]
    )
    assert returned_estimated_param_names == estimated_param_names
    assert fixed_params == ({"dispersion": 0.18} if fixed_dispersion else {})
    if n_latents:
        np.testing.assert_array_equal(latent_days, coords["latent_day"])
    else:
        assert latent_days is None

    samples = evidence._unconstrained_posterior_samples(
        trace=trace,
        estimated_param_names=estimated_param_names,
        latent_days=latent_days,
    )
    n_dims = len(estimated_param_names) + n_latents
    assert samples.shape == (n_chains, n_draws, n_dims)

    param_and_latent_samples = evidence._param_and_latent_samples(
        unconstrained_samples=samples.reshape(n_chains * n_draws, n_dims),
        estimated_param_names=estimated_param_names,
        fixed_params=fixed_params,
        latent_days=latent_days,
    )
    for name in [*estimated_param_names, *fixed_params]:
        np.testing.assert_allclose(
            param_and_latent_samples[f"{name}_samples"], data_vars[name][1].ravel()
        )
    if n_latents:
        np.testing.assert_allclose(
            scipy.stats.norm.cdf(
                param_and_latent_samples["latent_standard_normal_samples"]
            ),
            data_vars["u"][1].reshape(n_chains * n_draws, n_latents),
        )
        np.testing.assert_array_equal(
            param_and_latent_samples["latent_days"], coords["latent_day"]
        )
    else:
        assert "latent_standard_normal_samples" not in param_and_latent_samples
        assert "latent_days" not in param_and_latent_samples


def test_gamma_vec_samples():
    # Gamma samples with shape dispersion * mean and rate dispersion on the latent days,
    # and zero on the others
    rng = np.random.default_rng(0)
    mean_vec = np.array([2.0, 0.0, 1.0, 3.0, 0.0])
    latent_days = np.array([0, 2, 3])
    dispersion_samples = rng.lognormal(size=4)
    z = rng.normal(size=(4, 3))
    expected = np.zeros((4, 5))
    expected[:, latent_days] = scipy.stats.gamma.ppf(
        scipy.stats.norm.cdf(z),
        a=dispersion_samples[:, None] * mean_vec[latent_days],
        scale=1 / dispersion_samples[:, None],
    )
    np.testing.assert_allclose(
        evidence._gamma_vec_samples(
            mean_vec=mean_vec,
            dispersion_samples=dispersion_samples,
            latent_days=latent_days,
            latent_standard_normal_samples=z,
        ),
        expected,
        rtol=1e-10,
    )


@pytest.mark.parametrize("shape", [0.05, 50.0])
def test_gamma_vec_samples_tails(*, shape):
    # One latent day, with mean 2, so dispersion shape / 2
    z = np.linspace(-9, 9, 181)
    values = evidence._gamma_vec_samples(
        mean_vec=np.array([2.0]),
        dispersion_samples=np.full(len(z), shape / 2),
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


def test_under_reporting_raises(*, toy_inputs):
    posterior = xr.Dataset(
        {
            "unreported_incidence": (
                ("chain", "draw", "local_day"),
                np.zeros((1, 1, 17), int),
            )
        }
    )
    with pytest.raises(NotImplementedError, match="under-reporting"):
        evidence.compute_log_evidence(
            model_name="poisson",
            **toy_inputs,
            trace=types.SimpleNamespace(posterior=posterior),
            n_proposal_samples=10,
            rng=np.random.default_rng(0),
        )


def test_unsupported_prior_raises(*, toy_inputs):
    priors = toy_inputs["priors"] | {
        "dispersion": {"dist": "gamma", "median": 1.0, "quantile_025": 0.2}
    }
    with pytest.raises(ValueError, match="Unsupported prior"):
        evidence._log_prior_samples(
            unconstrained_samples=np.zeros((2, 3)),
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


def test_bridge_sampling_gaussian():
    rng = np.random.default_rng(0)
    n_dims = 4
    mean = rng.normal(size=n_dims)
    sqrt_cov = rng.normal(size=(n_dims, n_dims))
    cov = sqrt_cov @ sqrt_cov.T + n_dims * np.eye(n_dims)
    log_normaliser = 3.7

    def _log_density(*, unconstrained_samples):
        return log_normaliser + scipy.stats.multivariate_normal.logpdf(
            unconstrained_samples, mean=mean, cov=cov
        )

    result = evidence._bridge_sampling(
        posterior_samples=rng.multivariate_normal(mean, cov, size=(4, 1000)),
        log_density=_log_density,
        n_proposal_samples=2000,
        rng=rng,
    )
    assert abs(result["log_evidence"] - log_normaliser) < 4 * result["log_evidence_se"]


@pytest.mark.parametrize("model_name", ["poisson", "dlo"])
def test_bridge_sampling_matches_quadrature(*, model_name, toy_inputs_fixed_dispersion):
    # For the Poisson model, and the DLO model with fixed dispersion, the likelihood
    # factorises over days before and after the switch, and the priors are independent, so
    # the evidence is a product of 1-D integrals over log R and the posteriors of R before
    # and after the switch are independent (sampled exactly by inverse transform on a grid)
    inputs = toy_inputs_fixed_dispersion
    dispersion = inputs["priors"]["dispersion"]["value"]
    reported_incidence_vec = inputs["reported_incidence_vec"]
    n_days = len(reported_incidence_vec)
    total_inf_vec = np.convolve(
        reported_incidence_vec,
        np.concatenate([[0.0], inputs["delay_distributions"]["serial_interval"]]),
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
    if model_name == "dlo":
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
        step=pm.Slice,
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
    param_and_latent_samples = evidence._param_and_latent_samples(
        unconstrained_samples=_prior_unconstrained_samples(
            estimated_param_names=_PARAM_NAMES,
            n_latents=trace.posterior["u"].shape[-1],
            priors=toy_inputs["priors"],
            n_samples=n_samples,
            rng=rng,
        ),
        estimated_param_names=_PARAM_NAMES,
        fixed_params={},
        latent_days=trace.posterior["u"]["latent_day"].values,
    )
    log_likelihood_samples = evidence._ssi_log_likelihood_samples(
        reported_incidence_vec=toy_inputs["reported_incidence_vec"],
        switch_day=toy_inputs["switch_day"],
        delay_distributions=toy_inputs["delay_distributions"],
        **param_and_latent_samples,
    )
    log_sum_likelihood = scipy.special.logsumexp(log_likelihood_samples)
    ess = np.exp(
        2 * log_sum_likelihood - scipy.special.logsumexp(2 * log_likelihood_samples)
    )
    reference_log_evidence = log_sum_likelihood - np.log(n_samples)
    reference_se = np.sqrt(1 / ess - 1 / n_samples)

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


def _prior_unconstrained_samples(
    *, estimated_param_names, n_latents, priors, n_samples, rng
):
    # Prior samples in the bridge sampling parameterisation (log params, then z)
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
