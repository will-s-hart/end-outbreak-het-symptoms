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


def test_log_density_matches_pymc(*, model_name, toy_inputs):
    # Numpy log prior + log likelihood in the bridge sampling parameterisation (log params,
    # z = Phi^-1(u)) against the PyMC model's log density in its sampling parameterisation
    # (log params, logit(u)), corrected by the log Jacobian of logit(u) -> z
    model = _build_pymc_model(model_name=model_name, **toy_inputs)
    param_names = [name for name in _PARAM_NAMES if name in model.named_vars]
    n_latents = (
        model.initial_point()["u_interval__"].size if "u" in model.named_vars else 0
    )
    unconstrained_samples = _prior_unconstrained_samples(
        param_names=param_names,
        n_latents=n_latents,
        priors=toy_inputs["priors"],
        n_samples=5,
        rng=np.random.default_rng(0),
    )

    log_prior_samples = evidence._log_prior_samples(
        unconstrained_samples=unconstrained_samples,
        param_names=param_names,
        has_latents=n_latents > 0,
        priors=toy_inputs["priors"],
    )
    log_likelihood_samples = getattr(evidence, f"_{model_name}_log_likelihood_samples")(
        incidence_vec=toy_inputs["incidence_vec"],
        switch_day=toy_inputs["switch_day"],
        delay_distributions=toy_inputs["delay_distributions"],
        **evidence._param_and_latent_samples(
            unconstrained_samples=unconstrained_samples,
            param_names=param_names,
            has_latents=n_latents > 0,
        ),
    )

    logp_fn = model.compile_logp()
    for sample, log_density in zip(
        unconstrained_samples, log_prior_samples + log_likelihood_samples
    ):
        point = {f"{name}_log__": sample[i] for i, name in enumerate(param_names)}
        z = sample[len(param_names) :]
        log_jacobian = 0.0
        if n_latents:
            log_cdf, log_sf = scipy.stats.norm.logcdf(z), scipy.stats.norm.logsf(z)
            point["u_interval__"] = log_cdf - log_sf
            log_jacobian = np.sum(scipy.stats.norm.logpdf(z) - log_cdf - log_sf)
        assert log_density == pytest.approx(logp_fn(point) + log_jacobian, rel=1e-9)


@pytest.mark.parametrize(
    ("param_names", "n_latents"),
    [(["rep_no_before", "rep_no_after"], 0), (_PARAM_NAMES, 4)],
)
def test_param_layout_and_unconstrained_posterior_samples(
    *, param_names, n_latents, toy_inputs
):
    rng = np.random.default_rng(0)
    n_chains, n_draws = 2, 3
    data_vars: dict[str, tuple] = {
        name: (("chain", "draw"), rng.lognormal(size=(n_chains, n_draws)))
        for name in param_names
    }
    if n_latents:
        data_vars["u"] = (
            ("chain", "draw", "u_dim_0"),
            rng.uniform(size=(n_chains, n_draws, n_latents)),
        )
        # Deterministics in the trace are not used
        data_vars["cohort_inf_case_days"] = (
            ("chain", "draw", "cohort_inf_case_days_dim_0"),
            rng.gamma(shape=1.0, size=(n_chains, n_draws, n_latents)),
        )
    trace = types.SimpleNamespace(posterior=xr.Dataset(data_vars))

    returned_param_names, has_latents = evidence._param_layout(trace=trace)
    assert returned_param_names == param_names
    assert has_latents == (n_latents > 0)

    samples = evidence._unconstrained_posterior_samples(
        trace=trace, param_names=param_names, has_latents=has_latents
    )
    n_dims = len(param_names) + n_latents
    assert samples.shape == (n_chains, n_draws, n_dims)

    param_and_latent_samples = evidence._param_and_latent_samples(
        unconstrained_samples=samples.reshape(n_chains * n_draws, n_dims),
        param_names=param_names,
        has_latents=has_latents,
    )
    for name in param_names:
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
    else:
        assert "latent_standard_normal_samples" not in param_and_latent_samples


def test_gamma_samples_from_standard_normal_match_naive_in_bulk():
    z = np.linspace(-3, 3, 61)
    np.testing.assert_allclose(
        evidence._gamma_samples_from_standard_normal(
            mean=2.0, shape=0.7, standard_normal_samples=z
        ),
        scipy.stats.gamma.ppf(scipy.stats.norm.cdf(z), a=0.7, scale=2.0 / 0.7),
        rtol=1e-10,
    )


@pytest.mark.parametrize("shape", [0.05, 50.0])
def test_gamma_samples_from_standard_normal_tails(*, shape):
    z = np.linspace(-9, 9, 181)
    values = evidence._gamma_samples_from_standard_normal(
        mean=2.0, shape=shape, standard_normal_samples=z
    )
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


def test_unsupported_prior_raises(*, toy_inputs):
    priors = toy_inputs["priors"] | {
        "dispersion": {"dist": "gamma", "median": 1.0, "quantile_025": 0.2}
    }
    with pytest.raises(ValueError, match="Unsupported prior"):
        evidence._log_prior_samples(
            unconstrained_samples=np.zeros((2, 3)),
            param_names=_PARAM_NAMES,
            has_latents=False,
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


def test_bridge_sampling_poisson_matches_quadrature(*, toy_inputs):
    # The likelihood factorises over days before and after the switch, and the priors are
    # independent, so the evidence is a product of 1-D integrals over log R and the
    # posteriors of R before and after the switch are independent (sampled exactly by
    # inverse transform on a grid)
    incidence_vec = toy_inputs["incidence_vec"]
    n_days = len(incidence_vec)
    total_inf_vec = np.convolve(
        incidence_vec,
        np.concatenate([[0.0], toy_inputs["delay_distributions"]["serial_interval"]]),
    )[:n_days]
    days = np.arange(n_days)
    rng = np.random.default_rng(0)

    log_evidence = 0.0
    posterior = {}
    for name, period_idx in [
        ("rep_no_before", (total_inf_vec > 0) & (days < toy_inputs["switch_day"])),
        ("rep_no_after", (total_inf_vec > 0) & (days >= toy_inputs["switch_day"])),
    ]:
        prior_params = parse_lognormal_prior(name=name, priors=toy_inputs["priors"])
        log_rep_no_grid = prior_params["mu"] + prior_params["sigma"] * np.linspace(
            -8, 8, 20001
        )
        log_unnormalised_density = scipy.stats.norm.logpdf(
            log_rep_no_grid, loc=prior_params["mu"], scale=prior_params["sigma"]
        ) + scipy.stats.poisson.logpmf(
            incidence_vec[period_idx],
            np.exp(log_rep_no_grid)[:, None] * total_inf_vec[period_idx],
        ).sum(axis=-1)
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

    result = evidence.compute_log_evidence(
        model_name="poisson",
        **toy_inputs,
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
            param_names=_PARAM_NAMES,
            n_latents=trace.posterior["u"].shape[-1],
            priors=toy_inputs["priors"],
            n_samples=n_samples,
            rng=rng,
        ),
        param_names=_PARAM_NAMES,
        has_latents=True,
    )
    log_likelihood_samples = evidence._ssi_log_likelihood_samples(
        incidence_vec=toy_inputs["incidence_vec"],
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


def _prior_unconstrained_samples(*, param_names, n_latents, priors, n_samples, rng):
    # Prior samples in the bridge sampling parameterisation (log params, then z)
    columns = []
    for name in param_names:
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
