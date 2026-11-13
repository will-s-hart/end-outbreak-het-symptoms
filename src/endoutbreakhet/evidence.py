import numpy as np
import scipy.special
import scipy.stats

from endoutbreakhet.bridge_sampling import bridge_sampling
from endoutbreakhet.utils import (
    delay_matrix,
    get_rep_no_vec_samples,
    parse_lognormal_prior,
    validate_delay_distributions,
)


def compute_log_evidence(
    *,
    model_name,
    reported_incidence_vec,
    switch_day,
    delay_distributions,
    priors,
    trace,
    n_proposal_samples,
    rng,
    reporting_prob=1.0,
):
    # Log evidence of the model given reported_incidence_vec, when each case is reported
    # independently with probability reporting_prob, by bridge sampling over the
    # parameters, any gamma latents and any unreported case counts, as in the notes.
    # reporting_prob must be the one the trace was fitted with: the trace only shows
    # whether reporting was complete
    log_likelihood_funcs = {
        "poisson": _poisson_log_likelihood_samples,
        "sse": _sse_log_likelihood_samples,
        "ssi": _ssi_log_likelihood_samples,
        "ssd": _ssd_log_likelihood_samples,
        "poisson_so": _poisson_so_log_likelihood_samples,
        "sse_so": _sse_so_log_likelihood_samples,
        "ssi_so": _ssi_so_log_likelihood_samples,
    }
    if model_name not in log_likelihood_funcs:
        raise ValueError(f"Unknown model name: {model_name}")
    if not 0 < reporting_prob <= 1:
        raise ValueError(f"Reporting probability must be in (0, 1]: {reporting_prob}")
    validate_delay_distributions(
        model_name=model_name, delay_distributions=delay_distributions
    )
    if (reporting_prob < 1) != ("unreported_incidence" in trace.posterior):
        raise ValueError(
            f"Trace doesn't match reporting probability {reporting_prob}: unreported "
            "incidence is latent if and only if reporting is incomplete"
        )

    estimated_param_names, fixed_params, latent_days, unreported_days = _sample_layout(
        trace=trace, priors=priors
    )
    posterior_samples = _posterior_samples(
        trace=trace,
        estimated_param_names=estimated_param_names,
        latent_days=latent_days,
        unreported_days=unreported_days,
    )

    def _log_density(*, samples):
        log_prior_samples = _log_prior_samples(
            samples=samples,
            estimated_param_names=estimated_param_names,
            latent_days=latent_days,
            priors=priors,
        )
        unpacked_samples = _unpack_samples(
            samples=samples,
            estimated_param_names=estimated_param_names,
            fixed_params=fixed_params,
            latent_days=latent_days,
            unreported_days=unreported_days,
            reported_incidence_vec=reported_incidence_vec,
        )
        return log_prior_samples + log_likelihood_funcs[model_name](
            reported_incidence_vec=reported_incidence_vec,
            switch_day=switch_day,
            delay_distributions=delay_distributions,
            reporting_prob=reporting_prob,
            **unpacked_samples,
        )

    return bridge_sampling(
        posterior_samples=posterior_samples,
        log_density=_log_density,
        n_proposal_samples=n_proposal_samples,
        rng=rng,
        n_count_dims=0 if unreported_days is None else len(unreported_days),
    )


def compute_model_probabilities(
    *, log_evidence_by_model, prior_probability_by_model=None
):
    # Posterior model probabilities from the log evidences, given prior model probabilities
    # (uniform if None, and need not be normalised)
    model_names = list(log_evidence_by_model)
    log_weights = np.array(
        [log_evidence_by_model[name] for name in model_names], dtype=float
    )
    if prior_probability_by_model is not None:
        log_weights += np.log(
            [prior_probability_by_model[name] for name in model_names]
        )
    return {
        name: float(prob)
        for name, prob in zip(
            model_names, scipy.special.softmax(log_weights), strict=True
        )
    }


def _poisson_log_likelihood_samples(
    *,
    reported_incidence_vec,
    switch_day,
    delay_distributions,
    reporting_prob,
    total_incidence_vec_samples,
    rep_no_before_samples,
    rep_no_after_samples,
):
    n_days = len(reported_incidence_vec)
    serial_interval_matrix = delay_matrix(
        delay=delay_distributions["serial_interval"], n_days=n_days
    )

    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )

    total_inf_vec_samples = total_incidence_vec_samples @ serial_interval_matrix.T
    expected_incidence_local_samples = rep_no_vec_samples * total_inf_vec_samples

    return _incidence_log_likelihood_samples(
        reported_incidence_vec=reported_incidence_vec,
        reporting_prob=reporting_prob,
        total_incidence_vec_samples=total_incidence_vec_samples,
        expected_incidence_local_samples=expected_incidence_local_samples,
    )


def _sse_log_likelihood_samples(
    *,
    reported_incidence_vec,
    switch_day,
    delay_distributions,
    reporting_prob,
    total_incidence_vec_samples,
    rep_no_before_samples,
    rep_no_after_samples,
    dispersion_samples,
):
    n_days = len(reported_incidence_vec)
    serial_interval_matrix = delay_matrix(
        delay=delay_distributions["serial_interval"], n_days=n_days
    )

    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )

    total_inf_vec_samples = total_incidence_vec_samples @ serial_interval_matrix.T
    expected_incidence_local_samples = rep_no_vec_samples * total_inf_vec_samples

    return _incidence_log_likelihood_samples(
        reported_incidence_vec=reported_incidence_vec,
        reporting_prob=reporting_prob,
        total_incidence_vec_samples=total_incidence_vec_samples,
        expected_incidence_local_samples=expected_incidence_local_samples,
        incidence_dispersion_samples=dispersion_samples[:, None]
        * total_inf_vec_samples,
    )


def _ssi_log_likelihood_samples(
    *,
    reported_incidence_vec,
    switch_day,
    delay_distributions,
    reporting_prob,
    total_incidence_vec_samples,
    rep_no_before_samples,
    rep_no_after_samples,
    dispersion_samples,
    latent_days,
    latent_standard_normal_samples,
):
    n_days = len(reported_incidence_vec)
    serial_interval_matrix = delay_matrix(
        delay=delay_distributions["serial_interval"], n_days=n_days
    )

    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )

    cohort_inf_vec_samples = _gamma_vec_samples(
        mean_vec_samples=total_incidence_vec_samples,
        rate_samples=dispersion_samples,
        latent_days=latent_days,
        latent_standard_normal_samples=latent_standard_normal_samples,
    )
    total_inf_vec_samples = cohort_inf_vec_samples @ serial_interval_matrix.T
    expected_incidence_local_samples = rep_no_vec_samples * total_inf_vec_samples

    return _incidence_log_likelihood_samples(
        reported_incidence_vec=reported_incidence_vec,
        reporting_prob=reporting_prob,
        total_incidence_vec_samples=total_incidence_vec_samples,
        expected_incidence_local_samples=expected_incidence_local_samples,
    )


def _ssd_log_likelihood_samples(
    *,
    reported_incidence_vec,
    switch_day,
    delay_distributions,
    reporting_prob,
    total_incidence_vec_samples,
    rep_no_before_samples,
    rep_no_after_samples,
    dispersion_samples,
):
    n_days = len(reported_incidence_vec)
    serial_interval_matrix = delay_matrix(
        delay=delay_distributions["serial_interval"], n_days=n_days
    )

    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )

    total_inf_vec_samples = total_incidence_vec_samples @ serial_interval_matrix.T
    expected_incidence_local_samples = rep_no_vec_samples * total_inf_vec_samples

    return _incidence_log_likelihood_samples(
        reported_incidence_vec=reported_incidence_vec,
        reporting_prob=reporting_prob,
        total_incidence_vec_samples=total_incidence_vec_samples,
        expected_incidence_local_samples=expected_incidence_local_samples,
        incidence_dispersion_samples=dispersion_samples[:, None],
    )


def _poisson_so_log_likelihood_samples(
    *,
    reported_incidence_vec,
    switch_day,
    delay_distributions,
    reporting_prob,
    total_incidence_vec_samples,
    rep_no_before_samples,
    rep_no_after_samples,
):
    n_days = len(reported_incidence_vec)
    tost_matrix = delay_matrix(delay=delay_distributions["tost"], n_days=n_days)
    inc_period_matrix = delay_matrix(
        delay=delay_distributions["inc_period"], n_days=n_days
    )

    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )

    total_inf_vec_samples = total_incidence_vec_samples @ tost_matrix.T
    expected_incidence_local_samples = (
        rep_no_vec_samples * total_inf_vec_samples
    ) @ inc_period_matrix.T

    return _incidence_log_likelihood_samples(
        reported_incidence_vec=reported_incidence_vec,
        reporting_prob=reporting_prob,
        total_incidence_vec_samples=total_incidence_vec_samples,
        expected_incidence_local_samples=expected_incidence_local_samples,
    )


def _sse_so_log_likelihood_samples(
    *,
    reported_incidence_vec,
    switch_day,
    delay_distributions,
    reporting_prob,
    total_incidence_vec_samples,
    rep_no_before_samples,
    rep_no_after_samples,
    dispersion_samples,
    latent_days,
    latent_standard_normal_samples,
):
    n_days = len(reported_incidence_vec)
    tost_matrix = delay_matrix(delay=delay_distributions["tost"], n_days=n_days)
    inc_period_matrix = delay_matrix(
        delay=delay_distributions["inc_period"], n_days=n_days
    )

    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )

    expected_total_inf_vec_samples = total_incidence_vec_samples @ tost_matrix.T
    total_inf_vec_samples = _gamma_vec_samples(
        mean_vec_samples=expected_total_inf_vec_samples,
        rate_samples=dispersion_samples,
        latent_days=latent_days,
        latent_standard_normal_samples=latent_standard_normal_samples,
    )
    expected_incidence_local_samples = (
        rep_no_vec_samples * total_inf_vec_samples
    ) @ inc_period_matrix.T

    return _incidence_log_likelihood_samples(
        reported_incidence_vec=reported_incidence_vec,
        reporting_prob=reporting_prob,
        total_incidence_vec_samples=total_incidence_vec_samples,
        expected_incidence_local_samples=expected_incidence_local_samples,
    )


def _ssi_so_log_likelihood_samples(
    *,
    reported_incidence_vec,
    switch_day,
    delay_distributions,
    reporting_prob,
    total_incidence_vec_samples,
    rep_no_before_samples,
    rep_no_after_samples,
    dispersion_samples,
    latent_days,
    latent_standard_normal_samples,
):
    n_days = len(reported_incidence_vec)
    tost_matrix = delay_matrix(delay=delay_distributions["tost"], n_days=n_days)
    inc_period_matrix = delay_matrix(
        delay=delay_distributions["inc_period"], n_days=n_days
    )

    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )

    cohort_inf_vec_samples = _gamma_vec_samples(
        mean_vec_samples=total_incidence_vec_samples,
        rate_samples=dispersion_samples,
        latent_days=latent_days,
        latent_standard_normal_samples=latent_standard_normal_samples,
    )
    total_inf_vec_samples = cohort_inf_vec_samples @ tost_matrix.T
    expected_infection_incidence_local_samples = (
        rep_no_vec_samples * total_inf_vec_samples
    )
    expected_incidence_local_samples = (
        expected_infection_incidence_local_samples @ inc_period_matrix.T
    )

    return _incidence_log_likelihood_samples(
        reported_incidence_vec=reported_incidence_vec,
        reporting_prob=reporting_prob,
        total_incidence_vec_samples=total_incidence_vec_samples,
        expected_incidence_local_samples=expected_incidence_local_samples,
    )


def _incidence_log_likelihood_samples(
    *,
    reported_incidence_vec,
    reporting_prob,
    total_incidence_vec_samples,
    expected_incidence_local_samples,
    incidence_dispersion_samples=None,
):
    # Log probability, summed over days after day 0 (whose cases are imported, and taken to
    # be reported), of the total incidence (of shape (n_samples, n_days), or (1, n_days)
    # if shared by every sample), and of the reported incidence given the total when each
    # case is reported with probability reporting_prob, as in the PyMC model. Total
    # incidence is Poisson with mean expected_incidence_local_samples (or negative binomial
    # with dispersion incidence_dispersion_samples, if given), which is a point mass at
    # zero where the mean is zero, so with complete reporting, days with zero force of
    # infection/disease contribute nothing. A NaN or infinite mean (from a gamma latent
    # that's infinite in floating point) has zero density
    total_incidence_vec_samples = total_incidence_vec_samples[:, 1:]
    expected_incidence_samples = expected_incidence_local_samples[:, 1:]
    if incidence_dispersion_samples is None:
        total_incidence_log_prob_samples = scipy.stats.poisson.logpmf(
            total_incidence_vec_samples, expected_incidence_samples
        )
    else:
        incidence_dispersion_samples = np.broadcast_to(
            incidence_dispersion_samples, expected_incidence_local_samples.shape
        )[:, 1:]
        # The SSE model's dispersion is zero where its mean is, which would give NaN, but
        # with zero mean, any positive dispersion gives the point mass at zero
        total_incidence_log_prob_samples = _nbinom_logpmf(
            k=total_incidence_vec_samples,
            mean=expected_incidence_samples,
            dispersion=np.where(
                incidence_dispersion_samples > 0, incidence_dispersion_samples, 1
            ),
        )
    total_incidence_log_prob_samples = np.where(
        np.isfinite(expected_incidence_samples),
        total_incidence_log_prob_samples,
        -np.inf,
    )
    reporting_log_prob_samples = scipy.stats.binom.logpmf(
        reported_incidence_vec[1:], total_incidence_vec_samples, reporting_prob
    )
    return (total_incidence_log_prob_samples + reporting_log_prob_samples).sum(axis=-1)


def _nbinom_logpmf(*, k, mean, dispersion):
    # Same parameterisation as pm.NegativeBinomial(mu=mean, alpha=dispersion)
    return scipy.stats.nbinom.logpmf(
        k, n=dispersion, p=dispersion / (dispersion + mean)
    )


def _gamma_vec_samples(
    *, mean_vec_samples, rate_samples, latent_days, latent_standard_normal_samples
):
    # Gamma latents on every day, with means of shape (n_samples, n_days) (or (1, n_days)
    # if shared by every sample) and rates of shape (n_samples,), so shape rate * mean, as
    # the gamma quantile at Phi(z) for standard normal samples z on the latent days, and
    # zero on the other days and wherever the mean is zero (outside the latent days, it
    # is). Uses the upper tail for z >= 0, where Phi(z) rounds to 1
    z = latent_standard_normal_samples
    shape = rate_samples[:, None] * mean_vec_samples[:, latent_days]
    # The quantile is NaN at zero shape, so zero shapes are replaced in the unused branch
    safe_shape = np.where(shape > 0, shape, 1)
    scale = 1 / rate_samples[:, None]
    gamma_vec_samples = np.zeros((len(rate_samples), mean_vec_samples.shape[1]))
    gamma_vec_samples[:, latent_days] = np.where(
        shape > 0,
        np.where(
            z < 0,
            scipy.stats.gamma.ppf(scipy.stats.norm.cdf(z), a=safe_shape, scale=scale),
            scipy.stats.gamma.isf(scipy.stats.norm.sf(z), a=safe_shape, scale=scale),
        ),
        0,
    )
    return gamma_vec_samples


def _sample_layout(*, trace, priors):
    # Names of the estimated parameters, values of the fixed parameters, the days with
    # gamma latents and the days with unreported case counts (each None if there are
    # none), from the variables in the trace (where fixed parameters are deterministics).
    # In the bridge sampling layout, the last axis holds logs of the estimated parameters,
    # then standard normal samples z = Phi^-1(u) underlying the latents (if any), then the
    # unreported case counts (if any)
    model_param_names = [
        name
        for name in ("rep_no_before", "rep_no_after", "dispersion")
        if name in trace.posterior
    ]
    fixed_params = {
        name: priors[name]["value"]
        for name in model_param_names
        if priors[name].get("dist", None) == "fixed"
    }
    estimated_param_names = [
        name for name in model_param_names if name not in fixed_params
    ]
    latent_days = (
        trace.posterior["u"]["latent_day"].values if "u" in trace.posterior else None
    )
    unreported_days = (
        trace.posterior["unreported_incidence"]["local_day"].values
        if "unreported_incidence" in trace.posterior
        else None
    )
    return estimated_param_names, fixed_params, latent_days, unreported_days


def _posterior_samples(*, trace, estimated_param_names, latent_days, unreported_days):
    # Posterior samples in the bridge sampling layout (see _sample_layout), of shape
    # (n_chains, n_draws, n_dims)
    posterior = trace.posterior
    columns = [
        np.log(posterior[name].values)[..., None] for name in estimated_param_names
    ]
    if latent_days is not None:
        columns.append(scipy.stats.norm.ppf(posterior["u"].values))
    if unreported_days is not None:
        columns.append(posterior["unreported_incidence"].values)
    return np.concatenate(columns, axis=-1)


def _log_prior_samples(*, samples, estimated_param_names, latent_days, priors):
    # Log prior density of samples of shape (n_samples, n_dims) in the bridge sampling
    # layout (see _sample_layout). If X ~ LogNormal(mu, sigma) then log X ~ Normal(mu,
    # sigma), and z ~ Normal(0, 1). Unreported case counts get a flat prior, as their
    # distribution is in the likelihood
    n_params = len(estimated_param_names)
    log_prior_samples = np.zeros(len(samples))
    for i, name in enumerate(estimated_param_names):
        prior_params = parse_lognormal_prior(name=name, priors=priors)
        log_prior_samples += scipy.stats.norm.logpdf(
            samples[:, i], loc=prior_params["mu"], scale=prior_params["sigma"]
        )
    if latent_days is not None:
        log_prior_samples += scipy.stats.norm.logpdf(
            samples[:, n_params : n_params + len(latent_days)]
        ).sum(axis=-1)
    return log_prior_samples


def _unpack_samples(
    *,
    samples,
    estimated_param_names,
    fixed_params,
    latent_days,
    unreported_days,
    reported_incidence_vec,
):
    # Parameter and latent samples (with the days the latents are on) and total incidence
    # samples, as keyword arguments for the log-likelihood functions, from samples of shape
    # (n_samples, n_dims) in the bridge sampling layout (see _sample_layout), with each
    # fixed parameter's value repeated for every sample. Without unreported cases, the
    # total incidence is the reported incidence, as one row shared by every sample
    n_params = len(estimated_param_names)
    n_latents = 0 if latent_days is None else len(latent_days)
    unpacked_samples = {
        f"{name}_samples": np.exp(samples[:, i])
        for i, name in enumerate(estimated_param_names)
    }
    for name, value in fixed_params.items():
        unpacked_samples[f"{name}_samples"] = np.full(len(samples), value)
    if latent_days is not None:
        unpacked_samples["latent_days"] = latent_days
        unpacked_samples["latent_standard_normal_samples"] = samples[
            :, n_params : n_params + n_latents
        ]
    if unreported_days is None:
        total_incidence_vec_samples = reported_incidence_vec[None, :]
    else:
        total_incidence_vec_samples = np.tile(reported_incidence_vec, (len(samples), 1))
        total_incidence_vec_samples[:, unreported_days] += samples[
            :, n_params + n_latents :
        ].astype(np.int64)
    unpacked_samples["total_incidence_vec_samples"] = total_incidence_vec_samples
    return unpacked_samples
