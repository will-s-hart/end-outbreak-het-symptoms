import arviz_stats
import numpy as np
import scipy.special
import scipy.stats

from endoutbreakhet.utils import (
    delay_matrix,
    get_likelihood_idx,
    get_rep_no_vec_samples,
    parse_lognormal_prior,
)


def compute_log_evidence(
    *,
    model_name,
    incidence_vec,
    switch_day,
    delay_distributions,
    priors,
    trace,
    n_proposal_samples,
    rng,
):
    log_likelihood_funcs = {
        "poisson": _poisson_log_likelihood_samples,
        "sse": _sse_log_likelihood_samples,
        "ssi": _ssi_log_likelihood_samples,
        "dlo": _dlo_log_likelihood_samples,
        "poisson_so": _poisson_so_log_likelihood_samples,
        "sse_so": _sse_so_log_likelihood_samples,
        "ssi_so": _ssi_so_log_likelihood_samples,
    }
    if model_name not in log_likelihood_funcs:
        raise ValueError(f"Unknown model name: {model_name}")

    param_names, has_latents = _param_layout(trace=trace)
    posterior_samples = _unconstrained_posterior_samples(
        trace=trace, param_names=param_names, has_latents=has_latents
    )

    def _log_density(*, unconstrained_samples):
        log_prior_samples = _log_prior_samples(
            unconstrained_samples=unconstrained_samples,
            param_names=param_names,
            has_latents=has_latents,
            priors=priors,
        )
        param_and_latent_samples = _param_and_latent_samples(
            unconstrained_samples=unconstrained_samples,
            param_names=param_names,
            has_latents=has_latents,
        )
        return log_prior_samples + log_likelihood_funcs[model_name](
            incidence_vec=incidence_vec,
            switch_day=switch_day,
            delay_distributions=delay_distributions,
            **param_and_latent_samples,
        )

    return _bridge_sampling(
        posterior_samples=posterior_samples,
        log_density=_log_density,
        n_proposal_samples=n_proposal_samples,
        rng=rng,
    )


def _bridge_sampling(
    *,
    posterior_samples,
    log_density,
    n_proposal_samples,
    rng,
    tol=1e-10,
    max_iterations=1000,
):
    # Log normalising constant of exp(log_density) by bridge sampling with the optimal
    # bridge function (Meng & Wong 1996, as in Gronau et al. 2017), given posterior samples
    # of shape (n_chains, n_draws, n_dims). A multivariate normal proposal is fitted to the
    # first half of each chain, and the iterative scheme is run on the second half, with the
    # posterior sample size in the weights s1 and s2 replaced by its median effective sample
    # size across dimensions. The standard error of the log evidence is the root relative
    # mean squared error of the evidence (Fruhwirth-Schnatter 2004), using the effective
    # sample size of the denominator terms to account for autocorrelation
    n_chains, n_draws, n_dims = posterior_samples.shape
    fit_samples = posterior_samples[:, : n_draws // 2].reshape(-1, n_dims)
    iter_samples = posterior_samples[:, n_draws // 2 :]
    proposal = scipy.stats.multivariate_normal(
        mean=fit_samples.mean(axis=0), cov=np.cov(fit_samples, rowvar=False)
    )
    proposal_samples = proposal.rvs(size=n_proposal_samples, random_state=rng)

    # Log ratios of the unnormalised posterior to the proposal density
    log_ratio_posterior_samples = log_density(
        unconstrained_samples=iter_samples.reshape(-1, n_dims)
    ) - proposal.logpdf(iter_samples.reshape(-1, n_dims))
    log_ratio_proposal_samples = log_density(
        unconstrained_samples=proposal_samples
    ) - proposal.logpdf(proposal_samples)

    n_posterior_eff = np.median(arviz_stats.ess(iter_samples, method="mean"))
    log_s1 = np.log(n_posterior_eff / (n_posterior_eff + n_proposal_samples))
    log_s2 = np.log(n_proposal_samples / (n_posterior_eff + n_proposal_samples))

    def _log_bridge_terms(*, log_evidence):
        # Log terms of the numerator (at proposal samples) and denominator (at posterior
        # samples) of the bridge sampling estimate of the evidence relative to exp(log_evidence)
        log_numerator_terms = (
            log_ratio_proposal_samples
            - log_evidence
            - np.logaddexp(log_s1 + log_ratio_proposal_samples - log_evidence, log_s2)
        )
        log_denominator_terms = -np.logaddexp(
            log_s1 + log_ratio_posterior_samples - log_evidence, log_s2
        )
        return log_numerator_terms, log_denominator_terms

    log_evidence = np.median(log_ratio_posterior_samples)
    for n_iterations in range(1, max_iterations + 1):
        log_numerator_terms, log_denominator_terms = _log_bridge_terms(
            log_evidence=log_evidence
        )
        log_evidence_prev = log_evidence
        log_evidence += (
            scipy.special.logsumexp(log_numerator_terms)
            - np.log(n_proposal_samples)
            - scipy.special.logsumexp(log_denominator_terms)
            + np.log(log_denominator_terms.size)
        )
        if abs(log_evidence - log_evidence_prev) < tol:
            break
    else:
        raise RuntimeError(
            f"Bridge sampling did not converge in {max_iterations} iterations"
        )

    # Terms are bounded above by 1/s1 (numerator) and 1/s2 (denominator)
    log_numerator_terms, log_denominator_terms = _log_bridge_terms(
        log_evidence=log_evidence
    )
    numerator_terms = np.exp(log_numerator_terms)
    denominator_terms = np.exp(log_denominator_terms).reshape(n_chains, -1)
    relative_mse = np.var(numerator_terms, ddof=1) / (
        n_proposal_samples * np.mean(numerator_terms) ** 2
    ) + np.var(denominator_terms, ddof=1) / (
        arviz_stats.ess(denominator_terms, method="mean")
        * np.mean(denominator_terms) ** 2
    )
    return {
        "log_evidence": float(log_evidence),
        "log_evidence_se": float(np.sqrt(relative_mse)),
        "n_iterations": n_iterations,
    }


def _poisson_log_likelihood_samples(
    *,
    incidence_vec,
    switch_day,
    delay_distributions,
    rep_no_before_samples,
    rep_no_after_samples,
):
    n_days = len(incidence_vec)
    serial_interval_matrix = delay_matrix(
        dist_vec=delay_distributions["serial_interval"], n_days=n_days, first_lag=1
    )
    total_inf_vec = serial_interval_matrix @ incidence_vec
    likelihood_idx = get_likelihood_idx(
        incidence_vec=incidence_vec, force_vec=total_inf_vec
    )

    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )

    expected_incidence_local_samples = rep_no_vec_samples * total_inf_vec

    return scipy.stats.poisson.logpmf(
        k=incidence_vec[likelihood_idx],
        mu=expected_incidence_local_samples[:, likelihood_idx],
    ).sum(axis=-1)


def _sse_log_likelihood_samples(
    *,
    incidence_vec,
    switch_day,
    delay_distributions,
    rep_no_before_samples,
    rep_no_after_samples,
    dispersion_samples,
):
    n_days = len(incidence_vec)
    serial_interval_matrix = delay_matrix(
        dist_vec=delay_distributions["serial_interval"], n_days=n_days, first_lag=1
    )
    total_inf_vec = serial_interval_matrix @ incidence_vec
    likelihood_idx = get_likelihood_idx(
        incidence_vec=incidence_vec, force_vec=total_inf_vec
    )

    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )

    expected_incidence_local_samples = rep_no_vec_samples * total_inf_vec

    return _nbinom_logpmf(
        k=incidence_vec[likelihood_idx],
        mean=expected_incidence_local_samples[:, likelihood_idx],
        dispersion=dispersion_samples[:, None] * total_inf_vec[likelihood_idx],
    ).sum(axis=-1)


def _ssi_log_likelihood_samples(
    *,
    incidence_vec,
    switch_day,
    delay_distributions,
    rep_no_before_samples,
    rep_no_after_samples,
    dispersion_samples,
    latent_standard_normal_samples,
):
    n_days = len(incidence_vec)
    serial_interval_matrix = delay_matrix(
        dist_vec=delay_distributions["serial_interval"], n_days=n_days, first_lag=1
    )
    likelihood_idx = get_likelihood_idx(
        incidence_vec=incidence_vec, force_vec=serial_interval_matrix @ incidence_vec
    )
    nonzero_incidence_idx = incidence_vec > 0

    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )

    cohort_inf_case_day_samples = _gamma_samples_from_standard_normal(
        mean=incidence_vec[nonzero_incidence_idx],
        shape=dispersion_samples[:, None] * incidence_vec[nonzero_incidence_idx],
        standard_normal_samples=latent_standard_normal_samples,
    )
    total_inf_vec_samples = (
        cohort_inf_case_day_samples @ serial_interval_matrix[:, nonzero_incidence_idx].T
    )
    expected_incidence_local_samples = rep_no_vec_samples * total_inf_vec_samples

    return scipy.stats.poisson.logpmf(
        k=incidence_vec[likelihood_idx],
        mu=expected_incidence_local_samples[:, likelihood_idx],
    ).sum(axis=-1)


def _dlo_log_likelihood_samples(
    *,
    incidence_vec,
    switch_day,
    delay_distributions,
    rep_no_before_samples,
    rep_no_after_samples,
    dispersion_samples,
):
    n_days = len(incidence_vec)
    serial_interval_matrix = delay_matrix(
        dist_vec=delay_distributions["serial_interval"], n_days=n_days, first_lag=1
    )
    total_inf_vec = serial_interval_matrix @ incidence_vec
    likelihood_idx = get_likelihood_idx(
        incidence_vec=incidence_vec, force_vec=total_inf_vec
    )

    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )

    expected_incidence_local_samples = rep_no_vec_samples * total_inf_vec

    return _nbinom_logpmf(
        k=incidence_vec[likelihood_idx],
        mean=expected_incidence_local_samples[:, likelihood_idx],
        dispersion=dispersion_samples[:, None],
    ).sum(axis=-1)


def _poisson_so_log_likelihood_samples(
    *,
    incidence_vec,
    switch_day,
    delay_distributions,
    rep_no_before_samples,
    rep_no_after_samples,
):
    n_days = len(incidence_vec)
    tost_matrix = delay_matrix(
        dist_vec=delay_distributions["tost"], n_days=n_days, first_lag=0
    )
    inc_period_matrix = delay_matrix(
        dist_vec=delay_distributions["inc_period"], n_days=n_days, first_lag=1
    )
    total_inf_vec = tost_matrix @ incidence_vec
    likelihood_idx = get_likelihood_idx(
        incidence_vec=incidence_vec, force_vec=inc_period_matrix @ total_inf_vec
    )

    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )

    expected_incidence_local_samples = (
        rep_no_vec_samples * total_inf_vec
    ) @ inc_period_matrix.T

    return scipy.stats.poisson.logpmf(
        k=incidence_vec[likelihood_idx],
        mu=expected_incidence_local_samples[:, likelihood_idx],
    ).sum(axis=-1)


def _sse_so_log_likelihood_samples(
    *,
    incidence_vec,
    switch_day,
    delay_distributions,
    rep_no_before_samples,
    rep_no_after_samples,
    dispersion_samples,
    latent_standard_normal_samples,
):
    n_days = len(incidence_vec)
    tost_matrix = delay_matrix(
        dist_vec=delay_distributions["tost"], n_days=n_days, first_lag=0
    )
    inc_period_matrix = delay_matrix(
        dist_vec=delay_distributions["inc_period"], n_days=n_days, first_lag=1
    )
    expected_total_inf_vec = tost_matrix @ incidence_vec
    nonzero_total_inf_idx = expected_total_inf_vec > 0
    likelihood_idx = get_likelihood_idx(
        incidence_vec=incidence_vec,
        force_vec=inc_period_matrix @ expected_total_inf_vec,
    )

    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )

    total_inf_nonzero_day_samples = _gamma_samples_from_standard_normal(
        mean=expected_total_inf_vec[nonzero_total_inf_idx],
        shape=dispersion_samples[:, None]
        * expected_total_inf_vec[nonzero_total_inf_idx],
        standard_normal_samples=latent_standard_normal_samples,
    )
    expected_incidence_local_samples = (
        total_inf_nonzero_day_samples * rep_no_vec_samples[:, nonzero_total_inf_idx]
    ) @ inc_period_matrix[:, nonzero_total_inf_idx].T

    return scipy.stats.poisson.logpmf(
        k=incidence_vec[likelihood_idx],
        mu=expected_incidence_local_samples[:, likelihood_idx],
    ).sum(axis=-1)


def _ssi_so_log_likelihood_samples(
    *,
    incidence_vec,
    switch_day,
    delay_distributions,
    rep_no_before_samples,
    rep_no_after_samples,
    dispersion_samples,
    latent_standard_normal_samples,
):
    n_days = len(incidence_vec)
    tost_matrix = delay_matrix(
        dist_vec=delay_distributions["tost"], n_days=n_days, first_lag=0
    )
    inc_period_matrix = delay_matrix(
        dist_vec=delay_distributions["inc_period"], n_days=n_days, first_lag=1
    )
    likelihood_idx = get_likelihood_idx(
        incidence_vec=incidence_vec,
        force_vec=inc_period_matrix @ (tost_matrix @ incidence_vec),
    )
    nonzero_incidence_idx = incidence_vec > 0

    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )

    cohort_inf_case_day_samples = _gamma_samples_from_standard_normal(
        mean=incidence_vec[nonzero_incidence_idx],
        shape=dispersion_samples[:, None] * incidence_vec[nonzero_incidence_idx],
        standard_normal_samples=latent_standard_normal_samples,
    )
    total_inf_vec_samples = (
        cohort_inf_case_day_samples @ tost_matrix[:, nonzero_incidence_idx].T
    )
    expected_infection_incidence_local_samples = (
        rep_no_vec_samples * total_inf_vec_samples
    )
    expected_incidence_local_samples = (
        expected_infection_incidence_local_samples @ inc_period_matrix.T
    )

    return scipy.stats.poisson.logpmf(
        k=incidence_vec[likelihood_idx],
        mu=expected_incidence_local_samples[:, likelihood_idx],
    ).sum(axis=-1)


def _param_layout(*, trace):
    # Names of the parameters, and whether there are gamma latents, from the variables in
    # the trace. In the bridge sampling parameterisation, the last axis holds logs of the
    # parameters, then standard normal samples z = Phi^-1(u) underlying the latents (if any)
    param_names = [
        name
        for name in ("rep_no_before", "rep_no_after", "dispersion")
        if name in trace.posterior
    ]
    has_latents = "u" in trace.posterior
    return param_names, has_latents


def _unconstrained_posterior_samples(*, trace, param_names, has_latents):
    # Posterior samples in the bridge sampling parameterisation (see _param_layout), of
    # shape (n_chains, n_draws, n_dims)
    posterior = trace.posterior
    columns = [np.log(posterior[name].values)[..., None] for name in param_names]
    if has_latents:
        columns.append(scipy.stats.norm.ppf(posterior["u"].values))
    return np.concatenate(columns, axis=-1)


def _param_and_latent_samples(*, unconstrained_samples, param_names, has_latents):
    # Parameter and latent samples, as keyword arguments for the log-likelihood functions,
    # from samples of shape (n_samples, n_dims) in the bridge sampling parameterisation
    # (see _param_layout)
    param_and_latent_samples = {
        f"{name}_samples": np.exp(unconstrained_samples[:, i])
        for i, name in enumerate(param_names)
    }
    if has_latents:
        param_and_latent_samples["latent_standard_normal_samples"] = (
            unconstrained_samples[:, len(param_names) :]
        )
    return param_and_latent_samples


def _log_prior_samples(*, unconstrained_samples, param_names, has_latents, priors):
    # Log prior density of samples of shape (n_samples, n_dims) in the bridge sampling
    # parameterisation (see _param_layout). If X ~ LogNormal(mu, sigma) then
    # log X ~ Normal(mu, sigma), and z ~ Normal(0, 1)
    log_prior_samples = np.zeros(len(unconstrained_samples))
    for i, name in enumerate(param_names):
        prior_params = parse_lognormal_prior(name=name, priors=priors)
        log_prior_samples += scipy.stats.norm.logpdf(
            unconstrained_samples[:, i],
            loc=prior_params["mu"],
            scale=prior_params["sigma"],
        )
    if has_latents:
        log_prior_samples += scipy.stats.norm.logpdf(
            unconstrained_samples[:, len(param_names) :]
        ).sum(axis=-1)
    return log_prior_samples


def _nbinom_logpmf(*, k, mean, dispersion):
    # Same parameterisation as pm.NegativeBinomial(mu=mean, alpha=dispersion)
    return scipy.stats.nbinom.logpmf(
        k, n=dispersion, p=dispersion / (dispersion + mean)
    )


def _gamma_samples_from_standard_normal(*, mean, shape, standard_normal_samples):
    # Gamma samples with given mean and shape, as the gamma quantile at Phi(z) for standard
    # normal samples z. Uses the upper tail for z >= 0, where Phi(z) rounds to 1
    z = standard_normal_samples
    scale = mean / shape
    return np.where(
        z < 0,
        scipy.stats.gamma.ppf(scipy.stats.norm.cdf(z), a=shape, scale=scale),
        scipy.stats.gamma.isf(scipy.stats.norm.sf(z), a=shape, scale=scale),
    )
