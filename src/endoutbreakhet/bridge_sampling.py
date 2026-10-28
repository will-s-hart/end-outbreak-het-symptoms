import arviz_stats
import numpy as np
import scipy.special
import scipy.stats


def bridge_sampling(
    *,
    posterior_samples,
    log_density,
    n_proposal_samples,
    rng,
    n_count_dims=0,
    tol=1e-10,
    max_iterations=1000,
):
    # Log normalising constant of exp(log_density) by bridge sampling (see
    # _bridge_estimate), given posterior samples of shape (n_chains, n_draws, n_dims). The
    # last n_count_dims dimensions are nonnegative integer counts, which are summed over
    # rather than integrated, after mapping them to continuous variables (see
    # _dequantised_samples_and_density). A multivariate normal proposal is fitted to the
    # first half of each chain, and the iterative scheme is run on the second half, whose
    # effective sample size is taken as the median across dimensions (before
    # dequantisation, which would inflate it, and leaving out dimensions that are constant
    # there). Every dimension can only be constant if they're all counts (a constant
    # continuous dimension would make the proposal degenerate), which dequantisation
    # turns into independent samples, so the effective sample size is the sample size
    n_chains, n_draws, n_dims = posterior_samples.shape
    fit_samples = posterior_samples[:, : n_draws // 2].reshape(-1, n_dims)
    iter_samples = posterior_samples[:, n_draws // 2 :]

    is_constant_vec = np.all(iter_samples == iter_samples[:1, :1], axis=(0, 1))
    if is_constant_vec.all():
        n_posterior_eff = iter_samples.shape[0] * iter_samples.shape[1]
    else:
        n_posterior_eff = np.median(
            arviz_stats.ess(iter_samples[..., ~is_constant_vec], method="mean")
        )
    iter_samples = iter_samples.reshape(-1, n_dims)

    if n_count_dims:
        fit_samples, iter_samples, log_density = _dequantised_samples_and_density(
            fit_samples=fit_samples,
            iter_samples=iter_samples,
            log_density=log_density,
            n_count_dims=n_count_dims,
            rng=rng,
        )
    proposal = scipy.stats.multivariate_normal(
        mean=fit_samples.mean(axis=0), cov=np.cov(fit_samples, rowvar=False)
    )
    # Reshaped since with one dimension, the draws have no dimension axis
    proposal_samples = proposal.rvs(size=n_proposal_samples, random_state=rng).reshape(
        n_proposal_samples, n_dims
    )

    # Log ratios of the unnormalised posterior to the proposal density
    return _bridge_estimate(
        log_ratio_posterior_samples=(
            log_density(samples=iter_samples) - proposal.logpdf(iter_samples)
        ).reshape(n_chains, -1),
        log_ratio_proposal_samples=log_density(samples=proposal_samples)
        - proposal.logpdf(proposal_samples),
        n_posterior_eff=n_posterior_eff,
        tol=tol,
        max_iterations=max_iterations,
    )


def _dequantised_samples_and_density(
    *, fit_samples, iter_samples, log_density, n_count_dims, rng
):
    # fit_samples and iter_samples (each of shape (n_samples, n_dims), with nonnegative
    # integer counts in the last n_count_dims dimensions) with the counts mapped to
    # continuous variables by a Gaussian copula dequantisation (see _dequantise) fitted to
    # fit_samples, using fresh uniforms, and the log density of the dequantised variables,
    # whose integral over them is the sum of exp(log_density) over the counts
    n_continuous_dims = fit_samples.shape[1] - n_count_dims
    reference_pmfs = _fit_reference_pmfs(
        count_samples=fit_samples[:, n_continuous_dims:].astype(np.int64)
    )

    def _dequantised(*, samples):
        counts = samples[:, n_continuous_dims:].astype(np.int64)
        return np.concatenate(
            [
                samples[:, :n_continuous_dims],
                _dequantise(
                    counts=counts,
                    uniforms=rng.uniform(size=counts.shape),
                    reference_pmfs=reference_pmfs,
                ),
            ],
            axis=-1,
        )

    def _log_dequantised_density(*, samples):
        # log_density at the counts the samples map to, plus the log conditional density
        # of the dequantised variables given those counts
        counts, log_conditional_density_samples = _quantise(
            y=samples[:, n_continuous_dims:], reference_pmfs=reference_pmfs
        )
        return (
            log_density(
                samples=np.concatenate(
                    [samples[:, :n_continuous_dims], counts], axis=-1
                )
            )
            + log_conditional_density_samples
        )

    return (
        _dequantised(samples=fit_samples),
        _dequantised(samples=iter_samples),
        _log_dequantised_density,
    )


def _fit_reference_pmfs(*, count_samples, geom_weight=1e-3):
    # Reference pmf p_ref of each dimension of count_samples (of shape (n_samples,
    # n_dims)) for the dequantisation (see _dequantise): the empirical pmf, mixed with
    # weight geom_weight with a geometric pmf on 0, 1, ... with mean one more than the
    # empirical mean, so that every count has positive mass (counts unseen in
    # count_samples would otherwise be left out of the sum). Tabulated as log p_ref, log
    # F_ref and log S_ref (the CDF and survival function, S_ref = 1 - F_ref), mixing in
    # logs so that the geometric part doesn't underflow. Each table has shape
    # (max_fitted_count + 2, n_dims) for the largest count seen max_fitted_count, with
    # count k = -1, 0, ..., max_fitted_count in row k + 1. Beyond the tables only the
    # geometric part remains, so its success probabilities and geom_weight are kept too
    n_samples = len(count_samples)
    max_fitted_count = count_samples.max()
    # As a column, so as to broadcast across dimensions
    tabulated_count_vec = np.arange(-1, max_fitted_count + 1)[:, None]
    # How often each tabulated count is seen (never, for count -1)
    frequency_table = np.pad(
        np.stack(
            [
                np.bincount(col, minlength=max_fitted_count + 1)
                for col in count_samples.T
            ],
            axis=-1,
        ),
        ((1, 0), (0, 0)),
    )
    cum_frequency_table = np.cumsum(frequency_table, axis=0)
    geom_prob_vec = 1 / (count_samples.mean(axis=0) + 2)
    geom_dist = scipy.stats.geom(geom_prob_vec, loc=-1)

    def _log_mixture_table(*, empirical_table, log_geom_table):
        with np.errstate(divide="ignore"):  # log 0 = -inf where an empirical value is 0
            return np.logaddexp(
                np.log1p(-geom_weight) + np.log(empirical_table),
                np.log(geom_weight) + log_geom_table,
            )

    return {
        "log_pmf_table": _log_mixture_table(
            empirical_table=frequency_table / n_samples,
            log_geom_table=geom_dist.logpmf(tabulated_count_vec),
        ),
        "log_cdf_table": _log_mixture_table(
            empirical_table=cum_frequency_table / n_samples,
            log_geom_table=geom_dist.logcdf(tabulated_count_vec),
        ),
        "log_sf_table": _log_mixture_table(
            empirical_table=(n_samples - cum_frequency_table) / n_samples,
            log_geom_table=geom_dist.logsf(tabulated_count_vec),
        ),
        "geom_prob_vec": geom_prob_vec,
        "geom_weight": geom_weight,
    }


def _dequantise(*, counts, uniforms, reference_pmfs):
    # Gaussian copula dequantisation of counts U of shape (n_samples, n_dims), with
    # reference pmfs p_ref and CDFs F_ref (see _fit_reference_pmfs):
    # y = Phi^-1(F_ref(U - 1) + V p_ref(U)) for uniforms V on [0, 1). Given U, Phi(y) is
    # uniform on [F_ref(U - 1), F_ref(U)), so y has conditional density phi(y) / p_ref(U)
    # there (see _quantise). So as to stay finite far into the tails, computed from
    # log Phi(y) for y < 0 and, with S_ref = 1 - F_ref, from
    # log Phi(-y) = log(S_ref(U) + (1 - V) p_ref(U)) otherwise
    log_ref_pmf_samples, _, log_ref_sf_samples = _reference_pmf_values(
        counts=counts, reference_pmfs=reference_pmfs
    )
    _, log_ref_cdf_prev_samples, _ = _reference_pmf_values(
        counts=counts - 1, reference_pmfs=reference_pmfs
    )
    with np.errstate(divide="ignore"):  # log V = -inf where V = 0
        log_cdf_samples = np.logaddexp(
            log_ref_cdf_prev_samples, np.log(uniforms) + log_ref_pmf_samples
        )
    log_sf_samples = np.logaddexp(
        log_ref_sf_samples, np.log1p(-uniforms) + log_ref_pmf_samples
    )
    return np.where(
        log_cdf_samples < log_sf_samples,
        scipy.special.ndtri_exp(log_cdf_samples),
        -scipy.special.ndtri_exp(log_sf_samples),
    )


def _quantise(*, y, reference_pmfs):
    # Counts U with F_ref(U - 1) <= Phi(y) < F_ref(U) for dequantised samples y of shape
    # (n_samples, n_dims) (see _dequantise), and the log conditional density of y given
    # U, log phi(y) - log p_ref(U), summed over dimensions. As in _dequantise, U is found
    # from log F_ref for y < 0, as the number of counts k with log F_ref(k) <= log Phi(y),
    # and from log S_ref otherwise, as the number with log S_ref(k) >= log Phi(-y). Each
    # is searched for in the tables, by whose end F_ref is past 1/2, so U can only be
    # beyond them for y >= 0
    # The tables' rows from count 0
    log_ref_cdf_table = reference_pmfs["log_cdf_table"][1:]
    log_ref_sf_table = reference_pmfs["log_sf_table"][1:]
    max_fitted_count = len(log_ref_cdf_table) - 1
    log_cdf_samples = scipy.stats.norm.logcdf(y)
    log_sf_samples = scipy.stats.norm.logsf(y)
    counts = np.stack(
        [
            np.where(
                y[:, j] < 0,
                np.searchsorted(
                    log_ref_cdf_table[:, j], log_cdf_samples[:, j], side="right"
                ),
                np.searchsorted(
                    -log_ref_sf_table[:, j], -log_sf_samples[:, j], side="right"
                ),
            )
            for j in range(y.shape[1])
        ],
        axis=-1,
    )
    # If every tabulated count k has log S_ref(k) >= log Phi(-y), U is beyond the tables,
    # where only the geometric part remains, with S_ref(k) = geom_weight (1 - p)^(k + 1)
    # for success probability p. U is then the smallest k with S_ref(k) < Phi(-y), so
    # with k + 1 > (log Phi(-y) - log(geom_weight)) / log(1 - p) (flipped since
    # log(1 - p) < 0), found in logs as Phi(-y) can underflow
    tail_counts = np.floor(
        (log_sf_samples - np.log(reference_pmfs["geom_weight"]))
        / np.log1p(-reference_pmfs["geom_prob_vec"])
    ).astype(np.int64)
    counts = np.where(counts <= max_fitted_count, counts, tail_counts)
    log_ref_pmf_samples, _, _ = _reference_pmf_values(
        counts=counts, reference_pmfs=reference_pmfs
    )
    return counts, (scipy.stats.norm.logpdf(y) - log_ref_pmf_samples).sum(axis=-1)


def _reference_pmf_values(*, counts, reference_pmfs):
    # log p_ref(k), log F_ref(k) and log S_ref(k) of the reference pmfs (see
    # _fit_reference_pmfs) at counts k >= -1 of shape (n_samples, n_dims), looked up in
    # the tables, beyond which only the geometric part remains
    geom_weight = reference_pmfs["geom_weight"]
    geom_dist = scipy.stats.geom(reference_pmfs["geom_prob_vec"], loc=-1)
    max_fitted_count = len(reference_pmfs["log_pmf_table"]) - 2
    is_in_table = counts <= max_fitted_count
    table_idx = (np.minimum(counts, max_fitted_count) + 1, np.arange(counts.shape[1]))
    log_tail_sf = np.log(geom_weight) + geom_dist.logsf(counts)
    log_pmf = np.where(
        is_in_table,
        reference_pmfs["log_pmf_table"][table_idx],
        np.log(geom_weight) + geom_dist.logpmf(counts),
    )
    log_cdf = np.where(
        is_in_table,
        reference_pmfs["log_cdf_table"][table_idx],
        np.log1p(-np.exp(log_tail_sf)),
    )
    log_sf = np.where(
        is_in_table, reference_pmfs["log_sf_table"][table_idx], log_tail_sf
    )
    return log_pmf, log_cdf, log_sf


def _bridge_estimate(
    *,
    log_ratio_posterior_samples,
    log_ratio_proposal_samples,
    n_posterior_eff,
    tol,
    max_iterations,
):
    # Log evidence by the iterative scheme for the optimal bridge function (Meng & Wong
    # 1996, as in Gronau et al. 2017), from the log ratios of the unnormalised posterior to
    # the proposal density at posterior samples, of shape (n_chains, n_draws), and at
    # independent proposal samples, with weights s1 = n_posterior_eff / (n_posterior_eff +
    # n_proposal_samples) and s2 = n_proposal_samples / (n_posterior_eff +
    # n_proposal_samples) for the posterior samples' effective sample size n_posterior_eff
    # (for independent samples, their number, which gives the optimal weights). The
    # standard error of the log evidence is the root relative mean squared error of the
    # evidence (Fruhwirth-Schnatter 2004), using the effective sample size of the
    # denominator terms to account for autocorrelation
    n_proposal_samples = len(log_ratio_proposal_samples)
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
    denominator_terms = np.exp(log_denominator_terms)
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
