# Note that AI tools were used to generate tests
import numpy as np
import scipy.stats

from endoutbreakhet.bridge_sampling import (
    _dequantise,
    _fit_reference_pmfs,
    _quantise,
    _reference_pmf_values,
    bridge_sampling,
)


def test_count_dequantisation():
    # Reference pmfs fitted to a count that's always zero and one with spread. Counts up
    # to well beyond those fitted (into the geometric tail, far enough that its survival
    # function underflows) map to finite values that map back to the same counts,
    # including at the ends of each count's interval
    rng = np.random.default_rng(0)
    count_samples = np.stack(
        [np.zeros(500, dtype=np.int64), rng.poisson(3.0, size=500)], axis=-1
    )
    reference_pmfs = _fit_reference_pmfs(count_samples=count_samples)
    # Tabulated from count -1 up to the largest count seen
    assert len(reference_pmfs["log_pmf_table"]) == count_samples.max() + 2
    counts = np.concatenate([np.tile(np.arange(60), (2, 1)).T, [[1100, 1100]]])
    for uniforms in [
        rng.uniform(size=counts.shape),
        np.full(counts.shape, 1e-9),
        np.full(counts.shape, 1 - 1e-9),
    ]:
        y = _dequantise(counts=counts, uniforms=uniforms, reference_pmfs=reference_pmfs)
        assert np.all(np.isfinite(y))
        quantised_counts, log_conditional_density_samples = _quantise(
            y=y, reference_pmfs=reference_pmfs
        )
        np.testing.assert_array_equal(quantised_counts, counts)
        log_pmf, _, _ = _reference_pmf_values(
            counts=counts, reference_pmfs=reference_pmfs
        )
        np.testing.assert_allclose(
            log_conditional_density_samples,
            (scipy.stats.norm.logpdf(y) - log_pmf).sum(axis=-1),
        )

    # Each count's interval of Phi(y) ends where the next count's starts, and has width
    # p_ref(U) (compared in logs, as the tail's widths underflow), so the density in y
    # integrates to the sum over the counts. This holds across the end of the tables
    log_sf_lower, log_sf_upper = (
        scipy.stats.norm.logsf(
            _dequantise(
                counts=interval_counts,
                uniforms=np.zeros(counts.shape),
                reference_pmfs=reference_pmfs,
            )
        )
        for interval_counts in (counts, counts + 1)
    )
    log_pmf, _, _ = _reference_pmf_values(counts=counts, reference_pmfs=reference_pmfs)
    np.testing.assert_allclose(
        log_sf_lower + np.log1p(-np.exp(log_sf_upper - log_sf_lower)),
        log_pmf,
        rtol=0,
        atol=1e-8,
    )
    # The pmf sums to 1, consistently with the CDF and survival function (which are 0
    # and 1 at count -1), with mass close to the empirical pmf
    log_pmf_table = reference_pmfs["log_pmf_table"]
    log_sf_table = reference_pmfs["log_sf_table"]
    np.testing.assert_allclose(
        np.exp(log_pmf_table).sum(axis=0) + np.exp(log_sf_table[-1]), 1.0
    )
    np.testing.assert_allclose(
        np.exp(reference_pmfs["log_cdf_table"]) + np.exp(log_sf_table), 1.0
    )
    # At count 0 of the count that's always zero
    np.testing.assert_allclose(np.exp(log_pmf_table[1, 0]), 1.0, atol=1e-3)


def test_bridge_sampling_gaussian():
    rng = np.random.default_rng(0)
    n_dims = 4
    mean = rng.normal(size=n_dims)
    sqrt_cov = rng.normal(size=(n_dims, n_dims))
    cov = sqrt_cov @ sqrt_cov.T + n_dims * np.eye(n_dims)
    log_normaliser = 3.7

    def _log_density(*, samples):
        return log_normaliser + scipy.stats.multivariate_normal.logpdf(
            samples, mean=mean, cov=cov
        )

    result = bridge_sampling(
        posterior_samples=rng.multivariate_normal(mean, cov, size=(4, 1000)),
        log_density=_log_density,
        n_proposal_samples=2000,
        rng=rng,
    )
    assert abs(result["log_evidence"] - log_normaliser) < 4 * result["log_evidence_se"]


def test_bridge_sampling_counts():
    # Log normaliser 0 of a joint density of normal theta and counts U_t | theta ~
    # Poisson(e^theta m_t), some nearly always zero, from exact draws. The counts are
    # summed over, so the log density only sees nonnegative integers
    rng = np.random.default_rng(0)
    mean_vec = np.geomspace(0.003, 8, 50)
    theta = rng.normal(scale=0.3, size=(4, 1000))
    counts = rng.poisson(np.exp(theta)[..., None] * mean_vec)

    def _log_density(*, samples):
        assert np.all(samples[:, 1:] >= 0)
        assert np.all(samples[:, 1:] == np.round(samples[:, 1:]))
        return scipy.stats.norm.logpdf(
            samples[:, 0], scale=0.3
        ) + scipy.stats.poisson.logpmf(
            samples[:, 1:], np.exp(samples[:, :1]) * mean_vec
        ).sum(axis=-1)

    result = bridge_sampling(
        posterior_samples=np.concatenate([theta[..., None], counts], axis=-1),
        log_density=_log_density,
        n_proposal_samples=2000,
        rng=rng,
        n_count_dims=len(mean_vec),
    )
    assert abs(result["log_evidence"]) < 4 * result["log_evidence_se"]
