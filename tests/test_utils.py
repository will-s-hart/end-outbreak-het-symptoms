# Note that AI tools were used to generate tests
import numpy as np
import pytest
import scipy.stats

from endoutbreakhet.utils import (
    delay_matrix,
    discretise_cori,
    gamma_dist_from_mean_sd,
    get_likelihood_idx,
    get_rep_no_vec_samples,
    parse_lognormal_prior,
)


@pytest.mark.parametrize("first_lag", [0, 1])
@pytest.mark.parametrize("n_days", [7, 3])  # n_days=3 truncates the delay vector
def test_delay_matrix(*, first_lag, n_days):
    dist_vec = np.array([0.1, 0.2, 0.3, 0.4])
    matrix = delay_matrix(dist_vec=dist_vec, n_days=n_days, first_lag=first_lag)
    assert matrix.shape == (n_days, n_days)
    for t in range(n_days):
        for s in range(n_days):
            lag = t - s - first_lag
            expected = dist_vec[lag] if 0 <= lag < len(dist_vec) else 0.0
            assert matrix[t, s] == expected


def test_get_likelihood_idx():
    # Day-0 case with zero force is allowed (imported)
    reported_incidence_vec = np.array([3, 1, 0, 2, 0])
    force_vec = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
    np.testing.assert_array_equal(
        get_likelihood_idx(
            reported_incidence_vec=reported_incidence_vec, force_vec=force_vec
        ),
        [False, True, False, True, False],
    )


def test_get_likelihood_idx_raises_for_local_case_without_force():
    with pytest.raises(ValueError, match="Local incidence"):
        get_likelihood_idx(
            reported_incidence_vec=np.array([3, 1, 1]),
            force_vec=np.array([0.0, 1.0, 0.0]),
        )


def test_get_rep_no_vec_samples():
    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=np.array([1.5, 2.0]),
        rep_no_after_samples=np.array([0.5, 0.3]),
        switch_day=2,
        n_days=4,
    )
    np.testing.assert_array_equal(
        rep_no_vec_samples, [[1.5, 1.5, 0.5, 0.5], [2.0, 2.0, 0.3, 0.3]]
    )


def test_parse_lognormal_prior():
    params = parse_lognormal_prior(
        name="x",
        priors={"x": {"dist": "lognormal", "median": 0.18, "quantile_025": 0.018}},
    )
    dist = scipy.stats.lognorm(s=params["sigma"], scale=np.exp(params["mu"]))
    assert dist.median() == pytest.approx(0.18)
    assert dist.ppf(0.025) == pytest.approx(0.018)


@pytest.mark.parametrize(
    "prior",
    [
        {"dist": "gamma", "median": 1.0, "quantile_025": 0.2},
        {"median": 1.0, "quantile_025": 0.2},
    ],
)
def test_parse_lognormal_prior_raises_for_unsupported_prior(*, prior):
    with pytest.raises(ValueError, match="Unsupported prior"):
        parse_lognormal_prior(name="x", priors={"x": prior})


def test_gamma_dist_from_mean_sd():
    dist = gamma_dist_from_mean_sd(mean=5.0, sd=3.0)
    assert dist.mean() == pytest.approx(5.0)
    assert dist.std() == pytest.approx(3.0)


@pytest.mark.parametrize("allow_zero", [False, True])
def test_discretise_cori(*, allow_zero):
    p_vec = discretise_cori(
        dist_cont=gamma_dist_from_mean_sd(mean=5.0, sd=3.0),
        max_val=20,
        allow_zero=allow_zero,
    )
    assert len(p_vec) == (21 if allow_zero else 20)
    assert np.all(p_vec >= 0)
    assert p_vec.sum() == pytest.approx(1.0)
