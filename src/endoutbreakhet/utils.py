import functools
from typing import Protocol

import numpy as np
import scipy.integrate
import scipy.linalg
import scipy.stats
from numpy.typing import NDArray


class _ContinuousDist(Protocol):
    def pdf(self, x: float) -> float: ...


def discretise_cori(
    *, dist_cont: _ContinuousDist, max_val: int, allow_zero: bool = False
) -> NDArray[np.float64]:
    """
    Function for discretising a continuous distribution using the method
    described in https://doi.org/10.1093/aje/kwt133 (web appendix 11).

    Parameters
    ----------
    dist_cont : scipy.stats frozen continuous distribution
        Continuous distribution to discretise (only its ``pdf`` method is used).
    max_val : int
        Maximum value (days) of the discretised distribution. Any residual probability
        mass is assigned to this value.
    allow_zero : bool
        Whether the discretised distribution may place mass on zero. If False (the
        default), mass at zero is folded into the value 1 and the returned vector starts
        at 1.

    Returns
    -------
    NDArray[np.float64]
        Probability mass vector of the discretised distribution.
    """

    def _integrand_func(y: float, *, x: float) -> float:
        # To get probability mass function at time x, need to integrate this expression
        # with respect to y between y=x-1 and and y=x+1
        return (1 - abs(x - y)) * dist_cont.pdf(y)

    if max_val < 0:
        raise ValueError("max_val must be non-negative")
    if not allow_zero and max_val < 1:
        raise ValueError("max_val must be at least 1 when allow_zero is False")

    # Calculate probability mass function at each x value from 0 to max_val
    p_vec = np.zeros(max_val + 1)
    for x in range(max_val + 1):
        integrand = functools.partial(_integrand_func, x=x)
        p_vec[x] = scipy.integrate.quad(
            integrand,
            x - 1 if x > 0 else 1e-12,
            x + 1,
        )[0]
    if not allow_zero:
        # Assign mass from 0 to 1
        p_vec[1] = p_vec[1] + p_vec[0]
        p_vec = p_vec[1:]
    # Assign residual mass to x_max
    p_vec[-1] = p_vec[-1] + 1 - np.sum(p_vec)
    return p_vec


def gamma_dist_from_mean_sd(*, mean: float, sd: float):
    return scipy.stats.gamma(a=(mean / sd) ** 2, scale=sd**2 / mean)


def lognormal_params_from_median_quantile_025(*, median, quantile_025):
    mu = np.log(median)
    sigma = (mu - np.log(quantile_025)) / scipy.stats.norm.ppf(0.975)
    return {"mu": mu, "sigma": sigma}


def parse_lognormal_prior(*, name, priors):
    prior = priors[name]
    if prior.get("dist", None) != "lognormal":
        raise ValueError(
            f"Unsupported prior distribution for {name}: {prior.get('dist', None)}"
        )
    return lognormal_params_from_median_quantile_025(
        median=prior["median"], quantile_025=prior["quantile_025"]
    )


def delay_matrix(*, dist_vec, n_days, first_lag):
    # Lower-triangular matrix with [t, s] entry dist_vec[t - s - first_lag], so that
    # (matrix @ x)[t] is the convolution of x with a delay whose pmf at lag first_lag + i
    # is dist_vec[i]
    dist_vec = dist_vec[: n_days - first_lag]
    first_col = np.zeros(n_days)
    first_col[first_lag : first_lag + len(dist_vec)] = dist_vec
    return scipy.linalg.toeplitz(first_col, np.zeros(n_days))


def get_likelihood_idx(*, incidence_vec, force_vec):
    # Days with nonzero force of infection/disease. Day-0 cases are imported; later cases
    # are local, so are only possible on these days
    likelihood_idx = force_vec > 0
    if np.any(incidence_vec[1:][~likelihood_idx[1:]]):
        raise ValueError(
            "Local incidence cannot be greater than zero when force of "
            "infection/disease is 0."
        )
    return likelihood_idx


def get_rep_no_vec_samples(
    *, rep_no_before_samples, rep_no_after_samples, switch_day, n_days
):
    return np.where(
        np.arange(n_days) < switch_day,
        rep_no_before_samples[:, None],
        rep_no_after_samples[:, None],
    )
