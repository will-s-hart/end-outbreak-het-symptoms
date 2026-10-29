import functools
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
import scipy.integrate
import scipy.linalg
import scipy.stats
from numpy.typing import NDArray

# First lag of each delay, which the models assume. A serial interval, or TOST plus
# incubation period, of at least a day stops a day's cases causing themselves
DELAY_FIRST_LAGS = {"serial_interval": 1, "inc_period": 1, "tost": 0}


# Identity equality, since arrays don't compare as booleans
@dataclass(frozen=True, kw_only=True, eq=False)
class DelayDistribution:
    """Probability mass function of a delay, with prob[i] the probability of lag
    first_lag + i."""

    prob: NDArray[np.float64]
    first_lag: int

    def __post_init__(self):
        if self.first_lag < 0:
            raise ValueError(f"first_lag must be non-negative: {self.first_lag}")
        if (
            self.prob.ndim != 1
            or np.any(self.prob < 0)
            or not np.isclose(np.sum(self.prob), 1)
        ):
            raise ValueError("prob must be a vector of probabilities summing to 1")

    @property
    def lags(self):
        return np.arange(self.first_lag, self.max_lag + 1)

    @property
    def max_lag(self):
        return self.first_lag + len(self.prob) - 1

    @classmethod
    def from_df(cls, *, df):
        # From a table of consecutive lags and their probabilities, as made by to_df
        lags = df["lag"].to_numpy()
        if not np.array_equal(lags, lags[0] + np.arange(len(lags))):
            raise ValueError("lags must be consecutive")
        return cls(prob=df["prob"].to_numpy(), first_lag=int(lags[0]))

    def to_df(self):
        return pd.DataFrame({"lag": self.lags, "prob": self.prob})


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


def delay_matrix(*, delay, n_days):
    # Lower-triangular matrix with [t, s] entry delay.prob[t - s - delay.first_lag],
    # so that (matrix @ x)[t] is the convolution of x with the delay
    first_col = np.zeros(n_days)
    if delay.first_lag < n_days:
        prob = delay.prob[: n_days - delay.first_lag]
        first_col[delay.first_lag : delay.first_lag + len(prob)] = prob
    return scipy.linalg.toeplitz(first_col, np.zeros(n_days))


def validate_delay_distributions(*, model_name, delay_distributions):
    # Checks the delays the model uses start at the lags it assumes
    delay_names = (
        ("tost", "inc_period") if model_name.endswith("_so") else ("serial_interval",)
    )
    for delay_name in delay_names:
        first_lag = delay_distributions[delay_name].first_lag
        if first_lag != DELAY_FIRST_LAGS[delay_name]:
            raise ValueError(
                f"{delay_name} must start at lag {DELAY_FIRST_LAGS[delay_name]}, got "
                f"{first_lag}"
            )


def get_likelihood_idx(*, reported_incidence_vec, force_vec):
    # Days with nonzero force of infection/disease. Day-0 cases are imported; later cases
    # are local, so are only possible on these days
    likelihood_idx = force_vec > 0
    if np.any(reported_incidence_vec[1:][~likelihood_idx[1:]]):
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
