import numpy as np
import scipy.special
import scipy.stats

from endoutbreakhet.utils import (
    delay_matrix,
    get_likelihood_idx,
    parse_lognormal_prior,
)


def compute_log_evidence(
    *,
    model_name,
    incidence_vec,
    switch_day,
    delay_distributions,
    priors,
    n_samples,
    rng,
):
    routines = {
        "poisson": _poisson_log_likelihood_samples,
        "sse": _sse_log_likelihood_samples,
        "ssi": _ssi_log_likelihood_samples,
        "dlo": _dlo_log_likelihood_samples,
        "poisson_so": _poisson_so_log_likelihood_samples,
        "sse_so": _sse_so_log_likelihood_samples,
        "ssi_so": _ssi_so_log_likelihood_samples,
    }
    if model_name not in routines:
        raise ValueError(f"Unknown model name: {model_name}")
    log_likelihood_samples = routines[model_name](
        incidence_vec=incidence_vec,
        switch_day=switch_day,
        delay_distributions=delay_distributions,
        priors=priors,
        n_samples=n_samples,
        rng=rng,
    )

    log_sum_likelihood = scipy.special.logsumexp(log_likelihood_samples)
    # Effective sample size of the likelihood weights, (sum w)^2 / sum w^2
    ess = np.exp(
        2 * log_sum_likelihood - scipy.special.logsumexp(2 * log_likelihood_samples)
    )
    return {
        "log_evidence": log_sum_likelihood - np.log(n_samples),
        # Delta-method standard error of the log evidence estimate
        "log_evidence_se": np.sqrt(1 / ess - 1 / n_samples),
        "ess": ess,
    }


def _poisson_log_likelihood_samples(
    *, incidence_vec, switch_day, delay_distributions, priors, n_samples, rng
):
    n_days = len(incidence_vec)
    serial_interval_matrix = delay_matrix(
        dist_vec=delay_distributions["serial_interval"], n_days=n_days, first_lag=1
    )
    total_inf_vec = serial_interval_matrix @ incidence_vec
    likelihood_idx = get_likelihood_idx(
        incidence_vec=incidence_vec, force_vec=total_inf_vec
    )

    rep_no_vec_samples = _rep_no_vec_samples(
        priors=priors,
        switch_day=switch_day,
        n_days=n_days,
        n_samples=n_samples,
        rng=rng,
    )

    expected_incidence_local_samples = rep_no_vec_samples * total_inf_vec

    return scipy.stats.poisson.logpmf(
        k=incidence_vec[likelihood_idx],
        mu=expected_incidence_local_samples[:, likelihood_idx],
    ).sum(axis=-1)


def _sse_log_likelihood_samples(
    *, incidence_vec, switch_day, delay_distributions, priors, n_samples, rng
):
    n_days = len(incidence_vec)
    serial_interval_matrix = delay_matrix(
        dist_vec=delay_distributions["serial_interval"], n_days=n_days, first_lag=1
    )
    total_inf_vec = serial_interval_matrix @ incidence_vec
    likelihood_idx = get_likelihood_idx(
        incidence_vec=incidence_vec, force_vec=total_inf_vec
    )

    rep_no_vec_samples = _rep_no_vec_samples(
        priors=priors,
        switch_day=switch_day,
        n_days=n_days,
        n_samples=n_samples,
        rng=rng,
    )
    dispersion_samples = _prior_samples(
        name="dispersion", priors=priors, n_samples=n_samples, rng=rng
    )

    expected_incidence_local_samples = rep_no_vec_samples * total_inf_vec

    return _nbinom_logpmf(
        k=incidence_vec[likelihood_idx],
        mean=expected_incidence_local_samples[:, likelihood_idx],
        dispersion=dispersion_samples[:, None] * total_inf_vec[likelihood_idx],
    ).sum(axis=-1)


def _ssi_log_likelihood_samples(
    *, incidence_vec, switch_day, delay_distributions, priors, n_samples, rng
):
    n_days = len(incidence_vec)
    serial_interval_matrix = delay_matrix(
        dist_vec=delay_distributions["serial_interval"], n_days=n_days, first_lag=1
    )
    likelihood_idx = get_likelihood_idx(
        incidence_vec=incidence_vec, force_vec=serial_interval_matrix @ incidence_vec
    )
    nonzero_incidence_idx = incidence_vec > 0

    rep_no_vec_samples = _rep_no_vec_samples(
        priors=priors,
        switch_day=switch_day,
        n_days=n_days,
        n_samples=n_samples,
        rng=rng,
    )
    dispersion_samples = _prior_samples(
        name="dispersion", priors=priors, n_samples=n_samples, rng=rng
    )

    cohort_inf_case_day_samples = _gamma_samples(
        mean=incidence_vec[nonzero_incidence_idx],
        shape=dispersion_samples[:, None] * incidence_vec[nonzero_incidence_idx],
        rng=rng,
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
    *, incidence_vec, switch_day, delay_distributions, priors, n_samples, rng
):
    n_days = len(incidence_vec)
    serial_interval_matrix = delay_matrix(
        dist_vec=delay_distributions["serial_interval"], n_days=n_days, first_lag=1
    )
    total_inf_vec = serial_interval_matrix @ incidence_vec
    likelihood_idx = get_likelihood_idx(
        incidence_vec=incidence_vec, force_vec=total_inf_vec
    )

    rep_no_vec_samples = _rep_no_vec_samples(
        priors=priors,
        switch_day=switch_day,
        n_days=n_days,
        n_samples=n_samples,
        rng=rng,
    )
    dispersion_samples = _prior_samples(
        name="dispersion", priors=priors, n_samples=n_samples, rng=rng
    )

    expected_incidence_local_samples = rep_no_vec_samples * total_inf_vec

    return _nbinom_logpmf(
        k=incidence_vec[likelihood_idx],
        mean=expected_incidence_local_samples[:, likelihood_idx],
        dispersion=dispersion_samples[:, None],
    ).sum(axis=-1)


def _poisson_so_log_likelihood_samples(
    *, incidence_vec, switch_day, delay_distributions, priors, n_samples, rng
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

    rep_no_vec_samples = _rep_no_vec_samples(
        priors=priors,
        switch_day=switch_day,
        n_days=n_days,
        n_samples=n_samples,
        rng=rng,
    )

    expected_incidence_local_samples = (
        rep_no_vec_samples * total_inf_vec
    ) @ inc_period_matrix.T

    return scipy.stats.poisson.logpmf(
        k=incidence_vec[likelihood_idx],
        mu=expected_incidence_local_samples[:, likelihood_idx],
    ).sum(axis=-1)


def _sse_so_log_likelihood_samples(
    *, incidence_vec, switch_day, delay_distributions, priors, n_samples, rng
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

    rep_no_vec_samples = _rep_no_vec_samples(
        priors=priors,
        switch_day=switch_day,
        n_days=n_days,
        n_samples=n_samples,
        rng=rng,
    )
    dispersion_samples = _prior_samples(
        name="dispersion", priors=priors, n_samples=n_samples, rng=rng
    )

    total_inf_nonzero_day_samples = _gamma_samples(
        mean=expected_total_inf_vec[nonzero_total_inf_idx],
        shape=dispersion_samples[:, None]
        * expected_total_inf_vec[nonzero_total_inf_idx],
        rng=rng,
    )
    expected_incidence_local_samples = (
        total_inf_nonzero_day_samples * rep_no_vec_samples[:, nonzero_total_inf_idx]
    ) @ inc_period_matrix[:, nonzero_total_inf_idx].T

    return scipy.stats.poisson.logpmf(
        k=incidence_vec[likelihood_idx],
        mu=expected_incidence_local_samples[:, likelihood_idx],
    ).sum(axis=-1)


def _ssi_so_log_likelihood_samples(
    *, incidence_vec, switch_day, delay_distributions, priors, n_samples, rng
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

    rep_no_vec_samples = _rep_no_vec_samples(
        priors=priors,
        switch_day=switch_day,
        n_days=n_days,
        n_samples=n_samples,
        rng=rng,
    )
    dispersion_samples = _prior_samples(
        name="dispersion", priors=priors, n_samples=n_samples, rng=rng
    )

    cohort_inf_case_day_samples = _gamma_samples(
        mean=incidence_vec[nonzero_incidence_idx],
        shape=dispersion_samples[:, None] * incidence_vec[nonzero_incidence_idx],
        rng=rng,
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


def _rep_no_vec_samples(*, priors, switch_day, n_days, n_samples, rng):
    rep_no_before_samples = _prior_samples(
        name="rep_no_before", priors=priors, n_samples=n_samples, rng=rng
    )
    rep_no_after_samples = _prior_samples(
        name="rep_no_after", priors=priors, n_samples=n_samples, rng=rng
    )

    return np.where(
        np.arange(n_days) < switch_day,
        rep_no_before_samples[:, None],
        rep_no_after_samples[:, None],
    )


def _prior_samples(*, name, priors, n_samples, rng):
    prior_params = parse_lognormal_prior(name=name, priors=priors)
    return rng.lognormal(
        mean=prior_params["mu"], sigma=prior_params["sigma"], size=n_samples
    )


def _nbinom_logpmf(*, k, mean, dispersion):
    # Same parameterisation as pm.NegativeBinomial(mu=mean, alpha=dispersion)
    return scipy.stats.nbinom.logpmf(
        k, n=dispersion, p=dispersion / (dispersion + mean)
    )


def _gamma_samples(*, mean, shape, rng):
    # Gamma samples with given mean and shape (numpy uses shape and scale)
    return rng.gamma(shape=shape, scale=mean / shape)
