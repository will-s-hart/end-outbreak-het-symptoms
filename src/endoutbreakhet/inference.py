import numpy as np
import pymc as pm
import scipy.linalg

from endoutbreakhet.utils import lognormal_params_from_median_percentile_2_5

DEFAULT_PRIORS = {
    "rep_no_before": (
        pm.LogNormal,
        lognormal_params_from_median_percentile_2_5(1, 0.2),
    ),
    "rep_no_after": (
        pm.LogNormal,
        lognormal_params_from_median_percentile_2_5(1, 0.2),
    ),
    "dispersion": (
        pm.LogNormal,
        lognormal_params_from_median_percentile_2_5(0.2, 0.1),
    ),
}


def fit_model(
    model_name,
    incidence_vec,
    switch_day,
    delay_distributions=None,
    priors=None,
    step=None,
    thin=1,
    **kwargs_sample,
):
    pm_model = _build_pymc_model(
        model_name=model_name,
        incidence_vec=incidence_vec,
        switch_day=switch_day,
        delay_distributions=delay_distributions,
        priors=DEFAULT_PRIORS | (priors or {}),
    )

    with pm_model:
        if step is not None:
            kwargs_sample["step"] = step()

        trace = pm.sample(
            idata_kwargs={"log_likelihood": True, "log_prior": True}, **kwargs_sample
        )
    if thin > 1:
        trace = trace.isel(draw=slice(0, None, thin))
        trace = trace.assign_coords(draw=np.arange(len(trace.posterior.draw)))
    return trace


def _build_pymc_model(
    model_name, incidence_vec, switch_day, delay_distributions, priors
):
    builders = {
        "poisson": _build_poisson_model,
        "sse": _build_sse_model,
        "ssi": _build_ssi_model,
        "dlo": _build_dlo_model,
        "poisson_so": _build_poisson_so_model,
        "sse_so": _build_sse_so_model,
        "ssi_so": _build_ssi_so_model,
    }
    if model_name not in builders:
        raise ValueError(f"Unknown model name: {model_name}")
    return builders[model_name](
        incidence_vec=incidence_vec,
        switch_day=switch_day,
        delay_distributions=delay_distributions,
        priors=priors,
    )


def _build_poisson_model(incidence_vec, switch_day, delay_distributions, priors):
    n_days = len(incidence_vec)
    gen_time_matrix = _delay_matrix(
        delay_distributions["gen_time"], n_days, first_lag=1
    )
    foi_vec = gen_time_matrix @ incidence_vec
    likelihood_idx = _get_likelihood_idx(incidence_vec, foi_vec)

    with pm.Model() as model:
        rep_no_vec = _build_rep_no_vec(
            priors=priors, switch_day=switch_day, n_days=n_days
        )

        expected_incidence_local = rep_no_vec * foi_vec

        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[likelihood_idx],
            observed=incidence_vec[likelihood_idx],
        )
    return model


def _build_sse_model(incidence_vec, switch_day, delay_distributions, priors):
    n_days = len(incidence_vec)
    gen_time_matrix = _delay_matrix(
        delay_distributions["gen_time"], n_days, first_lag=1
    )
    foi_vec = gen_time_matrix @ incidence_vec
    likelihood_idx = _get_likelihood_idx(incidence_vec, foi_vec)

    with pm.Model() as model:
        rep_no_vec = _build_rep_no_vec(
            priors=priors, switch_day=switch_day, n_days=n_days
        )
        dispersion = _prior_rv("dispersion", priors)

        expected_incidence_local = rep_no_vec * foi_vec

        pm.NegativeBinomial(
            "likelihood",
            alpha=dispersion * foi_vec[likelihood_idx],
            mu=expected_incidence_local[likelihood_idx],
            observed=incidence_vec[likelihood_idx],
        )
    return model


def _build_ssi_model(incidence_vec, switch_day, delay_distributions, priors):
    n_days = len(incidence_vec)
    gen_time_matrix = _delay_matrix(
        delay_distributions["gen_time"], n_days, first_lag=1
    )
    likelihood_idx = _get_likelihood_idx(incidence_vec, gen_time_matrix @ incidence_vec)
    nonzero_incidence_idx = incidence_vec > 0

    with pm.Model() as model:
        rep_no_vec = _build_rep_no_vec(
            priors=priors, switch_day=switch_day, n_days=n_days
        )
        dispersion = _prior_rv("dispersion", priors)

        scaled_incidence_case_days = _gamma_via_uniform(
            "scaled_incidence_case_days",
            mean_vec=incidence_vec[nonzero_incidence_idx],
            dispersion=dispersion,
        )
        foi_vec = pm.math.dot(
            gen_time_matrix[:, nonzero_incidence_idx], scaled_incidence_case_days
        )
        expected_incidence_local = rep_no_vec * foi_vec

        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[likelihood_idx],
            observed=incidence_vec[likelihood_idx],
        )
    return model


def _build_dlo_model(incidence_vec, switch_day, delay_distributions, priors):
    n_days = len(incidence_vec)
    gen_time_matrix = _delay_matrix(
        delay_distributions["gen_time"], n_days, first_lag=1
    )
    foi_vec = gen_time_matrix @ incidence_vec
    likelihood_idx = _get_likelihood_idx(incidence_vec, foi_vec)

    with pm.Model() as model:
        rep_no_vec = _build_rep_no_vec(
            priors=priors, switch_day=switch_day, n_days=n_days
        )
        dispersion = _prior_rv("dispersion", priors)

        expected_incidence_local = rep_no_vec * foi_vec

        pm.NegativeBinomial(
            "likelihood",
            alpha=dispersion,
            mu=expected_incidence_local[likelihood_idx],
            observed=incidence_vec[likelihood_idx],
        )
    return model


def _build_poisson_so_model(incidence_vec, switch_day, delay_distributions, priors):
    n_days = len(incidence_vec)
    tost_matrix = _delay_matrix(delay_distributions["tost"], n_days, first_lag=0)
    inc_period_matrix = _delay_matrix(
        delay_distributions["inc_period"], n_days, first_lag=1
    )
    foi_vec = tost_matrix @ incidence_vec
    likelihood_idx = _get_likelihood_idx(incidence_vec, inc_period_matrix @ foi_vec)

    with pm.Model() as model:
        rep_no_vec = _build_rep_no_vec(
            priors=priors, switch_day=switch_day, n_days=n_days
        )

        expected_incidence_local = pm.math.dot(inc_period_matrix, rep_no_vec * foi_vec)

        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[likelihood_idx],
            observed=incidence_vec[likelihood_idx],
        )
    return model


def _build_sse_so_model(incidence_vec, switch_day, delay_distributions, priors):
    n_days = len(incidence_vec)
    tost_matrix = _delay_matrix(delay_distributions["tost"], n_days, first_lag=0)
    inc_period_matrix = _delay_matrix(
        delay_distributions["inc_period"], n_days, first_lag=1
    )
    foi_vec = tost_matrix @ incidence_vec
    nonzero_foi_idx = foi_vec > 0
    likelihood_idx = _get_likelihood_idx(incidence_vec, inc_period_matrix @ foi_vec)

    with pm.Model() as model:
        rep_no_vec = _build_rep_no_vec(
            priors=priors, switch_day=switch_day, n_days=n_days
        )
        dispersion = _prior_rv("dispersion", priors)

        foi_scaled_nonzero_days = _gamma_via_uniform(
            "foi_scaled_nonzero_days",
            mean_vec=foi_vec[nonzero_foi_idx],
            dispersion=dispersion,
        )
        expected_incidence_local = pm.math.dot(
            inc_period_matrix[:, nonzero_foi_idx],
            foi_scaled_nonzero_days * rep_no_vec[nonzero_foi_idx],
        )

        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[likelihood_idx],
            observed=incidence_vec[likelihood_idx],
        )
    return model


def _build_ssi_so_model(incidence_vec, switch_day, delay_distributions, priors):
    n_days = len(incidence_vec)
    tost_matrix = _delay_matrix(delay_distributions["tost"], n_days, first_lag=0)
    inc_period_matrix = _delay_matrix(
        delay_distributions["inc_period"], n_days, first_lag=1
    )
    likelihood_idx = _get_likelihood_idx(
        incidence_vec, inc_period_matrix @ (tost_matrix @ incidence_vec)
    )
    nonzero_incidence_idx = incidence_vec > 0

    with pm.Model() as model:
        rep_no_vec = _build_rep_no_vec(
            priors=priors, switch_day=switch_day, n_days=n_days
        )
        dispersion = _prior_rv("dispersion", priors)

        scaled_incidence_case_days = _gamma_via_uniform(
            "scaled_incidence_case_days",
            mean_vec=incidence_vec[nonzero_incidence_idx],
            dispersion=dispersion,
        )
        expected_infection_incidence_local = rep_no_vec * pm.math.dot(
            tost_matrix[:, nonzero_incidence_idx],
            scaled_incidence_case_days,
        )
        expected_incidence_local = pm.math.dot(
            inc_period_matrix, expected_infection_incidence_local
        )

        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[likelihood_idx],
            observed=incidence_vec[likelihood_idx],
        )
    return model


def _build_rep_no_vec(priors, switch_day, n_days):
    rep_no_before = _prior_rv("rep_no_before", priors)
    rep_no_after = _prior_rv("rep_no_after", priors)

    return pm.math.concatenate(
        [
            pm.math.full(switch_day, rep_no_before),
            pm.math.full(n_days - switch_day, rep_no_after),
        ]
    )


def _prior_rv(name, priors):
    prior_func, prior_kwargs = priors[name]
    return prior_func(name, **prior_kwargs)


def _delay_matrix(dist_vec, n_days, first_lag):
    # Lower-triangular matrix with [t, s] entry dist_vec[t - s - first_lag], so that
    # (matrix @ x)[t] is the convolution of x with a delay whose pmf at lag first_lag + i
    # is dist_vec[i]
    dist_vec = dist_vec[: n_days - first_lag]
    first_col = np.zeros(n_days)
    first_col[first_lag : first_lag + len(dist_vec)] = dist_vec
    return scipy.linalg.toeplitz(first_col, np.zeros(n_days))


def _get_likelihood_idx(incidence_vec, force_vec):
    # Days with nonzero force of infection/disease. Day-0 cases are imported; later cases
    # are local, so are only possible on these days
    likelihood_idx = force_vec > 0
    if np.any(incidence_vec[1:][~likelihood_idx[1:]]):
        raise ValueError(
            "Local incidence cannot be greater than zero when force of "
            "infection/disease is 0."
        )
    return likelihood_idx


def _gamma_via_uniform(name, mean_vec, dispersion):
    # Gamma(dispersion * mean_vec, dispersion) variable, so with mean mean_vec. Inverse
    # gamma transformation prevents NUTS sampling but improves efficiency
    u = pm.Uniform("u", 0, 1, shape=len(mean_vec))
    return pm.Deterministic(
        name,
        pm.icdf(
            pm.Gamma.dist(
                alpha=dispersion * mean_vec, beta=dispersion, shape=len(mean_vec)
            ),
            u,
        ),
    )
