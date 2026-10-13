import numpy as np
import pymc as pm
import pytensor.tensor as pt

from endoutbreakhet.utils import (
    delay_matrix,
    get_likelihood_idx,
    parse_lognormal_prior,
)


def fit_model(
    *,
    model_name,
    incidence_vec,
    switch_day,
    delay_distributions,
    priors,
    rng,
    step=None,
    thin=1,
    **sample_kwargs,
):
    pm_model = _build_pymc_model(
        model_name=model_name,
        incidence_vec=incidence_vec,
        switch_day=switch_day,
        delay_distributions=delay_distributions,
        priors=priors,
    )

    with pm_model:
        if step is not None:
            sample_kwargs["step"] = step()

        trace = pm.sample(random_seed=rng, **sample_kwargs)
    if thin > 1:
        trace = trace.isel(draw=slice(0, None, thin))
        trace = trace.map_over_datasets(
            lambda ds: (
                ds.assign_coords(draw=np.arange(ds.sizes["draw"]))
                if "draw" in ds.dims
                else ds
            )
        )
    return trace


def _build_pymc_model(
    *, model_name, incidence_vec, switch_day, delay_distributions, priors
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


def _build_poisson_model(*, incidence_vec, switch_day, delay_distributions, priors):
    n_days = len(incidence_vec)
    serial_interval_matrix = delay_matrix(
        dist_vec=delay_distributions["serial_interval"], n_days=n_days, first_lag=1
    )
    total_inf_vec = serial_interval_matrix @ incidence_vec
    likelihood_idx = get_likelihood_idx(
        incidence_vec=incidence_vec, force_vec=total_inf_vec
    )

    with pm.Model() as model:
        rep_no_vec = _build_rep_no_vec(
            priors=priors, switch_day=switch_day, n_days=n_days
        )

        expected_incidence_local = rep_no_vec * total_inf_vec

        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[likelihood_idx],
            observed=incidence_vec[likelihood_idx],
        )
    return model


def _build_sse_model(*, incidence_vec, switch_day, delay_distributions, priors):
    n_days = len(incidence_vec)
    serial_interval_matrix = delay_matrix(
        dist_vec=delay_distributions["serial_interval"], n_days=n_days, first_lag=1
    )
    total_inf_vec = serial_interval_matrix @ incidence_vec
    likelihood_idx = get_likelihood_idx(
        incidence_vec=incidence_vec, force_vec=total_inf_vec
    )

    with pm.Model() as model:
        rep_no_vec = _build_rep_no_vec(
            priors=priors, switch_day=switch_day, n_days=n_days
        )
        dispersion = _prior_rv(name="dispersion", priors=priors)

        expected_incidence_local = rep_no_vec * total_inf_vec

        pm.NegativeBinomial(
            "likelihood",
            alpha=dispersion * total_inf_vec[likelihood_idx],
            mu=expected_incidence_local[likelihood_idx],
            observed=incidence_vec[likelihood_idx],
        )
    return model


def _build_ssi_model(*, incidence_vec, switch_day, delay_distributions, priors):
    n_days = len(incidence_vec)
    serial_interval_matrix = delay_matrix(
        dist_vec=delay_distributions["serial_interval"], n_days=n_days, first_lag=1
    )
    likelihood_idx = get_likelihood_idx(
        incidence_vec=incidence_vec, force_vec=serial_interval_matrix @ incidence_vec
    )
    nonzero_incidence_idx = incidence_vec > 0

    with pm.Model() as model:
        rep_no_vec = _build_rep_no_vec(
            priors=priors, switch_day=switch_day, n_days=n_days
        )
        dispersion = _prior_rv(name="dispersion", priors=priors)

        cohort_inf_case_days = _gamma_via_uniform(
            name="cohort_inf_case_days",
            mean_vec=incidence_vec[nonzero_incidence_idx],
            shape_vec=dispersion * incidence_vec[nonzero_incidence_idx],
        )
        total_inf_vec = pm.math.dot(
            serial_interval_matrix[:, nonzero_incidence_idx], cohort_inf_case_days
        )
        expected_incidence_local = rep_no_vec * total_inf_vec

        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[likelihood_idx],
            observed=incidence_vec[likelihood_idx],
        )
    return model


def _build_dlo_model(*, incidence_vec, switch_day, delay_distributions, priors):
    n_days = len(incidence_vec)
    serial_interval_matrix = delay_matrix(
        dist_vec=delay_distributions["serial_interval"], n_days=n_days, first_lag=1
    )
    total_inf_vec = serial_interval_matrix @ incidence_vec
    likelihood_idx = get_likelihood_idx(
        incidence_vec=incidence_vec, force_vec=total_inf_vec
    )

    with pm.Model() as model:
        rep_no_vec = _build_rep_no_vec(
            priors=priors, switch_day=switch_day, n_days=n_days
        )
        dispersion = _prior_rv(name="dispersion", priors=priors)

        expected_incidence_local = rep_no_vec * total_inf_vec

        pm.NegativeBinomial(
            "likelihood",
            alpha=dispersion,
            mu=expected_incidence_local[likelihood_idx],
            observed=incidence_vec[likelihood_idx],
        )
    return model


def _build_poisson_so_model(*, incidence_vec, switch_day, delay_distributions, priors):
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

    with pm.Model() as model:
        rep_no_vec = _build_rep_no_vec(
            priors=priors, switch_day=switch_day, n_days=n_days
        )

        expected_incidence_local = pm.math.dot(
            inc_period_matrix, rep_no_vec * total_inf_vec
        )

        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[likelihood_idx],
            observed=incidence_vec[likelihood_idx],
        )
    return model


def _build_sse_so_model(*, incidence_vec, switch_day, delay_distributions, priors):
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

    with pm.Model() as model:
        rep_no_vec = _build_rep_no_vec(
            priors=priors, switch_day=switch_day, n_days=n_days
        )
        dispersion = _prior_rv(name="dispersion", priors=priors)

        total_inf_nonzero_days = _gamma_via_uniform(
            name="total_inf_nonzero_days",
            mean_vec=expected_total_inf_vec[nonzero_total_inf_idx],
            shape_vec=dispersion * expected_total_inf_vec[nonzero_total_inf_idx],
        )
        expected_incidence_local = pm.math.dot(
            inc_period_matrix[:, nonzero_total_inf_idx],
            total_inf_nonzero_days * rep_no_vec[nonzero_total_inf_idx],
        )

        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[likelihood_idx],
            observed=incidence_vec[likelihood_idx],
        )
    return model


def _build_ssi_so_model(*, incidence_vec, switch_day, delay_distributions, priors):
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

    with pm.Model() as model:
        rep_no_vec = _build_rep_no_vec(
            priors=priors, switch_day=switch_day, n_days=n_days
        )
        dispersion = _prior_rv(name="dispersion", priors=priors)

        cohort_inf_case_days = _gamma_via_uniform(
            name="cohort_inf_case_days",
            mean_vec=incidence_vec[nonzero_incidence_idx],
            shape_vec=dispersion * incidence_vec[nonzero_incidence_idx],
        )
        total_inf_vec = pm.math.dot(
            tost_matrix[:, nonzero_incidence_idx], cohort_inf_case_days
        )
        expected_infection_incidence_local = rep_no_vec * total_inf_vec
        expected_incidence_local = pm.math.dot(
            inc_period_matrix, expected_infection_incidence_local
        )

        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[likelihood_idx],
            observed=incidence_vec[likelihood_idx],
        )
    return model


def _build_rep_no_vec(*, priors, switch_day, n_days):
    rep_no_before = _prior_rv(name="rep_no_before", priors=priors)
    rep_no_after = _prior_rv(name="rep_no_after", priors=priors)

    return pm.math.concatenate(
        [
            pm.math.full(switch_day, rep_no_before),
            pm.math.full(n_days - switch_day, rep_no_after),
        ]
    )


def _prior_rv(*, name, priors):
    # A fixed parameter is a deterministic, so it's in the trace like an estimated one
    if priors[name].get("dist", None) == "fixed":
        return pm.Deterministic(name, pt.as_tensor(priors[name]["value"]))
    return pm.LogNormal(name, **parse_lognormal_prior(name=name, priors=priors))


def _gamma_via_uniform(*, name, mean_vec, shape_vec):
    # Gamma variable with given mean and shape. Inverse gamma transformation prevents NUTS
    # sampling but improves efficiency
    u = pm.Uniform("u", 0, 1, shape=len(mean_vec))
    return pm.Deterministic(
        name,
        pm.icdf(
            pm.Gamma.dist(
                alpha=shape_vec, beta=shape_vec / mean_vec, shape=len(mean_vec)
            ),
            u,
        ),
    )
