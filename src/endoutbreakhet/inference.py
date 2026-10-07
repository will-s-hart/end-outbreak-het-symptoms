import numpy as np
import pymc as pm

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
    priors = priors or {}

    pm_model = _build_pymc_model(
        model_name=model_name,
        incidence_vec=incidence_vec,
        switch_day=switch_day,
        delay_distributions=delay_distributions,
        priors=priors,
    )

    with pm_model as model:
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
    model_name, incidence_vec, switch_day, delay_distributions=None, priors=None
):
    if model_name == "sse":
        return _build_sse_model(
            incidence_vec=incidence_vec,
            switch_day=switch_day,
            gen_time_dist_vec=delay_distributions["gen_time"],
            priors=priors,
        )
    elif model_name == "ssi":
        return _build_ssi_model(
            incidence_vec=incidence_vec,
            switch_day=switch_day,
            gen_time_dist_vec=delay_distributions["gen_time"],
            priors=priors,
        )
    elif model_name == "dlo":
        return _build_dlo_model(
            incidence_vec=incidence_vec,
            switch_day=switch_day,
            gen_time_dist_vec=delay_distributions["gen_time"],
            priors=priors,
        )
    elif model_name == "sse_so":
        return _build_sse_so_model(
            incidence_vec=incidence_vec,
            switch_day=switch_day,
            tost_dist_vec=delay_distributions["tost"],
            inc_period_dist_vec=delay_distributions["inc_period"],
            priors=priors,
        )
    elif model_name == "ssi_so":
        return _build_ssi_so_model(
            incidence_vec=incidence_vec,
            switch_day=switch_day,
            tost_dist_vec=delay_distributions["tost"],
            inc_period_dist_vec=delay_distributions["inc_period"],
            priors=priors,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def _build_sse_model(incidence_vec, switch_day, gen_time_dist_vec=None, priors=None):
    n_days = len(incidence_vec)
    gen_time_dist_vec = np.concatenate([
        gen_time_dist_vec,
        np.zeros(n_days - 1 - len(gen_time_dist_vec)),
    ])

    incidence_vec_local = np.zeros(n_days)
    incidence_vec_local[1:] = incidence_vec[1:]

    foi_vec = np.zeros(n_days)
    for t in range(1, n_days):
        foi_vec[t] = np.sum(incidence_vec[:t][::-1] * gen_time_dist_vec[:t])

    nonzero_foi_idx = foi_vec > 0
    if np.any(incidence_vec_local[~nonzero_foi_idx]):
        raise ValueError(
            "Local incidence cannot be greater than zero when force of infection is 0."
        )

    with pm.Model() as model:
        rep_no_vec, dispersion = _build_rep_no_vec_dispersion(
            priors=priors, switch_day=switch_day, n_days=n_days
        )

        expected_incidence_local = rep_no_vec * foi_vec

        pm.NegativeBinomial(
            "likelihood",
            alpha=dispersion * foi_vec[nonzero_foi_idx],
            mu=expected_incidence_local[nonzero_foi_idx],
            observed=incidence_vec_local[nonzero_foi_idx],
        )
    return model


def _build_ssi_model(incidence_vec, switch_day, gen_time_dist_vec=None, priors=None):
    n_days = len(incidence_vec)
    gen_time_dist_vec = np.concatenate([
        gen_time_dist_vec,
        np.zeros(n_days - 1 - len(gen_time_dist_vec)),
    ])

    nonzero_incidence_idx = incidence_vec > 0
    n_case_days = int(np.sum(nonzero_incidence_idx))
    incidence_vec_local = np.zeros(n_days)
    incidence_vec_local[1:] = incidence_vec[1:]

    conv_matrix = np.zeros((n_days, n_days))
    for t in range(1, n_days):
        conv_matrix[t, :t] = gen_time_dist_vec[:t][::-1]

    foi_expected_vec = np.dot(conv_matrix, incidence_vec)
    nonzero_foi_idx = foi_expected_vec > 0
    if np.any(incidence_vec_local[~nonzero_foi_idx]):
        raise ValueError(
            "Local incidence cannot be greater than zero when force of infection is 0."
        )

    with pm.Model() as model:
        rep_no_vec, dispersion = _build_rep_no_vec_dispersion(
            priors=priors, switch_day=switch_day, n_days=n_days
        )

        # Inverse gamma transformation prevents NUTS sampling but improves efficiency
        u = pm.Uniform("u", 0, 1, shape=n_case_days)
        scaled_incidence_case_days = pm.Deterministic(
            "scaled_incidence_case_days",
            pm.icdf(
                pm.Gamma.dist(
                    alpha=dispersion * (incidence_vec[nonzero_incidence_idx]),
                    beta=dispersion,
                    shape=n_case_days,
                ),
                u,
            ),
        )

        foi_vec = pm.math.dot(
            conv_matrix[:, nonzero_incidence_idx], scaled_incidence_case_days
        )
        expected_incidence_local = rep_no_vec * foi_vec

        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[nonzero_foi_idx],
            observed=incidence_vec_local[nonzero_foi_idx],
        )
    return model


def _build_dlo_model(incidence_vec, switch_day, gen_time_dist_vec=None, priors=None):
    n_days = len(incidence_vec)
    gen_time_dist_vec = np.concatenate([
        gen_time_dist_vec,
        np.zeros(n_days - 1 - len(gen_time_dist_vec)),
    ])

    incidence_vec_local = np.zeros(n_days)
    incidence_vec_local[1:] = incidence_vec[1:]

    foi_vec = np.zeros(n_days)
    for t in range(1, n_days):
        foi_vec[t] = np.sum(incidence_vec[:t][::-1] * gen_time_dist_vec[:t])

    nonzero_foi_idx = foi_vec > 0
    if np.any(incidence_vec_local[~nonzero_foi_idx]):
        raise ValueError(
            "Local incidence cannot be greater than zero when force of infection is 0."
        )

    with pm.Model() as model:
        rep_no_vec, dispersion = _build_rep_no_vec_dispersion(
            priors=priors, switch_day=switch_day, n_days=n_days
        )

        expected_incidence_local = rep_no_vec * foi_vec

        pm.NegativeBinomial(
            "likelihood",
            alpha=dispersion,
            mu=expected_incidence_local[nonzero_foi_idx],
            observed=incidence_vec_local[nonzero_foi_idx],
        )
    return model


def _build_sse_so_model(
    incidence_vec, switch_day, tost_dist_vec=None, inc_period_dist_vec=None, priors=None
):
    n_days = len(incidence_vec)
    tost_dist_vec = np.concatenate([
        tost_dist_vec,
        np.zeros(n_days - len(tost_dist_vec)),
    ])
    inc_period_dist_vec = np.concatenate([
        inc_period_dist_vec,
        np.zeros(n_days - 1 - len(inc_period_dist_vec)),
    ])

    incidence_vec_local = np.zeros(n_days)
    incidence_vec_local[1:] = incidence_vec[1:]

    foi_vec = np.zeros(n_days)
    for t in range(n_days):
        foi_vec[t] = np.sum(incidence_vec[: (t + 1)][::-1] * tost_dist_vec[: (t + 1)])
    nonzero_foi_idx = foi_vec > 0

    conv_matrix = np.zeros((n_days, n_days))
    for t in range(1, n_days):
        conv_matrix[t, :t] = inc_period_dist_vec[:t][::-1]

    nonzero_fod_idx = np.dot(conv_matrix, foi_vec) > 0
    if np.any(incidence_vec_local[~nonzero_fod_idx]):
        raise ValueError(
            "Local incidence cannot be greater than zero when force of disease is 0."
        )

    with pm.Model() as model:
        rep_no_vec, dispersion = _build_rep_no_vec_dispersion(
            priors=priors, switch_day=switch_day, n_days=n_days
        )

        u = pm.Uniform("u", 0, 1, shape=int(np.sum(nonzero_foi_idx)))
        foi_scaled_nonzero_days = pm.Deterministic(
            "foi_scaled_nonzero_days",
            pm.icdf(
                pm.Gamma.dist(
                    alpha=dispersion * foi_vec[nonzero_foi_idx],
                    beta=dispersion,
                    shape=int(np.sum(nonzero_foi_idx)),
                ),
                u,
            ),
        )

        expected_incidence_local = pm.math.dot(
            conv_matrix[:, nonzero_foi_idx],
            foi_scaled_nonzero_days * rep_no_vec[nonzero_foi_idx],
        )

        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[nonzero_fod_idx],
            observed=incidence_vec_local[nonzero_fod_idx],
        )
    return model


def _build_ssi_so_model(
    incidence_vec,
    switch_day,
    tost_dist_vec=None,
    inc_period_dist_vec=None,
    priors=None,
):
    priors = priors or {}
    n_days = len(incidence_vec)
    tost_dist_vec = np.concatenate([
        tost_dist_vec,
        np.zeros(n_days - len(tost_dist_vec)),
    ])
    inc_period_dist_vec = np.concatenate([
        inc_period_dist_vec,
        np.zeros(n_days - 1 - len(inc_period_dist_vec)),
    ])

    nonzero_incidence_idx = incidence_vec > 0
    no_case_days = int(np.sum(nonzero_incidence_idx))
    incidence_vec_local = np.zeros(n_days)
    incidence_vec_local[1:] = incidence_vec[1:]

    conv_matrix_1 = np.zeros((n_days, n_days))
    for t in range(n_days):
        conv_matrix_1[t, : (t + 1)] = tost_dist_vec[: (t + 1)][::-1]

    conv_matrix_2 = np.zeros((n_days, n_days))
    for t in range(1, n_days):
        conv_matrix_2[t, :t] = inc_period_dist_vec[:t][::-1]

    nonzero_fod_idx = np.dot(conv_matrix_2, np.dot(conv_matrix_1, incidence_vec)) > 0
    if np.any(incidence_vec_local[~nonzero_fod_idx]):
        raise ValueError(
            "Local incidence cannot be greater than zero when force of disease is 0."
        )

    with pm.Model() as model:
        rep_no_vec, dispersion = _build_rep_no_vec_dispersion(
            priors=priors, switch_day=switch_day, n_days=n_days
        )

        u = pm.Uniform("u", 0, 1, shape=no_case_days)
        scaled_incidence_case_days = pm.Deterministic(
            "scaled_incidence_case_days",
            pm.icdf(
                pm.Gamma.dist(
                    alpha=dispersion * (incidence_vec[nonzero_incidence_idx]),
                    beta=dispersion,
                    shape=no_case_days,
                ),
                u,
            ),
        )

        expected_infection_incidence_local = rep_no_vec * pm.math.dot(
            conv_matrix_1[:, nonzero_incidence_idx],
            scaled_incidence_case_days,
        )
        expected_incidence_local = pm.math.dot(
            conv_matrix_2, expected_infection_incidence_local
        )

        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[nonzero_fod_idx],
            observed=incidence_vec_local[nonzero_fod_idx],
        )
    return model


def _build_rep_no_vec_dispersion(priors, switch_day, n_days):
    rep_no_before_prior_func, rep_no_before_prior_kwargs = priors.get(
        "rep_no_before", DEFAULT_PRIORS["rep_no_before"]
    )
    rep_no_after_prior_func, rep_no_after_prior_kwargs = priors.get(
        "rep_no_after", DEFAULT_PRIORS["rep_no_after"]
    )
    dispersion_prior_func, dispersion_prior_kwargs = priors.get(
        "dispersion", DEFAULT_PRIORS["dispersion"]
    )
    rep_no_before = rep_no_before_prior_func(
        "rep_no_before", **rep_no_before_prior_kwargs
    )
    rep_no_after = rep_no_after_prior_func("rep_no_after", **rep_no_after_prior_kwargs)
    dispersion = dispersion_prior_func("dispersion", **dispersion_prior_kwargs)

    rep_no_vec = pm.math.concatenate([
        pm.math.full(switch_day, rep_no_before),
        pm.math.full(n_days - switch_day, rep_no_after),
    ])
    return rep_no_vec, dispersion
