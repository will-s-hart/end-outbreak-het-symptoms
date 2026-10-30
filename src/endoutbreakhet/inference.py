import functools
from collections.abc import Sequence

import arviz_stats
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.initial_point import make_initial_point_fn
from pymc.step_methods.compound import Competence, CompoundStep

from endoutbreakhet.utils import (
    delay_matrix,
    get_likelihood_idx,
    parse_lognormal_prior,
    validate_delay_distributions,
)

_MAX_INIT_ATTEMPTS = 10


def fit_model(
    *,
    model_name,
    reported_incidence_vec,
    switch_day,
    delay_distributions,
    priors,
    rng,
    reporting_prob=1.0,
    step=None,
    chains=4,
    thin=1,
    **sample_kwargs,
):
    # Posterior of the model given reported_incidence_vec, when each case is reported
    # independently with probability reporting_prob (the day-0 cases are imported, and taken
    # to be reported). step, if given, is either a step method class (e.g. pm.Slice) for the
    # continuous variables, leaving pm.sample to assign Metropolis to any unreported case
    # counts, or a factory for step methods with explicit variables. Each chain starts from
    # its own point (see _dispersed_starts), and the number of divergences after warm-up (if
    # NUTS ran) is kept in the sample stats' attrs as n_divergences
    owned_kwargs = sorted(
        {"initvals", "init", "jitter_max_retries"} & set(sample_kwargs)
    )
    if owned_kwargs:
        raise ValueError(
            f"fit_model sets the chains' starts, so doesn't take {', '.join(owned_kwargs)}"
        )
    pm_model = _build_pymc_model(
        model_name=model_name,
        reported_incidence_vec=reported_incidence_vec,
        switch_day=switch_day,
        delay_distributions=delay_distributions,
        priors=priors,
        reporting_prob=reporting_prob,
    )

    with pm_model:
        if isinstance(step, type):
            # Without vars, a step method class would take every variable
            sample_kwargs["step"] = step(vars=pm_model.continuous_value_vars)
        elif step is not None:
            sample_kwargs["step"] = step()
        if step is not None:
            _check_step(step=sample_kwargs["step"], model=pm_model)

        # Exact starts on both of pm.sample's paths: "adapt_diag" stops it jittering them
        # where NUTS samples every variable, and it never does otherwise
        trace = pm.sample(
            chains=chains,
            initvals=_dispersed_starts(
                model=pm_model,
                reported_incidence_vec=reported_incidence_vec,
                chains=chains,
                rng=rng,
            ),
            init="adapt_diag",
            random_seed=rng,
            **sample_kwargs,
        )
    # Counted before thinning, which would leave out divergent draws
    n_divergences = (
        int(trace.sample_stats["diverging"].sum())
        if "diverging" in trace.sample_stats
        else None
    )
    if thin > 1:
        trace = trace.isel(draw=slice(0, None, thin))
        trace = trace.map_over_datasets(
            lambda ds: (
                ds.assign_coords(draw=np.arange(ds.sizes["draw"]))
                if "draw" in ds.dims
                else ds
            )
        )
    if n_divergences is not None:
        trace.sample_stats.attrs["n_divergences"] = n_divergences
    return trace


def compute_fit_diagnostics(*, trace):
    # Convergence diagnostics of a fit from fit_model: the number of divergences after
    # warm-up (NaN if NUTS didn't run), and for each posterior variable (deterministics
    # included, as risks are computed from them) the numbers of elements with a non-finite
    # draw, of finite elements that are the same in every draw, and of the rest, over which
    # the worst and median rank-normalised R-hat and bulk and tail ESS are taken (NaN if
    # there are none). Elements with a non-finite draw are left out, since an infinite draw
    # can still give R-hat near 1, as are constant ones, which have no R-hat (whether they're
    # rare-event samples or chains stuck together). The keys depend only on the model, so
    # diagnostics of fits to different days stack. Also the posterior mean total of any
    # unreported cases
    diagnostics = {
        "n_divergences": trace.sample_stats.attrs.get("n_divergences", np.nan)
    }
    posterior = trace.posterior
    for name, values in posterior.data_vars.items():
        diagnostics |= _variable_diagnostics(name=name, values=values.values)
    if "unreported_incidence" in posterior:
        diagnostics["mean_total_unreported"] = float(
            posterior["unreported_incidence"].sum(dim="local_day").mean()
        )
    return diagnostics


def _build_pymc_model(
    *,
    model_name,
    reported_incidence_vec,
    switch_day,
    delay_distributions,
    priors,
    reporting_prob,
):
    # Each builder adds its model's variables to the model in context
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
    if not 0 < reporting_prob <= 1:
        raise ValueError(f"Reporting probability must be in (0, 1]: {reporting_prob}")
    validate_delay_distributions(
        model_name=model_name, delay_distributions=delay_distributions
    )

    with pm.Model() as model:
        builders[model_name](
            reported_incidence_vec=reported_incidence_vec,
            switch_day=switch_day,
            delay_distributions=delay_distributions,
            priors=priors,
            reporting_prob=reporting_prob,
        )
    return model


def _build_poisson_model(
    *, reported_incidence_vec, switch_day, delay_distributions, priors, reporting_prob
):
    n_days = len(reported_incidence_vec)
    serial_interval_matrix = delay_matrix(
        delay=delay_distributions["serial_interval"], n_days=n_days
    )

    rep_no_vec = _build_rep_no_vec(priors=priors, switch_day=switch_day, n_days=n_days)
    total_incidence_vec = _build_total_incidence_vec(
        reported_incidence_vec=reported_incidence_vec,
        force_matrix=serial_interval_matrix,
        reporting_prob=reporting_prob,
    )

    total_inf_vec = pm.math.dot(serial_interval_matrix, total_incidence_vec)
    expected_incidence_local = rep_no_vec * total_inf_vec

    _build_likelihood(
        reported_incidence_vec=reported_incidence_vec,
        total_incidence_vec=total_incidence_vec,
        expected_incidence_local=expected_incidence_local,
        force_matrix=serial_interval_matrix,
        reporting_prob=reporting_prob,
    )


def _build_sse_model(
    *, reported_incidence_vec, switch_day, delay_distributions, priors, reporting_prob
):
    n_days = len(reported_incidence_vec)
    serial_interval_matrix = delay_matrix(
        delay=delay_distributions["serial_interval"], n_days=n_days
    )

    rep_no_vec = _build_rep_no_vec(priors=priors, switch_day=switch_day, n_days=n_days)
    dispersion = _prior_rv(name="dispersion", priors=priors)
    total_incidence_vec = _build_total_incidence_vec(
        reported_incidence_vec=reported_incidence_vec,
        force_matrix=serial_interval_matrix,
        reporting_prob=reporting_prob,
    )

    total_inf_vec = pm.math.dot(serial_interval_matrix, total_incidence_vec)
    expected_incidence_local = rep_no_vec * total_inf_vec

    _build_likelihood(
        reported_incidence_vec=reported_incidence_vec,
        total_incidence_vec=total_incidence_vec,
        expected_incidence_local=expected_incidence_local,
        incidence_dispersion=dispersion * total_inf_vec,
        force_matrix=serial_interval_matrix,
        reporting_prob=reporting_prob,
    )


def _build_ssi_model(
    *, reported_incidence_vec, switch_day, delay_distributions, priors, reporting_prob
):
    n_days = len(reported_incidence_vec)
    serial_interval_matrix = delay_matrix(
        delay=delay_distributions["serial_interval"], n_days=n_days
    )
    latent_idx = _latent_idx(
        mean_vec=reported_incidence_vec, reporting_prob=reporting_prob
    )

    rep_no_vec = _build_rep_no_vec(priors=priors, switch_day=switch_day, n_days=n_days)
    dispersion = _prior_rv(name="dispersion", priors=priors)
    total_incidence_vec = _build_total_incidence_vec(
        reported_incidence_vec=reported_incidence_vec,
        force_matrix=serial_interval_matrix,
        reporting_prob=reporting_prob,
    )

    cohort_inf_vec = _gamma_via_uniform(
        name="cohort_inf",
        mean_vec=total_incidence_vec,
        rate=dispersion,
        latent_idx=latent_idx,
    )
    total_inf_vec = pm.math.dot(serial_interval_matrix, cohort_inf_vec)
    expected_incidence_local = rep_no_vec * total_inf_vec

    _build_likelihood(
        reported_incidence_vec=reported_incidence_vec,
        total_incidence_vec=total_incidence_vec,
        expected_incidence_local=expected_incidence_local,
        force_matrix=serial_interval_matrix,
        reporting_prob=reporting_prob,
    )


def _build_dlo_model(
    *, reported_incidence_vec, switch_day, delay_distributions, priors, reporting_prob
):
    n_days = len(reported_incidence_vec)
    serial_interval_matrix = delay_matrix(
        delay=delay_distributions["serial_interval"], n_days=n_days
    )

    rep_no_vec = _build_rep_no_vec(priors=priors, switch_day=switch_day, n_days=n_days)
    dispersion = _prior_rv(name="dispersion", priors=priors)
    total_incidence_vec = _build_total_incidence_vec(
        reported_incidence_vec=reported_incidence_vec,
        force_matrix=serial_interval_matrix,
        reporting_prob=reporting_prob,
    )

    total_inf_vec = pm.math.dot(serial_interval_matrix, total_incidence_vec)
    expected_incidence_local = rep_no_vec * total_inf_vec

    _build_likelihood(
        reported_incidence_vec=reported_incidence_vec,
        total_incidence_vec=total_incidence_vec,
        expected_incidence_local=expected_incidence_local,
        incidence_dispersion=dispersion,
        force_matrix=serial_interval_matrix,
        reporting_prob=reporting_prob,
    )


def _build_poisson_so_model(
    *, reported_incidence_vec, switch_day, delay_distributions, priors, reporting_prob
):
    n_days = len(reported_incidence_vec)
    tost_matrix = delay_matrix(delay=delay_distributions["tost"], n_days=n_days)
    inc_period_matrix = delay_matrix(
        delay=delay_distributions["inc_period"], n_days=n_days
    )
    force_matrix = inc_period_matrix @ tost_matrix

    rep_no_vec = _build_rep_no_vec(priors=priors, switch_day=switch_day, n_days=n_days)
    total_incidence_vec = _build_total_incidence_vec(
        reported_incidence_vec=reported_incidence_vec,
        force_matrix=force_matrix,
        reporting_prob=reporting_prob,
    )

    total_inf_vec = pm.math.dot(tost_matrix, total_incidence_vec)
    expected_incidence_local = pm.math.dot(
        inc_period_matrix, rep_no_vec * total_inf_vec
    )

    _build_likelihood(
        reported_incidence_vec=reported_incidence_vec,
        total_incidence_vec=total_incidence_vec,
        expected_incidence_local=expected_incidence_local,
        force_matrix=force_matrix,
        reporting_prob=reporting_prob,
    )


def _build_sse_so_model(
    *, reported_incidence_vec, switch_day, delay_distributions, priors, reporting_prob
):
    n_days = len(reported_incidence_vec)
    tost_matrix = delay_matrix(delay=delay_distributions["tost"], n_days=n_days)
    inc_period_matrix = delay_matrix(
        delay=delay_distributions["inc_period"], n_days=n_days
    )
    force_matrix = inc_period_matrix @ tost_matrix
    latent_idx = _latent_idx(
        mean_vec=tost_matrix @ reported_incidence_vec, reporting_prob=reporting_prob
    )

    rep_no_vec = _build_rep_no_vec(priors=priors, switch_day=switch_day, n_days=n_days)
    dispersion = _prior_rv(name="dispersion", priors=priors)
    total_incidence_vec = _build_total_incidence_vec(
        reported_incidence_vec=reported_incidence_vec,
        force_matrix=force_matrix,
        reporting_prob=reporting_prob,
    )

    expected_total_inf_vec = pm.math.dot(tost_matrix, total_incidence_vec)
    total_inf_vec = _gamma_via_uniform(
        name="total_inf",
        mean_vec=expected_total_inf_vec,
        rate=dispersion,
        latent_idx=latent_idx,
    )
    expected_incidence_local = pm.math.dot(
        inc_period_matrix, rep_no_vec * total_inf_vec
    )

    _build_likelihood(
        reported_incidence_vec=reported_incidence_vec,
        total_incidence_vec=total_incidence_vec,
        expected_incidence_local=expected_incidence_local,
        force_matrix=force_matrix,
        reporting_prob=reporting_prob,
    )


def _build_ssi_so_model(
    *, reported_incidence_vec, switch_day, delay_distributions, priors, reporting_prob
):
    n_days = len(reported_incidence_vec)
    tost_matrix = delay_matrix(delay=delay_distributions["tost"], n_days=n_days)
    inc_period_matrix = delay_matrix(
        delay=delay_distributions["inc_period"], n_days=n_days
    )
    force_matrix = inc_period_matrix @ tost_matrix
    latent_idx = _latent_idx(
        mean_vec=reported_incidence_vec, reporting_prob=reporting_prob
    )

    rep_no_vec = _build_rep_no_vec(priors=priors, switch_day=switch_day, n_days=n_days)
    dispersion = _prior_rv(name="dispersion", priors=priors)
    total_incidence_vec = _build_total_incidence_vec(
        reported_incidence_vec=reported_incidence_vec,
        force_matrix=force_matrix,
        reporting_prob=reporting_prob,
    )

    cohort_inf_vec = _gamma_via_uniform(
        name="cohort_inf",
        mean_vec=total_incidence_vec,
        rate=dispersion,
        latent_idx=latent_idx,
    )
    total_inf_vec = pm.math.dot(tost_matrix, cohort_inf_vec)
    expected_infection_incidence_local = rep_no_vec * total_inf_vec
    expected_incidence_local = pm.math.dot(
        inc_period_matrix, expected_infection_incidence_local
    )

    _build_likelihood(
        reported_incidence_vec=reported_incidence_vec,
        total_incidence_vec=total_incidence_vec,
        expected_incidence_local=expected_incidence_local,
        force_matrix=force_matrix,
        reporting_prob=reporting_prob,
    )


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


def _build_total_incidence_vec(*, reported_incidence_vec, force_matrix, reporting_prob):
    # Incidence including any unreported cases, as latent counts on days 1, 2, ... (day-0
    # cases are imported, and taken to be reported), kept in the trace as total_incidence.
    # The latent counts get a flat prior here, as their distribution depends on earlier
    # counts and is added by _build_likelihood
    if reporting_prob == 1:
        return reported_incidence_vec
    _add_coord(name="day", values=np.arange(len(reported_incidence_vec)))
    _add_coord(name="local_day", values=np.arange(1, len(reported_incidence_vec)))
    unreported_incidence = pm.CustomDist(
        "unreported_incidence",
        logp=_nonnegative_flat_logp,
        dtype="int64",
        dims="local_day",
        initval=_initial_unreported_incidence(
            reported_incidence_vec=reported_incidence_vec, force_matrix=force_matrix
        ),
    )
    return pm.Deterministic(
        "total_incidence",
        pt.concatenate(
            [
                reported_incidence_vec[:1],
                reported_incidence_vec[1:] + unreported_incidence,
            ]
        ),
        dims="day",
    )


def _initial_unreported_incidence(*, reported_incidence_vec, force_matrix):
    # A start with nonzero density: an unreported case on each day before the last reported
    # case that has no reported cases but nonzero force of infection/disease (force_matrix @
    # incidence) from earlier cases, so that unreported cases bridge any gap the reported
    # ones leave
    total_incidence_vec = reported_incidence_vec.copy()
    for t in range(1, max(np.flatnonzero(reported_incidence_vec), default=0)):
        if total_incidence_vec[t] == 0 and force_matrix[t] @ total_incidence_vec > 0:
            total_incidence_vec[t] = 1
    # Raises if no unreported cases could explain the reported ones
    get_likelihood_idx(
        reported_incidence_vec=reported_incidence_vec,
        force_vec=force_matrix @ total_incidence_vec,
    )
    return total_incidence_vec[1:] - reported_incidence_vec[1:]


def _nonnegative_flat_logp(value):
    return pt.switch(pt.ge(value, 0), 0.0, -np.inf)


def _build_likelihood(
    *,
    reported_incidence_vec,
    total_incidence_vec,
    expected_incidence_local,
    force_matrix,
    reporting_prob,
    incidence_dispersion=None,
):
    # Likelihood of the reported incidence, when total incidence after day 0 is Poisson with
    # mean expected_incidence_local (or negative binomial with dispersion
    # incidence_dispersion, if given), and each case is reported with probability
    # reporting_prob
    if incidence_dispersion is not None:
        incidence_dispersion = pt.broadcast_to(
            incidence_dispersion, expected_incidence_local.shape
        )

    if reporting_prob == 1:
        # Local cases are only possible on days with nonzero force of infection/disease
        # (force_matrix @ incidence)
        likelihood_idx = get_likelihood_idx(
            reported_incidence_vec=reported_incidence_vec,
            force_vec=force_matrix @ reported_incidence_vec,
        )
        _add_coord(name="likelihood_day", values=np.flatnonzero(likelihood_idx))
        if incidence_dispersion is None:
            pm.Poisson(
                "likelihood",
                mu=expected_incidence_local[likelihood_idx],
                observed=reported_incidence_vec[likelihood_idx],
                dims="likelihood_day",
            )
        else:
            pm.NegativeBinomial(
                "likelihood",
                mu=expected_incidence_local[likelihood_idx],
                alpha=incidence_dispersion[likelihood_idx],
                observed=reported_incidence_vec[likelihood_idx],
                dims="likelihood_day",
            )
        return

    # Otherwise the days with local cases possible depend on the unreported cases, so the
    # model includes every day after day 0
    total_incidence_logp = _total_incidence_logp(
        total_incidence_vec=total_incidence_vec[1:],
        expected_incidence_local=expected_incidence_local[1:],
        incidence_dispersion=(
            None if incidence_dispersion is None else incidence_dispersion[1:]
        ),
    )
    pm.Potential("total_incidence_logp", total_incidence_logp.sum())
    _add_coord(name="likelihood_day", values=np.arange(1, len(reported_incidence_vec)))
    pm.Binomial(
        "likelihood",
        n=total_incidence_vec[1:],
        p=np.float64(reporting_prob),
        observed=reported_incidence_vec[1:],
        dims="likelihood_day",
    )


def _total_incidence_logp(
    *, total_incidence_vec, expected_incidence_local, incidence_dispersion
):
    # Log probability of the total incidence on each day: Poisson with mean
    # expected_incidence_local (or negative binomial with dispersion incidence_dispersion, if
    # given) where that mean is positive, and a point mass at zero where it is zero. The
    # distribution is degenerate at zero mean (NaN gradients, and zero dispersion in the SSE
    # model), so its parameters are replaced there in the unused branch
    positive_mean_idx = expected_incidence_local > 0
    safe_expected_incidence = pt.switch(positive_mean_idx, expected_incidence_local, 1)
    if incidence_dispersion is None:
        incidence_dist = pm.Poisson.dist(mu=safe_expected_incidence)
    else:
        incidence_dist = pm.NegativeBinomial.dist(
            mu=safe_expected_incidence,
            alpha=pt.switch(positive_mean_idx, incidence_dispersion, 1),
        )
    total_incidence_logp = pt.switch(
        positive_mean_idx,
        pm.logp(incidence_dist, total_incidence_vec, warn_rvs=False),
        pt.switch(pt.eq(total_incidence_vec, 0), 0, -np.inf),
    )
    # A NaN or infinite mean (from a gamma latent that's infinite in floating point) has
    # zero density, rather than falling into the zero-mean branch (NaN) or giving a NaN
    # density (infinity)
    return pt.switch(
        pt.isfinite(expected_incidence_local), total_incidence_logp, -np.inf
    )


def _latent_idx(*, mean_vec, reporting_prob):
    # Days with a gamma latent with mean mean_vec, computed from the incidence: those where
    # mean_vec is nonzero if all cases are reported, else every day, as unreported cases can
    # make it nonzero on any day
    if reporting_prob == 1:
        return mean_vec > 0
    return np.full(len(mean_vec), True)


def _gamma_via_uniform(*, name, mean_vec, rate, latent_idx):
    # Gamma variables on every day with means mean_vec and rate rate, so shape
    # rate * mean_vec, which are zero where mean_vec is. Only the days in latent_idx
    # (outside which mean_vec is zero) get a uniform u. Inverse gamma transformation of
    # uniforms improves efficiency, but has no gradient with respect to the shape, so an
    # estimated rate (the models' dispersion) can't be sampled by NUTS (u can)
    _add_coord(name="day", values=np.arange(len(latent_idx)))
    _add_coord(name="latent_day", values=np.flatnonzero(latent_idx))
    latent_mean_vec = mean_vec[latent_idx]
    nonzero_mean_idx = latent_mean_vec > 0
    # Both branches of the switch are evaluated, and a zero shape anywhere makes the
    # inverse CDF raise, so zero means are replaced in the unused branch (any positive value
    # would do)
    safe_mean_vec = pt.switch(nonzero_mean_idx, latent_mean_vec, 1)
    u = pm.Uniform("u", 0, 1, dims="latent_day")
    latent_gamma_vec = pm.icdf(pm.Gamma.dist(alpha=rate * safe_mean_vec, beta=rate), u)
    return pm.Deterministic(
        name,
        pt.set_subtensor(
            pt.zeros(len(latent_idx))[latent_idx],
            pt.switch(nonzero_mean_idx, latent_gamma_vec, 0),
        ),
        dims="day",
    )


def _add_coord(*, name, values):
    # Coordinate on the model in context, which pm.modelcontext types as a BaseModel
    # (without add_coord)
    model = pm.modelcontext(None)
    assert isinstance(model, pm.Model)
    model.add_coord(name, values=values)


def _check_step(*, step, model):
    # Raise if a step method samples a discrete variable (unreported case counts) that it
    # can't (e.g. slice sampling, whose proposals are truncated to integers), by PyMC's own
    # competence ranking
    if isinstance(step, Sequence):
        for method in step:
            _check_step(step=method, model=model)
        return
    if isinstance(step, CompoundStep):
        _check_step(step=step.methods, model=model)
        return
    for var in step.vars:
        if var not in model.discrete_value_vars:
            continue
        (competence,) = type(step)._competence(model.values_to_rvs[var], False)
        if competence == Competence.INCOMPATIBLE:
            raise ValueError(
                f"{type(step).__name__} can't sample the discrete variable {var.name}: "
                "leave it to pm.sample's automatic assignment, or use pm.Metropolis"
            )


def _dispersed_starts(*, model, reported_incidence_vec, chains, rng):
    # A start for each chain, by transformed variable name: the model's initial point with
    # U(-1, 1) jitter on the continuous variables in sampling space (as pm.sample applies
    # only where NUTS samples every variable), and any unreported case counts at their
    # initial values plus the chain's index on each day with reported cases. Those days
    # already have nonzero force of infection/disease, and adding cases only raises it on
    # later days, so the counts stay in the support. Other days, which may have zero force,
    # keep their initial values: Metropolis updates each count separately, with its own
    # tuned scale, so a count at zero in every draw of every chain is more likely rarely
    # above zero than stuck. Each start is redrawn until it has finite density (a jittered
    # gamma latent can underflow to zero)
    initial_point_func = make_initial_point_fn(
        model=model,
        jitter_rvs={model.values_to_rvs[var] for var in model.continuous_value_vars},
        return_transformed=True,
    )
    logp_func = model.compile_logp()
    starts = []
    for chain in range(chains):
        extra_unreported_vec = chain * (reported_incidence_vec[1:] > 0)
        for _ in range(_MAX_INIT_ATTEMPTS):
            start = initial_point_func(rng.integers(2**30))
            if "unreported_incidence" in start:
                start["unreported_incidence"] = (
                    start["unreported_incidence"] + extra_unreported_vec
                )
            if np.isfinite(logp_func(start)):
                break
        else:
            raise RuntimeError(
                f"No start with finite log density for chain {chain} in "
                f"{_MAX_INIT_ATTEMPTS} attempts. Log densities at the last: "
                f"{model.point_logps(start)}"
            )
        starts.append(start)
    return starts


def _variable_diagnostics(*, name, values):
    # Diagnostics of a posterior variable with values of shape (chains, draws, ...), as in
    # compute_fit_diagnostics
    element_values = values.reshape(*values.shape[:2], -1).astype(float)
    is_nonfinite_vec = ~np.all(np.isfinite(element_values), axis=(0, 1))
    is_constant_vec = ~is_nonfinite_vec & np.all(
        element_values == element_values[:1, :1], axis=(0, 1)
    )
    diagnosed_values = element_values[..., ~is_nonfinite_vec & ~is_constant_vec]
    n_diagnosed = diagnosed_values.shape[-1]
    diagnostics = {
        f"n_nonfinite_{name}": int(is_nonfinite_vec.sum()),
        f"n_constant_{name}": int(is_constant_vec.sum()),
        f"n_diagnosed_{name}": n_diagnosed,
    }
    # The worst of each over the elements, and their median, which NaNs propagate to
    for stat_name, worst_name, worst_func, stat_func in [
        ("rhat", "max", np.max, arviz_stats.rhat),
        ("ess_bulk", "min", np.min, arviz_stats.ess),
        (
            "ess_tail",
            "min",
            np.min,
            functools.partial(arviz_stats.ess, method="tail", prob=(0.05, 0.95)),
        ),
    ]:
        stat_vec = stat_func(diagnosed_values) if n_diagnosed else np.full(1, np.nan)
        diagnostics[f"{worst_name}_{stat_name}_{name}"] = float(worst_func(stat_vec))
        diagnostics[f"median_{stat_name}_{name}"] = float(np.median(stat_vec))
    return diagnostics
