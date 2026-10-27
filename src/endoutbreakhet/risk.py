import numpy as np
from scipy.optimize.elementwise import find_root

from endoutbreakhet.utils import delay_matrix, get_rep_no_vec_samples


def compute_risk(
    *,
    model_name,
    reported_incidence_vec,
    switch_day,
    delay_distributions,
    trace,
    reporting_prob=1.0,
    risk_type="cases",
):
    # Posterior mean risk, after the final day of reported_incidence_vec, of risk_type:
    # "cases" (RAC: a case, reported or not), "transmission" (RAT: an infection),
    # "sustained" (RST: transmission that never dies out) or "reports" (RAR: a reported
    # case), if interventions are lifted from the start of the next day (so R^(fut) =
    # rep_no_before on every later day). Each risk conditions on parameters and total
    # incidence (and infectiousness latents, where present), as in the notes.
    # reporting_prob must be the one the trace was fitted with: the trace only shows
    # whether reporting was complete
    risk_samples_funcs = {
        "poisson": _poisson_risk_samples,
        "sse": _sse_risk_samples,
        "ssi": _ssi_risk_samples,
        "dlo": _dlo_risk_samples,
        "poisson_so": _poisson_so_risk_samples,
        "sse_so": _sse_so_risk_samples,
        "ssi_so": _ssi_so_risk_samples,
    }
    if model_name not in risk_samples_funcs:
        raise ValueError(f"Unknown model name: {model_name}")
    if risk_type not in ("cases", "transmission", "sustained", "reports"):
        raise ValueError(f"Unknown risk type: {risk_type}")
    if not 0 < reporting_prob <= 1:
        raise ValueError(f"Reporting probability must be in (0, 1]: {reporting_prob}")
    posterior = trace.posterior
    if (reporting_prob < 1) != ("total_incidence" in posterior):
        raise ValueError(
            f"Trace doesn't match reporting probability {reporting_prob}: total incidence "
            "is latent if and only if reporting is incomplete"
        )

    if reporting_prob < 1:
        total_incidence_vec_samples = _posterior_samples(
            posterior=posterior, name="total_incidence"
        )
        if total_incidence_vec_samples.shape[1] != len(reported_incidence_vec):
            raise ValueError(
                f"Trace's total incidence covers {total_incidence_vec_samples.shape[1]} "
                f"days, but reported_incidence_vec covers {len(reported_incidence_vec)}"
            )
    else:
        # One row shared by every posterior sample
        total_incidence_vec_samples = reported_incidence_vec[None, :]

    risk_samples = risk_samples_funcs[model_name](
        total_incidence_vec_samples=total_incidence_vec_samples,
        switch_day=switch_day,
        delay_distributions=delay_distributions,
        posterior=posterior,
        risk_type=risk_type,
        reporting_prob=reporting_prob,
    )
    return float(np.mean(risk_samples))


def _poisson_risk_samples(
    *,
    total_incidence_vec_samples,
    switch_day,
    delay_distributions,
    posterior,
    risk_type,
    reporting_prob,
):
    n_days = total_incidence_vec_samples.shape[1]
    rep_no_before_samples = _posterior_samples(
        posterior=posterior, name="rep_no_before"
    )

    serial_interval_survival_weight_vec = _survival_weight_vec(
        dist_vec=delay_distributions["serial_interval"], first_lag=1, n_days=n_days
    )
    remaining_inf_samples = (
        total_incidence_vec_samples @ serial_interval_survival_weight_vec
    )

    return _branching_risk_samples(
        rep_no_fut_samples=rep_no_before_samples,
        remaining_inf_samples=remaining_inf_samples,
        risk_type=risk_type,
        reporting_prob=reporting_prob,
    )


def _sse_risk_samples(
    *,
    total_incidence_vec_samples,
    switch_day,
    delay_distributions,
    posterior,
    risk_type,
    reporting_prob,
):
    n_days = total_incidence_vec_samples.shape[1]
    rep_no_before_samples = _posterior_samples(
        posterior=posterior, name="rep_no_before"
    )
    dispersion_samples = _posterior_samples(posterior=posterior, name="dispersion")

    serial_interval_survival_weight_vec = _survival_weight_vec(
        dist_vec=delay_distributions["serial_interval"], first_lag=1, n_days=n_days
    )
    remaining_inf_samples = (
        total_incidence_vec_samples @ serial_interval_survival_weight_vec
    )

    return _branching_risk_samples(
        rep_no_fut_samples=rep_no_before_samples,
        remaining_inf_samples=remaining_inf_samples,
        remaining_dispersion_per_inf_samples=dispersion_samples,
        offspring_dispersion_samples=dispersion_samples,
        risk_type=risk_type,
        reporting_prob=reporting_prob,
    )


def _ssi_risk_samples(
    *,
    total_incidence_vec_samples,
    switch_day,
    delay_distributions,
    posterior,
    risk_type,
    reporting_prob,
):
    n_days = total_incidence_vec_samples.shape[1]
    rep_no_before_samples = _posterior_samples(
        posterior=posterior, name="rep_no_before"
    )
    dispersion_samples = _posterior_samples(posterior=posterior, name="dispersion")

    cohort_inf_vec_samples = _posterior_samples(posterior=posterior, name="cohort_inf")

    serial_interval_survival_weight_vec = _survival_weight_vec(
        dist_vec=delay_distributions["serial_interval"], first_lag=1, n_days=n_days
    )
    remaining_inf_samples = cohort_inf_vec_samples @ serial_interval_survival_weight_vec

    return _branching_risk_samples(
        rep_no_fut_samples=rep_no_before_samples,
        remaining_inf_samples=remaining_inf_samples,
        offspring_dispersion_samples=dispersion_samples,
        risk_type=risk_type,
        reporting_prob=reporting_prob,
    )


def _dlo_risk_samples(
    *,
    total_incidence_vec_samples,
    switch_day,
    delay_distributions,
    posterior,
    risk_type,
    reporting_prob,
):
    if risk_type in ("sustained", "reports"):
        raise NotImplementedError(
            "DLO risk of sustained transmission or reports is not implemented"
        )
    n_days = total_incidence_vec_samples.shape[1]
    rep_no_before_samples = _posterior_samples(
        posterior=posterior, name="rep_no_before"
    )
    dispersion_samples = _posterior_samples(posterior=posterior, name="dispersion")

    serial_interval_dist_vec = delay_distributions["serial_interval"]
    serial_interval_max = len(serial_interval_dist_vec)
    # Force of infection on each day after the final day from cases up to the final day
    # (sums to remaining_inf in the other naive models)
    remaining_inf_vec_samples = (
        np.pad(total_incidence_vec_samples, ((0, 0), (0, serial_interval_max)))
        @ delay_matrix(
            dist_vec=serial_interval_dist_vec,
            n_days=n_days + serial_interval_max,
            first_lag=1,
        ).T
    )[:, n_days:]

    log_prob_zero_samples = _log_pgf(
        mean=rep_no_before_samples[:, None] * remaining_inf_vec_samples,
        dispersion=dispersion_samples[:, None],
        z=0.0,
    ).sum(axis=1)
    return -np.expm1(log_prob_zero_samples)


def _poisson_so_risk_samples(
    *,
    total_incidence_vec_samples,
    switch_day,
    delay_distributions,
    posterior,
    risk_type,
    reporting_prob,
):
    n_days = total_incidence_vec_samples.shape[1]
    rep_no_before_samples = _posterior_samples(
        posterior=posterior, name="rep_no_before"
    )
    rep_no_after_samples = _posterior_samples(posterior=posterior, name="rep_no_after")

    tost_survival_weight_vec = _survival_weight_vec(
        dist_vec=delay_distributions["tost"], first_lag=0, n_days=n_days
    )
    remaining_inf_samples = total_incidence_vec_samples @ tost_survival_weight_vec

    tost_matrix = delay_matrix(
        dist_vec=delay_distributions["tost"], n_days=n_days, first_lag=0
    )
    total_inf_vec_samples = total_incidence_vec_samples @ tost_matrix.T
    expected_incubating_samples = _expected_incubating_samples(
        total_inf_vec_samples=total_inf_vec_samples,
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        inc_period_dist_vec=delay_distributions["inc_period"],
    )

    return _branching_risk_samples(
        rep_no_fut_samples=rep_no_before_samples,
        remaining_inf_samples=remaining_inf_samples,
        expected_incubating_samples=expected_incubating_samples,
        risk_type=risk_type,
        reporting_prob=reporting_prob,
    )


def _sse_so_risk_samples(
    *,
    total_incidence_vec_samples,
    switch_day,
    delay_distributions,
    posterior,
    risk_type,
    reporting_prob,
):
    n_days = total_incidence_vec_samples.shape[1]
    rep_no_before_samples = _posterior_samples(
        posterior=posterior, name="rep_no_before"
    )
    rep_no_after_samples = _posterior_samples(posterior=posterior, name="rep_no_after")
    dispersion_samples = _posterior_samples(posterior=posterior, name="dispersion")

    tost_survival_weight_vec = _survival_weight_vec(
        dist_vec=delay_distributions["tost"], first_lag=0, n_days=n_days
    )
    remaining_inf_samples = total_incidence_vec_samples @ tost_survival_weight_vec

    total_inf_vec_samples = _posterior_samples(posterior=posterior, name="total_inf")
    expected_incubating_samples = _expected_incubating_samples(
        total_inf_vec_samples=total_inf_vec_samples,
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        inc_period_dist_vec=delay_distributions["inc_period"],
    )

    return _branching_risk_samples(
        rep_no_fut_samples=rep_no_before_samples,
        remaining_inf_samples=remaining_inf_samples,
        remaining_dispersion_per_inf_samples=dispersion_samples,
        offspring_dispersion_samples=dispersion_samples,
        expected_incubating_samples=expected_incubating_samples,
        risk_type=risk_type,
        reporting_prob=reporting_prob,
    )


def _ssi_so_risk_samples(
    *,
    total_incidence_vec_samples,
    switch_day,
    delay_distributions,
    posterior,
    risk_type,
    reporting_prob,
):
    n_days = total_incidence_vec_samples.shape[1]
    rep_no_before_samples = _posterior_samples(
        posterior=posterior, name="rep_no_before"
    )
    rep_no_after_samples = _posterior_samples(posterior=posterior, name="rep_no_after")
    dispersion_samples = _posterior_samples(posterior=posterior, name="dispersion")

    cohort_inf_vec_samples = _posterior_samples(posterior=posterior, name="cohort_inf")

    tost_survival_weight_vec = _survival_weight_vec(
        dist_vec=delay_distributions["tost"], first_lag=0, n_days=n_days
    )
    remaining_inf_samples = cohort_inf_vec_samples @ tost_survival_weight_vec

    tost_matrix = delay_matrix(
        dist_vec=delay_distributions["tost"], n_days=n_days, first_lag=0
    )
    total_inf_vec_samples = cohort_inf_vec_samples @ tost_matrix.T
    expected_incubating_samples = _expected_incubating_samples(
        total_inf_vec_samples=total_inf_vec_samples,
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        inc_period_dist_vec=delay_distributions["inc_period"],
    )

    return _branching_risk_samples(
        rep_no_fut_samples=rep_no_before_samples,
        remaining_inf_samples=remaining_inf_samples,
        offspring_dispersion_samples=dispersion_samples,
        expected_incubating_samples=expected_incubating_samples,
        risk_type=risk_type,
        reporting_prob=reporting_prob,
    )


def _posterior_samples(*, posterior, name):
    # Posterior samples of a variable, with chains and draws flattened into the first axis
    values = posterior[name].values
    n_samples = values.shape[0] * values.shape[1]
    return values.reshape(n_samples, *values.shape[2:])


def _survival_weight_vec(*, dist_vec, first_lag, n_days):
    # Weights on days 0, ..., n_days - 1 such that weight_vec @ x_vec totals the delayed
    # contributions of x_vec that fall after the final day, for a delay whose pmf at lag
    # first_lag + i is dist_vec[i]. The cohort of age a (a = 1 on the final day) has weight
    # 1 - F_{a-1}, as in the notes
    pmf_vec = np.concatenate([np.zeros(first_lag), dist_vec, np.zeros(n_days)])
    # survival_vec[a] = P(delay >= a) = 1 - F_{a-1}. Summed from the upper tail
    # (zero-padded so every age up to n_days is covered), so it is exactly zero past
    # the support
    survival_vec = np.cumsum(pmf_vec[::-1])[::-1]
    # Day j has age n_days - j
    return survival_vec[n_days:0:-1]


def _expected_incubating_samples(
    *,
    total_inf_vec_samples,
    rep_no_before_samples,
    rep_no_after_samples,
    switch_day,
    inc_period_dist_vec,
):
    # Expected number infected up to the final day with onset after it (N^(inc) bar), from
    # the total infectiousness on each day (lambda_t, or Z_t in SSE-SO)
    n_days = total_inf_vec_samples.shape[1]
    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )
    expected_infection_incidence_local_samples = (
        rep_no_vec_samples * total_inf_vec_samples
    )
    inc_period_survival_weight_vec = _survival_weight_vec(
        dist_vec=inc_period_dist_vec, first_lag=1, n_days=n_days
    )
    return expected_infection_incidence_local_samples @ inc_period_survival_weight_vec


def _branching_risk_samples(
    *,
    rep_no_fut_samples,
    remaining_inf_samples,
    risk_type,
    reporting_prob,
    remaining_dispersion_per_inf_samples=None,
    offspring_dispersion_samples=None,
    expected_incubating_samples=None,
):
    # Risk 1 - G^(rem)(z) G^(inc)(z') in a branching model with N^(rem) ~
    # Poisson(R^(fut) Lambda^(rem)) (or NB(R^(fut) Lambda^(rem), k Lambda^(rem)), given
    # remaining_dispersion_per_inf_samples k), N^(inc) ~
    # Poisson(expected_incubating_samples) (zero in naive models) and offspring
    # Poisson(R^(fut)) (or NB(R^(fut), k), given offspring_dispersion_samples k), with
    # (z, z') = (0, 0), (0, p_0^(off)), (q, q) or (q_rho, q_rho) by risk type, as in the
    # notes
    if risk_type == "cases":
        remaining_pgf_arg = incubating_pgf_arg = 0.0
    elif risk_type == "transmission":
        remaining_pgf_arg = 0.0
        incubating_pgf_arg = np.exp(
            _log_pgf(
                mean=rep_no_fut_samples, dispersion=offspring_dispersion_samples, z=0.0
            )
        )
    else:
        # q at reporting probability 0, for sustained transmission
        remaining_pgf_arg = incubating_pgf_arg = _lineage_unreported_extinction_prob(
            rep_no_samples=rep_no_fut_samples,
            dispersion_samples=offspring_dispersion_samples,
            reporting_prob=0.0 if risk_type == "sustained" else reporting_prob,
        )

    # NB(R Lambda, k Lambda) has log PGF Lambda times that of NB(R, k) (likewise Poisson),
    # which stays finite at Lambda = 0, where the totals' ratio is 0/0
    log_pgf_remaining_samples = remaining_inf_samples * _log_pgf(
        mean=rep_no_fut_samples,
        dispersion=remaining_dispersion_per_inf_samples,
        z=remaining_pgf_arg,
    )
    log_pgf_incubating_samples = (
        0.0
        if expected_incubating_samples is None
        else _log_pgf(
            mean=expected_incubating_samples, dispersion=None, z=incubating_pgf_arg
        )
    )
    return -np.expm1(log_pgf_remaining_samples + log_pgf_incubating_samples)


def _log_pgf(*, mean, dispersion, z):
    # Log PGF at z of Poisson(mean), or of NB(mean, dispersion) given a dispersion (as
    # pm.NegativeBinomial(mu=mean, alpha=dispersion))
    if dispersion is None:
        return -mean * (1 - z)
    return -dispersion * np.log1p(mean / dispersion * (1 - z))


def _lineage_unreported_extinction_prob(
    *, rep_no_samples, dispersion_samples, reporting_prob
):
    # Probability that a lineage dies out with no reported case, for offspring
    # Poisson(rep_no) (or NB(rep_no, dispersion), given dispersion_samples): q_rho (a
    # lineage that never dies out has a reported case almost surely), or the extinction
    # probability q if reporting_prob is 0. It is the smallest solution in [0, 1] of
    # x = (1 - reporting_prob) G^(off)(x)
    if reporting_prob == 1:
        # Every founder is reported
        return np.zeros_like(rep_no_samples)

    # Lower bracket: half the probability of dying out unreported within one generation
    # (an unreported founder with no offspring), which is at most the solution. Halving
    # makes the left side of the equation solved below at most -log 2 there (as G^(off) is
    # increasing), so rounding can't flip its sign when the solution is that probability
    # to working precision
    lower_samples = (
        (1 - reporting_prob)
        * np.exp(_log_pgf(mean=rep_no_samples, dispersion=dispersion_samples, z=0.0))
        / 2
    )
    # Upper bracket: (1 - reporting_prob) G^(off)(x) - x is positive at x = 0 and convex,
    # so it decreases until a turning point, where (1 - reporting_prob) G^(off)'(x) = 1,
    # and the smallest solution comes before that. The turning point is beyond 1 if the
    # mean number of unreported offspring, (1 - reporting_prob) G^(off)'(1), is at most 1.
    # Otherwise, stopping at it excludes the solution x = 1 when reporting_prob is 0
    unreported_rep_no_samples = (1 - reporting_prob) * rep_no_samples
    if dispersion_samples is None:
        # G^(off)'(x) = R exp(-R (1 - x))
        turning_point_samples = 1 - np.log(unreported_rep_no_samples) / rep_no_samples
    else:
        # G^(off)'(x) = R (1 + R (1 - x) / k)^(-k - 1). Via expm1, as
        # ((1 - reporting_prob) R)^(1 / (k + 1)) rounds to 1 for large k
        turning_point_samples = 1 - dispersion_samples / rep_no_samples * np.expm1(
            np.log(unreported_rep_no_samples) / (dispersion_samples + 1)
        )
    upper_samples = np.minimum(turning_point_samples, 1)

    # Solved as log x - log(1 - reporting_prob) - log G^(off)(x) = 0, which keeps full
    # precision near x = 1, where these terms are small. Exponentiating would round
    # G^(off)(x) to near 1, losing precision in its difference from x. find_root passes x
    # and args positionally, slicing args to the unconverged elements
    result = find_root(
        lambda x, rep_no, dispersion=None: (
            np.log(x)
            - np.log1p(-reporting_prob)
            - _log_pgf(mean=rep_no, dispersion=dispersion, z=x)
        ),
        (lower_samples, upper_samples),
        args=(
            (rep_no_samples,)
            if dispersion_samples is None
            else (rep_no_samples, dispersion_samples)
        ),
    )
    if not np.all(result.success):
        raise RuntimeError(
            "Failed to solve for the probability of dying out unreported"
        )
    return result.x
