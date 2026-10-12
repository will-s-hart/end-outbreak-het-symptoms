import numpy as np

from endoutbreakhet.utils import delay_matrix, get_rep_no_vec_samples


def compute_risk(
    *,
    model_name,
    incidence_vec,
    switch_day,
    delay_distributions,
    trace,
    risk_type="cases",
):
    # Risk of cases/transmission on or after the day after the final day of incidence_vec,
    # given all observed onsets, if interventions are lifted from the start of that day (so
    # R^(fut) = rep_no_before on every later day). Posterior mean of the risk conditional
    # on parameters (and latents, where present)
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
    if risk_type not in ("cases", "transmission"):
        raise ValueError(f"Unknown risk type: {risk_type}")

    risk_samples = risk_samples_funcs[model_name](
        incidence_vec=incidence_vec,
        switch_day=switch_day,
        delay_distributions=delay_distributions,
        posterior=trace.posterior,
        risk_type=risk_type,
    )
    return float(np.mean(risk_samples))


def _poisson_risk_samples(
    *, incidence_vec, switch_day, delay_distributions, posterior, risk_type
):
    n_days = len(incidence_vec)
    rep_no_before_samples = _posterior_samples(
        posterior=posterior, name="rep_no_before"
    )

    serial_interval_survival_weight_vec = _survival_weight_vec(
        dist_vec=delay_distributions["serial_interval"], first_lag=1, n_days=n_days
    )
    remaining_inf = serial_interval_survival_weight_vec @ incidence_vec
    return -np.expm1(-rep_no_before_samples * remaining_inf)


def _sse_risk_samples(
    *, incidence_vec, switch_day, delay_distributions, posterior, risk_type
):
    n_days = len(incidence_vec)
    rep_no_before_samples = _posterior_samples(
        posterior=posterior, name="rep_no_before"
    )
    dispersion_samples = _posterior_samples(posterior=posterior, name="dispersion")

    serial_interval_survival_weight_vec = _survival_weight_vec(
        dist_vec=delay_distributions["serial_interval"], first_lag=1, n_days=n_days
    )
    remaining_inf = serial_interval_survival_weight_vec @ incidence_vec

    log_prob_zero_offspring_samples = _nbinom_log_prob_zero(
        mean=rep_no_before_samples, dispersion=dispersion_samples
    )
    return -np.expm1(log_prob_zero_offspring_samples * remaining_inf)


def _ssi_risk_samples(
    *, incidence_vec, switch_day, delay_distributions, posterior, risk_type
):
    n_days = len(incidence_vec)
    rep_no_before_samples = _posterior_samples(
        posterior=posterior, name="rep_no_before"
    )

    cohort_inf_case_day_samples = _posterior_samples(
        posterior=posterior, name="cohort_inf_case_days"
    )

    serial_interval_survival_weight_vec = _survival_weight_vec(
        dist_vec=delay_distributions["serial_interval"], first_lag=1, n_days=n_days
    )
    nonzero_incidence_idx = incidence_vec > 0
    remaining_inf_samples = (
        cohort_inf_case_day_samples
        @ serial_interval_survival_weight_vec[nonzero_incidence_idx]
    )
    return -np.expm1(-rep_no_before_samples * remaining_inf_samples)


def _dlo_risk_samples(
    *, incidence_vec, switch_day, delay_distributions, posterior, risk_type
):
    n_days = len(incidence_vec)
    rep_no_before_samples = _posterior_samples(
        posterior=posterior, name="rep_no_before"
    )
    dispersion_samples = _posterior_samples(posterior=posterior, name="dispersion")

    serial_interval_dist_vec = delay_distributions["serial_interval"]
    serial_interval_max = len(serial_interval_dist_vec)
    # Force of infection on each day after the final day from cases up to the final day
    # (sums to remaining_inf in the other naive models)
    remaining_inf_vec = (
        delay_matrix(
            dist_vec=serial_interval_dist_vec,
            n_days=n_days + serial_interval_max,
            first_lag=1,
        )
        @ np.concatenate([incidence_vec, np.zeros(serial_interval_max)])
    )[n_days:]

    log_prob_zero_samples = _nbinom_log_prob_zero(
        mean=rep_no_before_samples[:, None] * remaining_inf_vec,
        dispersion=dispersion_samples[:, None],
    ).sum(axis=1)
    return -np.expm1(log_prob_zero_samples)


def _poisson_so_risk_samples(
    *, incidence_vec, switch_day, delay_distributions, posterior, risk_type
):
    n_days = len(incidence_vec)
    rep_no_before_samples = _posterior_samples(
        posterior=posterior, name="rep_no_before"
    )
    rep_no_after_samples = _posterior_samples(posterior=posterior, name="rep_no_after")

    tost_matrix = delay_matrix(
        dist_vec=delay_distributions["tost"], n_days=n_days, first_lag=0
    )
    total_inf_vec = tost_matrix @ incidence_vec
    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )
    expected_infection_incidence_local_samples = rep_no_vec_samples * total_inf_vec

    tost_survival_weight_vec = _survival_weight_vec(
        dist_vec=delay_distributions["tost"], first_lag=0, n_days=n_days
    )
    remaining_inf = tost_survival_weight_vec @ incidence_vec
    inc_period_survival_weight_vec = _survival_weight_vec(
        dist_vec=delay_distributions["inc_period"], first_lag=1, n_days=n_days
    )
    expected_incubating_samples = (
        expected_infection_incidence_local_samples @ inc_period_survival_weight_vec
    )

    log_prob_zero_offspring_samples = -rep_no_before_samples
    log_prob_zero_remaining_samples = -rep_no_before_samples * remaining_inf

    return _combine_so_risk_samples(
        log_prob_zero_remaining_samples=log_prob_zero_remaining_samples,
        expected_incubating_samples=expected_incubating_samples,
        log_prob_zero_offspring_samples=log_prob_zero_offspring_samples,
        risk_type=risk_type,
    )


def _sse_so_risk_samples(
    *, incidence_vec, switch_day, delay_distributions, posterior, risk_type
):
    n_days = len(incidence_vec)
    rep_no_before_samples = _posterior_samples(
        posterior=posterior, name="rep_no_before"
    )
    rep_no_after_samples = _posterior_samples(posterior=posterior, name="rep_no_after")
    dispersion_samples = _posterior_samples(posterior=posterior, name="dispersion")

    total_inf_nonzero_day_samples = _posterior_samples(
        posterior=posterior, name="total_inf_nonzero_days"
    )
    tost_matrix = delay_matrix(
        dist_vec=delay_distributions["tost"], n_days=n_days, first_lag=0
    )
    expected_total_inf_vec = tost_matrix @ incidence_vec
    nonzero_total_inf_idx = expected_total_inf_vec > 0
    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )

    tost_survival_weight_vec = _survival_weight_vec(
        dist_vec=delay_distributions["tost"], first_lag=0, n_days=n_days
    )
    remaining_inf = tost_survival_weight_vec @ incidence_vec
    inc_period_survival_weight_vec = _survival_weight_vec(
        dist_vec=delay_distributions["inc_period"], first_lag=1, n_days=n_days
    )
    expected_incubating_samples = (
        total_inf_nonzero_day_samples * rep_no_vec_samples[:, nonzero_total_inf_idx]
    ) @ inc_period_survival_weight_vec[nonzero_total_inf_idx]

    log_prob_zero_offspring_samples = _nbinom_log_prob_zero(
        mean=rep_no_before_samples, dispersion=dispersion_samples
    )
    log_prob_zero_remaining_samples = log_prob_zero_offspring_samples * remaining_inf

    return _combine_so_risk_samples(
        log_prob_zero_remaining_samples=log_prob_zero_remaining_samples,
        expected_incubating_samples=expected_incubating_samples,
        log_prob_zero_offspring_samples=log_prob_zero_offspring_samples,
        risk_type=risk_type,
    )


def _ssi_so_risk_samples(
    *, incidence_vec, switch_day, delay_distributions, posterior, risk_type
):
    n_days = len(incidence_vec)
    rep_no_before_samples = _posterior_samples(
        posterior=posterior, name="rep_no_before"
    )
    rep_no_after_samples = _posterior_samples(posterior=posterior, name="rep_no_after")
    dispersion_samples = _posterior_samples(posterior=posterior, name="dispersion")

    cohort_inf_case_day_samples = _posterior_samples(
        posterior=posterior, name="cohort_inf_case_days"
    )
    tost_matrix = delay_matrix(
        dist_vec=delay_distributions["tost"], n_days=n_days, first_lag=0
    )
    nonzero_incidence_idx = incidence_vec > 0
    total_inf_vec_samples = (
        cohort_inf_case_day_samples @ tost_matrix[:, nonzero_incidence_idx].T
    )
    rep_no_vec_samples = get_rep_no_vec_samples(
        rep_no_before_samples=rep_no_before_samples,
        rep_no_after_samples=rep_no_after_samples,
        switch_day=switch_day,
        n_days=n_days,
    )
    expected_infection_incidence_local_samples = (
        rep_no_vec_samples * total_inf_vec_samples
    )

    tost_survival_weight_vec = _survival_weight_vec(
        dist_vec=delay_distributions["tost"], first_lag=0, n_days=n_days
    )
    remaining_inf_samples = (
        cohort_inf_case_day_samples @ tost_survival_weight_vec[nonzero_incidence_idx]
    )
    inc_period_survival_weight_vec = _survival_weight_vec(
        dist_vec=delay_distributions["inc_period"], first_lag=1, n_days=n_days
    )
    expected_incubating_samples = (
        expected_infection_incidence_local_samples @ inc_period_survival_weight_vec
    )

    log_prob_zero_offspring_samples = _nbinom_log_prob_zero(
        mean=rep_no_before_samples, dispersion=dispersion_samples
    )
    log_prob_zero_remaining_samples = -rep_no_before_samples * remaining_inf_samples

    return _combine_so_risk_samples(
        log_prob_zero_remaining_samples=log_prob_zero_remaining_samples,
        expected_incubating_samples=expected_incubating_samples,
        log_prob_zero_offspring_samples=log_prob_zero_offspring_samples,
        risk_type=risk_type,
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


def _nbinom_log_prob_zero(*, mean, dispersion):
    # Log probability of zero under NB(mean, dispersion), with the same parameterisation as
    # pm.NegativeBinomial(mu=mean, alpha=dispersion). For SSE-type counts, pass the
    # per-offspring mean and dispersion and scale the result, rather than passing totals that
    # are zero when there is no remaining infectiousness (0/0 below)
    return -dispersion * np.log1p(mean / dispersion)


def _combine_so_risk_samples(
    *,
    log_prob_zero_remaining_samples,
    expected_incubating_samples,
    log_prob_zero_offspring_samples,
    risk_type,
):
    # Risk in an SO model from the log probability of no direct infections after the final
    # day by cases up to it, the (Poisson) mean number still incubating at the end of it,
    # and the log probability (log q_0) that an incubating individual has no offspring.
    # Each incubating individual gives a later case with certainty, and a transmission with
    # probability 1 - q_0
    if risk_type == "cases":
        log_prob_zero_incubating_samples = -expected_incubating_samples
    else:
        log_prob_zero_incubating_samples = (
            np.expm1(log_prob_zero_offspring_samples) * expected_incubating_samples
        )
    return -np.expm1(log_prob_zero_remaining_samples + log_prob_zero_incubating_samples)
