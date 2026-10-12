# Note that AI tools were used to generate tests
import types

import numpy as np
import pytest
import scipy.stats
import xarray as xr

from endoutbreakhet import risk


@pytest.fixture
def risk_inputs(*, toy_inputs):
    # Model inputs other than the priors
    return {name: value for name, value in toy_inputs.items() if name != "priors"}


# Last case on day 11 and serial interval lags 1-4, so only the day-11 case has remaining
# infectiousness on day 14 (w_4), and none remains from day 15. Day 1 has fewer days than
# the serial interval support
@pytest.mark.parametrize("final_day", [1, 5, 11, 14, 15])
@pytest.mark.parametrize("model_name", ["poisson", "dlo", "sse", "ssi"])
def test_naive_risk_matches_notes(*, model_name, final_day, risk_inputs):
    incidence_vec = risk_inputs["incidence_vec"][: final_day + 1]
    n_days = len(incidence_vec)
    serial_interval = risk_inputs["delay_distributions"]["serial_interval"]
    max_lag = len(serial_interval)
    nonzero_incidence_idx = incidence_vec > 0
    n_chains, n_draws = 2, 50
    rng = np.random.default_rng(0)
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), rng.lognormal(size=(n_chains, n_draws)))
            for name in ["rep_no_before", "rep_no_after", "dispersion"]
        }
        | {
            "cohort_inf_case_days": (
                ("chain", "draw", "cohort_inf_case_days_dim_0"),
                rng.gamma(
                    shape=1.0, size=(n_chains, n_draws, nonzero_incidence_idx.sum())
                ),
            )
        }
    )
    rep_no = posterior["rep_no_before"].values
    dispersion = posterior["dispersion"].values

    # Infectiousness of each day's cases: C_t, or in SSI the latent Y_t (zero on days
    # without cases)
    if model_name == "ssi":
        inf_vec_samples = np.zeros((n_chains, n_draws, n_days))
        inf_vec_samples[..., nonzero_incidence_idx] = posterior[
            "cohort_inf_case_days"
        ].values
    else:
        inf_vec_samples = np.broadcast_to(incidence_vec, (n_chains, n_draws, n_days))
    # Notes' infectiousness on each future day tau from cases up to final_day, and its sum
    remaining_inf_by_day = [
        sum(
            inf_vec_samples[..., tau - s] * serial_interval[s - 1]
            for s in range(tau - final_day, max_lag + 1)
            if tau - s >= 0
        )
        for tau in range(final_day + 1, final_day + max_lag + 1)
    ]
    remaining_inf = sum(remaining_inf_by_day)
    if model_name == "dlo":
        expected = np.mean(
            1
            - np.prod(
                [
                    (dispersion / (dispersion + rep_no * remaining_inf_day))
                    ** dispersion
                    for remaining_inf_day in remaining_inf_by_day
                ],
                axis=0,
            )
        )
    elif model_name == "sse":
        expected = np.mean(
            1 - (dispersion / (dispersion + rep_no)) ** (dispersion * remaining_inf)
        )
    else:
        expected = np.mean(1 - np.exp(-rep_no * remaining_inf))

    results = {
        risk_type: risk.compute_risk(
            model_name=model_name,
            **(risk_inputs | {"incidence_vec": incidence_vec}),
            trace=types.SimpleNamespace(posterior=posterior),
            risk_type=risk_type,
        )
        for risk_type in ("cases", "transmission")
    }
    for result in results.values():
        assert result == pytest.approx(expected, rel=1e-10, abs=0)
    assert results["cases"] == results["transmission"]


# Last case on day 11 and TOST lags 0-2, so no remaining infectiousness from day 13;
# infections up to day 13 and incubation lags 1-3 leave some still incubating on day 15,
# and none from day 16. switch_day is 6, so day 5 is before the switch
@pytest.mark.parametrize("final_day", [1, 5, 11, 13, 15, 16])
@pytest.mark.parametrize("risk_type", ["cases", "transmission"])
@pytest.mark.parametrize("model_name", ["poisson_so", "sse_so", "ssi_so"])
def test_so_risk_matches_notes(*, model_name, risk_type, final_day, risk_inputs):
    incidence_vec = risk_inputs["incidence_vec"][: final_day + 1]
    n_days = len(incidence_vec)
    switch_day = risk_inputs["switch_day"]
    tost = risk_inputs["delay_distributions"]["tost"]
    inc_period = risk_inputs["delay_distributions"]["inc_period"]
    tost_max = len(tost) - 1
    inc_period_max = len(inc_period)

    def tost_sum(*, x_vec, t):
        # Notes' sum over s of x_{t-s} tost_s, counting only days 0, ..., final_day
        return sum(
            x_vec[..., t - s] * tost[s]
            for s in range(tost_max + 1)
            if 0 <= t - s <= final_day
        )

    expected_total_inf_vec = np.array(
        [tost_sum(x_vec=incidence_vec, t=t) for t in range(n_days)]
    )
    nonzero_incidence_idx = incidence_vec > 0
    nonzero_total_inf_idx = expected_total_inf_vec > 0
    n_chains, n_draws = 2, 50
    rng = np.random.default_rng(0)
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), rng.lognormal(size=(n_chains, n_draws)))
            for name in ["rep_no_before", "rep_no_after", "dispersion"]
        }
        | {
            name: (
                ("chain", "draw", f"{name}_dim_0"),
                rng.gamma(shape=1.0, size=(n_chains, n_draws, nonzero_idx.sum())),
            )
            for name, nonzero_idx in [
                ("cohort_inf_case_days", nonzero_incidence_idx),
                ("total_inf_nonzero_days", nonzero_total_inf_idx),
            ]
        }
    )
    rep_no_before = posterior["rep_no_before"].values
    rep_no_after = posterior["rep_no_after"].values
    dispersion = posterior["dispersion"].values

    # Infectiousness of each day's cases: C_t, or in SSI-SO the latent Y_t (zero on days
    # without cases)
    if model_name == "ssi_so":
        cohort_inf_vec_samples = np.zeros((n_chains, n_draws, n_days))
        cohort_inf_vec_samples[..., nonzero_incidence_idx] = posterior[
            "cohort_inf_case_days"
        ].values
    else:
        cohort_inf_vec_samples = np.broadcast_to(
            incidence_vec, (n_chains, n_draws, n_days)
        )
    # Total infectiousness on each day: lambda_t, or in SSE-SO the latent Z_t (zero on days
    # with lambda_t = 0)
    if model_name == "sse_so":
        total_inf_vec_samples = np.zeros((n_chains, n_draws, n_days))
        total_inf_vec_samples[..., nonzero_total_inf_idx] = posterior[
            "total_inf_nonzero_days"
        ].values
    else:
        total_inf_vec_samples = np.stack(
            [tost_sum(x_vec=cohort_inf_vec_samples, t=t) for t in range(n_days)],
            axis=-1,
        )
    # Notes' infection intensity mu_t on each day up to final_day
    expected_infection_incidence_local = [
        (rep_no_before if t < switch_day else rep_no_after)
        * total_inf_vec_samples[..., t]
        for t in range(n_days)
    ]
    # Infectiousness after final_day from cases up to final_day, and expected number
    # infected up to final_day with onset after it
    remaining_inf = sum(
        tost_sum(x_vec=cohort_inf_vec_samples, t=tau)
        for tau in range(final_day + 1, final_day + tost_max + 1)
    )
    expected_incubating = sum(
        expected_infection_incidence_local[t] * inc_period[s - 1]
        for t in range(n_days)
        for s in range(final_day + 1 - t, inc_period_max + 1)
    )

    if model_name == "sse_so":
        prob_zero_remaining = (dispersion / (dispersion + rep_no_before)) ** (
            dispersion * remaining_inf
        )
    else:
        prob_zero_remaining = np.exp(-rep_no_before * remaining_inf)
    if model_name == "poisson_so":
        prob_zero_offspring = np.exp(-rep_no_before)
    else:
        prob_zero_offspring = (dispersion / (dispersion + rep_no_before)) ** dispersion
    # Each incubating individual gives a later case with certainty, and a transmission with
    # probability 1 - q_0
    incubating_event_prob = 1 if risk_type == "cases" else 1 - prob_zero_offspring
    expected = np.mean(
        1 - prob_zero_remaining * np.exp(-incubating_event_prob * expected_incubating)
    )

    result = risk.compute_risk(
        model_name=model_name,
        **(risk_inputs | {"incidence_vec": incidence_vec}),
        trace=types.SimpleNamespace(posterior=posterior),
        risk_type=risk_type,
    )
    assert result == pytest.approx(expected, rel=1e-10, abs=0)


@pytest.mark.parametrize("risk_type", ["cases", "transmission"])
def test_sse_so_risk_before_first_infectious_day(*, risk_type):
    # The seed has all its infectiousness ahead of it; no infections have occurred yet.
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), np.ones((2, 3)))
            for name in ("rep_no_before", "rep_no_after", "dispersion")
        }
        | {
            "total_inf_nonzero_days": (
                ("chain", "draw", "total_inf_nonzero_days_dim_0"),
                np.empty((2, 3, 0)),
            )
        }
    )
    result = risk.compute_risk(
        model_name="sse_so",
        incidence_vec=np.array([1]),
        switch_day=0,
        delay_distributions={
            "tost": np.array([0.0, 1.0]),
            "inc_period": np.array([1.0]),
        },
        trace=types.SimpleNamespace(posterior=posterior),
        risk_type=risk_type,
    )
    assert result == pytest.approx(0.5)


@pytest.mark.parametrize("risk_type", ["cases", "transmission"])
def test_zero_incidence_has_zero_risk(*, model_name, risk_type, risk_inputs):
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), np.ones((2, 3)))
            for name in ("rep_no_before", "rep_no_after", "dispersion")
        }
        | {
            name: (("chain", "draw", f"{name}_dim_0"), np.empty((2, 3, 0)))
            for name in ("cohort_inf_case_days", "total_inf_nonzero_days")
        }
    )
    result = risk.compute_risk(
        model_name=model_name,
        **(
            risk_inputs | {"incidence_vec": np.zeros_like(risk_inputs["incidence_vec"])}
        ),
        trace=types.SimpleNamespace(posterior=posterior),
        risk_type=risk_type,
    )
    assert result == 0.0


@pytest.mark.parametrize("model_name", ["poisson_so", "sse_so", "ssi_so"])
@pytest.mark.parametrize("risk_type", ["cases", "transmission"])
def test_so_risk_with_only_lag_zero_tost(*, model_name, risk_type):
    rep_no_before, rep_no_after, dispersion = 0.7, 0.4, 0.2
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), np.full((2, 3), value))
            for name, value in (
                ("rep_no_before", rep_no_before),
                ("rep_no_after", rep_no_after),
                ("dispersion", dispersion),
            )
        }
        | {
            name: (("chain", "draw", f"{name}_dim_0"), np.full((2, 3, 1), 1.5))
            for name in ("cohort_inf_case_days", "total_inf_nonzero_days")
        }
    )
    # All transmission by the two seeds occurs today, under rep_no_after. All resulting
    # infections are still incubating, with future offspring governed by rep_no_before.
    expected_incubating = rep_no_after * (2 if model_name == "poisson_so" else 1.5)
    prob_zero_offspring = (
        np.exp(-rep_no_before)
        if model_name == "poisson_so"
        else (dispersion / (dispersion + rep_no_before)) ** dispersion
    )
    incubating_event_prob = 1 if risk_type == "cases" else 1 - prob_zero_offspring
    expected = 1 - np.exp(-incubating_event_prob * expected_incubating)

    result = risk.compute_risk(
        model_name=model_name,
        incidence_vec=np.array([2]),
        switch_day=0,
        delay_distributions={"tost": np.array([1.0]), "inc_period": np.array([1.0])},
        trace=types.SimpleNamespace(posterior=posterior),
        risk_type=risk_type,
    )
    assert result == pytest.approx(expected, rel=1e-10, abs=0)


@pytest.mark.parametrize("first_lag", [0, 1])
@pytest.mark.parametrize("n_days", [7, 3])  # n_days=3 is shorter than the support
def test_survival_weight_vec(*, first_lag, n_days):
    dist_vec = np.array([0.1, 0.2, 0.3, 0.4])
    max_lag = first_lag + len(dist_vec) - 1
    weight_vec = risk._survival_weight_vec(
        dist_vec=dist_vec, first_lag=first_lag, n_days=n_days
    )
    assert weight_vec.shape == (n_days,)
    for day in range(n_days):
        age = n_days - day
        cdf = sum(dist_vec[i] for i in range(len(dist_vec)) if first_lag + i <= age - 1)
        assert weight_vec[day] == pytest.approx(1 - cdf)
        if age > max_lag:
            assert weight_vec[day] == 0


def test_nbinom_log_prob_zero():
    mean = np.array([0.0, 0.3, 1.5, 20.0])
    dispersion = np.array([0.05, 1.0, 0.18, 1e6])
    np.testing.assert_allclose(
        risk._nbinom_log_prob_zero(mean=mean, dispersion=dispersion),
        scipy.stats.nbinom.logpmf(0, n=dispersion, p=dispersion / (dispersion + mean)),
        rtol=1e-8,
    )


@pytest.mark.parametrize("n_dims", [0, 4])
def test_posterior_samples_shapes(*, n_dims):
    rng = np.random.default_rng(0)
    n_chains, n_draws = 2, 3
    posterior = xr.Dataset(
        {
            "scalar": (("chain", "draw"), rng.normal(size=(n_chains, n_draws))),
            "vector": (
                ("chain", "draw", "vector_dim_0"),
                rng.normal(size=(n_chains, n_draws, n_dims)),
            ),
        }
    )
    scalar_samples = risk._posterior_samples(posterior=posterior, name="scalar")
    vector_samples = risk._posterior_samples(posterior=posterior, name="vector")
    assert scalar_samples.shape == (n_chains * n_draws,)
    assert vector_samples.shape == (n_chains * n_draws, n_dims)
    for chain in range(n_chains):
        for draw in range(n_draws):
            sample = chain * n_draws + draw
            assert scalar_samples[sample] == posterior["scalar"].values[chain, draw]
            np.testing.assert_array_equal(
                vector_samples[sample], posterior["vector"].values[chain, draw]
            )


def test_unknown_model_raises(*, risk_inputs):
    with pytest.raises(ValueError, match="Unknown model name"):
        risk.compute_risk(model_name="nonexistent", **risk_inputs, trace=None)


def test_unknown_risk_type_raises(*, risk_inputs):
    with pytest.raises(ValueError, match="Unknown risk type"):
        risk.compute_risk(
            model_name="poisson", **risk_inputs, trace=None, risk_type="nonexistent"
        )
