# Note that AI tools were used to generate tests
import types

import numpy as np
import pytest
import scipy.stats
import xarray as xr

from endoutbreakhet import risk

_RISK_TYPES = ["cases", "transmission", "sustained", "reports"]
_BRANCHING_MODEL_NAMES = ["poisson", "sse", "ssi", "poisson_so", "sse_so", "ssi_so"]
# Every model with each risk type it has (DLO risks are of cases and transmission only)
_MODEL_AND_RISK_TYPES = [
    (model_name, risk_type)
    for model_name in _BRANCHING_MODEL_NAMES
    for risk_type in _RISK_TYPES
] + [("dlo", "cases"), ("dlo", "transmission")]


@pytest.fixture
def risk_inputs(*, toy_inputs):
    # Model inputs other than the priors
    return {name: value for name, value in toy_inputs.items() if name != "priors"}


# Last case on day 11 and serial interval lags 1-4, so with complete reporting only the
# day-11 case has remaining infectiousness on day 14 (w_4), and none remains from day 15.
# Day 1 has fewer days than the serial interval support
@pytest.mark.parametrize("reporting_prob", [1.0, 0.5])
@pytest.mark.parametrize("final_day", [1, 5, 11, 14, 15])
@pytest.mark.parametrize(
    ("model_name", "risk_type"),
    [
        (model_name, risk_type)
        for model_name, risk_type in _MODEL_AND_RISK_TYPES
        if not model_name.endswith("_so")
    ],
)
def test_naive_risk_matches_notes(
    *, model_name, risk_type, final_day, reporting_prob, risk_inputs
):
    reported_incidence_vec = risk_inputs["reported_incidence_vec"][: final_day + 1]
    serial_interval = risk_inputs["delay_distributions"]["serial_interval"]
    max_lag = len(serial_interval)
    posterior, total_incidence_vec_samples = _fake_posterior(
        reported_incidence_vec=reported_incidence_vec,
        tost=risk_inputs["delay_distributions"]["tost"],
        reporting_prob=reporting_prob,
    )
    rep_no = posterior["rep_no_before"].values
    dispersion = posterior["dispersion"].values

    # Infectiousness of each day's cases: C_t, or in SSI the latent Y_t
    if model_name == "ssi":
        inf_vec_samples = posterior["cohort_inf"].values
    else:
        inf_vec_samples = total_incidence_vec_samples
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
    else:
        # Future cases have Poisson(R) offspring in the Poisson model, else NB(R, k)
        remaining_pgf_arg, _ = _reference_pgf_args(
            risk_type=risk_type,
            rep_no=rep_no,
            dispersion=None if model_name == "poisson" else dispersion,
            reporting_prob=reporting_prob,
        )
        # N^(rem) is NB(R Lambda, k Lambda) in the SSE model, else Poisson(R Lambda)
        if model_name == "sse":
            pgf_remaining = (1 + rep_no / dispersion * (1 - remaining_pgf_arg)) ** (
                -dispersion * remaining_inf
            )
        else:
            pgf_remaining = np.exp(-rep_no * remaining_inf * (1 - remaining_pgf_arg))
        expected = np.mean(1 - pgf_remaining)

    result = risk.compute_risk(
        model_name=model_name,
        **(risk_inputs | {"reported_incidence_vec": reported_incidence_vec}),
        trace=types.SimpleNamespace(posterior=posterior),
        reporting_prob=reporting_prob,
        risk_type=risk_type,
    )
    assert result == pytest.approx(expected, rel=1e-10, abs=0)


# Last case on day 11 and TOST lags 0-2, so with complete reporting no remaining
# infectiousness from day 13; infections up to day 13 and incubation lags 1-3 leave some
# still incubating on day 15, and none from day 16. switch_day is 6, so day 5 is before
# the switch
@pytest.mark.parametrize("reporting_prob", [1.0, 0.5])
@pytest.mark.parametrize("final_day", [1, 5, 11, 13, 15, 16])
@pytest.mark.parametrize("risk_type", _RISK_TYPES)
@pytest.mark.parametrize("model_name", ["poisson_so", "sse_so", "ssi_so"])
def test_so_risk_matches_notes(
    *, model_name, risk_type, final_day, reporting_prob, risk_inputs
):
    reported_incidence_vec = risk_inputs["reported_incidence_vec"][: final_day + 1]
    n_days = len(reported_incidence_vec)
    switch_day = risk_inputs["switch_day"]
    tost = risk_inputs["delay_distributions"]["tost"]
    inc_period = risk_inputs["delay_distributions"]["inc_period"]
    tost_max = len(tost) - 1
    inc_period_max = len(inc_period)
    posterior, total_incidence_vec_samples = _fake_posterior(
        reported_incidence_vec=reported_incidence_vec,
        tost=tost,
        reporting_prob=reporting_prob,
    )
    rep_no_before = posterior["rep_no_before"].values
    rep_no_after = posterior["rep_no_after"].values
    dispersion = posterior["dispersion"].values

    def tost_sum(*, x_vec, t):
        # Notes' sum over s of x_{t-s} tost_s, counting only days 0, ..., final_day
        return sum(
            x_vec[..., t - s] * tost[s]
            for s in range(tost_max + 1)
            if 0 <= t - s <= final_day
        )

    # Infectiousness of each day's cases: C_t, or in SSI-SO the latent Y_t
    if model_name == "ssi_so":
        cohort_inf_vec_samples = posterior["cohort_inf"].values
    else:
        cohort_inf_vec_samples = total_incidence_vec_samples
    # Total infectiousness on each day: lambda_t, or in SSE-SO the latent Z_t
    if model_name == "sse_so":
        total_inf_vec_samples = posterior["total_inf"].values
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

    # Future cases have Poisson(R) offspring in the Poisson-SO model, else NB(R, k)
    remaining_pgf_arg, incubating_pgf_arg = _reference_pgf_args(
        risk_type=risk_type,
        rep_no=rep_no_before,
        dispersion=None if model_name == "poisson_so" else dispersion,
        reporting_prob=reporting_prob,
    )
    # N^(rem) is NB(R Lambda, k Lambda) in the SSE-SO model, else Poisson(R Lambda), and
    # N^(inc) is Poisson
    if model_name == "sse_so":
        pgf_remaining = (1 + rep_no_before / dispersion * (1 - remaining_pgf_arg)) ** (
            -dispersion * remaining_inf
        )
    else:
        pgf_remaining = np.exp(-rep_no_before * remaining_inf * (1 - remaining_pgf_arg))
    pgf_incubating = np.exp(-expected_incubating * (1 - incubating_pgf_arg))
    expected = np.mean(1 - pgf_remaining * pgf_incubating)

    result = risk.compute_risk(
        model_name=model_name,
        **(risk_inputs | {"reported_incidence_vec": reported_incidence_vec}),
        trace=types.SimpleNamespace(posterior=posterior),
        reporting_prob=reporting_prob,
        risk_type=risk_type,
    )
    assert result == pytest.approx(expected, rel=1e-10, abs=0)


@pytest.mark.parametrize("reporting_prob", [1.0, 0.5])
@pytest.mark.parametrize("model_name", _BRANCHING_MODEL_NAMES)
def test_risk_orderings(*, model_name, reporting_prob, risk_inputs):
    # As in the notes, RST <= RAT <= RAC and RST <= RAR <= RAC, with RAT = RAC in naive
    # models and RAR = RAC under complete reporting
    reported_incidence_vec = risk_inputs["reported_incidence_vec"][:12]
    posterior, _ = _fake_posterior(
        reported_incidence_vec=reported_incidence_vec,
        tost=risk_inputs["delay_distributions"]["tost"],
        reporting_prob=reporting_prob,
    )
    risks = {
        risk_type: risk.compute_risk(
            model_name=model_name,
            **(risk_inputs | {"reported_incidence_vec": reported_incidence_vec}),
            trace=types.SimpleNamespace(posterior=posterior),
            reporting_prob=reporting_prob,
            risk_type=risk_type,
        )
        for risk_type in _RISK_TYPES
    }
    assert 0 < risks["sustained"] <= risks["transmission"] <= risks["cases"] < 1
    assert risks["sustained"] <= risks["reports"] <= risks["cases"]
    if not model_name.endswith("_so"):
        assert risks["transmission"] == risks["cases"]
    if reporting_prob == 1:
        assert risks["reports"] == risks["cases"]


@pytest.mark.parametrize("risk_type", ["cases", "transmission"])
def test_sse_so_risk_before_first_infectious_day(*, risk_type):
    # The seed has all its infectiousness ahead of it; no infections have occurred yet.
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), np.ones((2, 3)))
            for name in ("rep_no_before", "rep_no_after", "dispersion")
        }
        | {"total_inf": (("chain", "draw", "day"), np.zeros((2, 3, 1)))}
    )
    result = risk.compute_risk(
        model_name="sse_so",
        reported_incidence_vec=np.array([1]),
        switch_day=0,
        delay_distributions={
            "tost": np.array([0.0, 1.0]),
            "inc_period": np.array([1.0]),
        },
        trace=types.SimpleNamespace(posterior=posterior),
        risk_type=risk_type,
    )
    assert result == pytest.approx(0.5)


@pytest.mark.parametrize("reporting_prob", [1.0, 0.5])
def test_reporting_prob_mismatch_raises(*, reporting_prob, risk_inputs):
    # Total incidence is in the trace if and only if reporting is incomplete
    n_days = len(risk_inputs["reported_incidence_vec"])
    posterior = xr.Dataset(
        {"rep_no_before": (("chain", "draw"), np.ones((1, 1)))}
        | (
            {
                "total_incidence": (
                    ("chain", "draw", "day"),
                    np.zeros((1, 1, n_days), int),
                )
            }
            if reporting_prob == 1
            else {}
        )
    )
    with pytest.raises(ValueError, match="doesn't match reporting probability"):
        risk.compute_risk(
            model_name="poisson",
            **risk_inputs,
            trace=types.SimpleNamespace(posterior=posterior),
            reporting_prob=reporting_prob,
        )


@pytest.mark.parametrize("reporting_prob", [0.0, 1.5, np.nan])
def test_invalid_reporting_prob_raises(*, reporting_prob, risk_inputs):
    with pytest.raises(ValueError, match="Reporting probability must be in"):
        risk.compute_risk(
            model_name="poisson",
            **risk_inputs,
            trace=None,
            reporting_prob=reporting_prob,
        )


def test_total_incidence_days_mismatch_raises(*, risk_inputs):
    # A trace fitted to more days than the reported incidence passed
    posterior, _ = _fake_posterior(
        reported_incidence_vec=risk_inputs["reported_incidence_vec"],
        tost=risk_inputs["delay_distributions"]["tost"],
        reporting_prob=0.5,
    )
    with pytest.raises(ValueError, match="covers 18 days"):
        risk.compute_risk(
            model_name="poisson",
            **(
                risk_inputs
                | {"reported_incidence_vec": risk_inputs["reported_incidence_vec"][:12]}
            ),
            trace=types.SimpleNamespace(posterior=posterior),
            reporting_prob=0.5,
        )


@pytest.mark.parametrize("risk_type", ["sustained", "reports"])
def test_dlo_sustained_and_reports_raise(*, risk_type, risk_inputs):
    with pytest.raises(NotImplementedError, match="DLO"):
        risk.compute_risk(
            model_name="dlo",
            **risk_inputs,
            trace=types.SimpleNamespace(posterior=xr.Dataset()),
            risk_type=risk_type,
        )


@pytest.mark.parametrize(("model_name", "risk_type"), _MODEL_AND_RISK_TYPES)
def test_zero_incidence_has_zero_risk(*, model_name, risk_type, risk_inputs):
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), np.ones((2, 3)))
            for name in ("rep_no_before", "rep_no_after", "dispersion")
        }
        | {
            name: (
                ("chain", "draw", "day"),
                np.zeros((2, 3, len(risk_inputs["reported_incidence_vec"]))),
            )
            for name in ("cohort_inf", "total_inf")
        }
    )
    result = risk.compute_risk(
        model_name=model_name,
        **(
            risk_inputs
            | {
                "reported_incidence_vec": np.zeros_like(
                    risk_inputs["reported_incidence_vec"]
                )
            }
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
            name: (("chain", "draw", "day"), np.full((2, 3, 1), 1.5))
            for name in ("cohort_inf", "total_inf")
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
        reported_incidence_vec=np.array([2]),
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


@pytest.mark.parametrize("z", [0.0, 0.37, 1.0])
def test_log_pgf(*, z):
    mean = np.array([0.0, 0.3, 1.5, 20.0])
    dispersion = np.array([0.05, 1.0, 0.18, 1e6])
    # PGFs as truncated sums of pmf_j z^j
    j = np.arange(2000)[:, None]
    np.testing.assert_allclose(
        np.exp(risk._log_pgf(mean=mean, dispersion=None, z=z)),
        (scipy.stats.poisson.pmf(j, mean) * z**j).sum(axis=0),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.exp(risk._log_pgf(mean=mean, dispersion=dispersion, z=z)),
        (
            scipy.stats.nbinom.pmf(j, dispersion, dispersion / (dispersion + mean))
            * z**j
        ).sum(axis=0),
        rtol=1e-10,
    )


# Poisson offspring with R = 50 and reporting_prob near 1 put the solution within rounding
# of the one-generation probability
@pytest.mark.parametrize("reporting_prob", [0.0, 0.3, 1 - 1e-8, 1.0])
@pytest.mark.parametrize("dispersion", [None, 0.1, 2.0])
def test_lineage_unreported_extinction_prob(*, dispersion, reporting_prob):
    rep_no = np.array([0.5, 1.5, 4.0, 50.0])
    dispersion_vec = None if dispersion is None else np.full(len(rep_no), dispersion)
    result = risk._lineage_unreported_extinction_prob(
        rep_no_samples=rep_no,
        dispersion_samples=dispersion_vec,
        reporting_prob=reporting_prob,
    )
    expected = _reference_lineage_unreported_extinction_prob(
        rep_no=rep_no, dispersion=dispersion_vec, reporting_prob=reporting_prob
    )

    # A root of x = (1 - rho) G^(off)(x)
    np.testing.assert_allclose(
        result,
        (1 - reporting_prob)
        * _offspring_pgf(rep_no=rep_no, dispersion=dispersion_vec, z=result),
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12)
    if reporting_prob == 0:
        # Extinction is certain if and only if R <= 1
        np.testing.assert_array_equal(result == 1, rep_no <= 1)
    elif reporting_prob == 1:
        np.testing.assert_array_equal(result, 0)


@pytest.mark.parametrize("reporting_prob", [0.0, 1e-12, 0.3, 1 - 1e-15])
def test_lineage_unreported_extinction_prob_geometric(*, reporting_prob):
    # Geometric offspring (dispersion 1) have G^(off)(x) = 1 / (1 + R (1 - x)), so the
    # solution is the smallest root of R x^2 - (1 + R) x + (1 - rho) = 0, written to avoid
    # cancellation near R = 1 and at both ends of rho
    rep_no = np.array([0.5, 1 - 1e-8, 1.0, 1 + 1e-8, 1.5, 50.0])
    result = risk._lineage_unreported_extinction_prob(
        rep_no_samples=rep_no,
        dispersion_samples=np.ones(len(rep_no)),
        reporting_prob=reporting_prob,
    )
    expected = (
        2
        * (1 - reporting_prob)
        / (1 + rep_no + np.sqrt((1 - rep_no) ** 2 + 4 * rep_no * reporting_prob))
    )
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=0)


@pytest.mark.parametrize("reporting_prob", [0.0, 0.3])
def test_lineage_unreported_extinction_prob_poisson_limit(*, reporting_prob):
    # NB(R, k) offspring tend to Poisson(R) as k grows, including where k is too large for
    # the NB PGF's power form (so the reference takes Poisson offspring)
    rep_no = np.array([0.5, 1.5, 4.0])
    result = risk._lineage_unreported_extinction_prob(
        rep_no_samples=rep_no,
        dispersion_samples=np.full(len(rep_no), 1e16),
        reporting_prob=reporting_prob,
    )
    expected = _reference_lineage_unreported_extinction_prob(
        rep_no=rep_no, dispersion=None, reporting_prob=reporting_prob
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12)


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


def _fake_posterior(*, reported_incidence_vec, tost, reporting_prob):
    # Fake posterior over 2 chains of 50 draws (more samples than days, so mixing up the
    # sample and day axes fails), with unreported cases on every day after day 0 if
    # reporting_prob < 1. The gamma latents are zero where their mean is, as in the model.
    # Also returns the total incidence samples, of shape (chains, draws, days)
    n_chains, n_draws = 2, 50
    n_days = len(reported_incidence_vec)
    rng = np.random.default_rng(0)
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), rng.lognormal(size=(n_chains, n_draws)))
            for name in ["rep_no_before", "rep_no_after", "dispersion"]
        }
    )
    total_incidence_vec_samples = np.broadcast_to(
        reported_incidence_vec, (n_chains, n_draws, n_days)
    )
    if reporting_prob < 1:
        unreported_incidence_vec_samples = rng.poisson(
            1.0, size=(n_chains, n_draws, n_days)
        )
        unreported_incidence_vec_samples[..., 0] = 0
        total_incidence_vec_samples = (
            total_incidence_vec_samples + unreported_incidence_vec_samples
        )
        posterior["total_incidence"] = (
            ("chain", "draw", "day"),
            total_incidence_vec_samples,
        )
    expected_total_inf_vec_samples = np.stack(
        [
            sum(
                total_incidence_vec_samples[..., t - s] * tost[s]
                for s in range(len(tost))
                if t - s >= 0
            )
            for t in range(n_days)
        ],
        axis=-1,
    )
    for name, mean_vec_samples in [
        ("cohort_inf", total_incidence_vec_samples),
        ("total_inf", expected_total_inf_vec_samples),
    ]:
        posterior[name] = (
            ("chain", "draw", "day"),
            rng.gamma(shape=1.0, size=(n_chains, n_draws, n_days))
            * (mean_vec_samples > 0),
        )
    return posterior, total_incidence_vec_samples


def _reference_pgf_args(*, risk_type, rep_no, dispersion, reporting_prob):
    # Notes' arguments (z, z') of G^(rem) and G^(inc) for each risk type, for offspring
    # Poisson(R) (or NB(R, k), given a dispersion)
    if risk_type == "cases":
        return 0.0, 0.0
    if risk_type == "transmission":
        return 0.0, _offspring_pgf(rep_no=rep_no, dispersion=dispersion, z=0.0)
    lineage_prob = _reference_lineage_unreported_extinction_prob(
        rep_no=rep_no,
        dispersion=dispersion,
        reporting_prob=0.0 if risk_type == "sustained" else reporting_prob,
    )
    return lineage_prob, lineage_prob


def _reference_lineage_unreported_extinction_prob(
    *, rep_no, dispersion, reporting_prob
):
    # Notes' probability (q, or q_rho) that a lineage dies out with no reported case, as
    # the limit of the probabilities x_n that it does so within n generations, with
    # x_0 = 0 and x_n = (1 - rho) G^(off)(x_{n-1})
    x = np.zeros_like(rep_no)
    for _ in range(20_000):
        x_next = (1 - reporting_prob) * _offspring_pgf(
            rep_no=rep_no, dispersion=dispersion, z=x
        )
        if np.array_equal(x_next, x):
            return x
        x = x_next
    raise RuntimeError("Fixed-point iteration didn't converge")


def _offspring_pgf(*, rep_no, dispersion, z):
    # Notes' PGF of Poisson(R) offspring, or NB(R, k) given a dispersion
    if dispersion is None:
        return np.exp(-rep_no * (1 - z))
    return (1 + rep_no / dispersion * (1 - z)) ** (-dispersion)
