# Note that AI tools were used to generate tests
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.stats

from endoutbreakhet.plotting import (
    _month_start_x_axis,
    plot_densities,
    plot_model_probabilities,
    plot_risk_incidence_overlay,
    save_figure,
)
from endoutbreakhet.utils import parse_lognormal_prior

# Days since this date that fall on month starts: 1 May (27), 1 Jun (58), 1 Jul (88), ...,
# 1 Dec (241), 1 Jan 2019 (272), 1 Feb 2019 (303)
_FIRST_CASE_DATE = pd.Timestamp("2018-04-04")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def posterior_df_by_model():
    rng = np.random.default_rng(1)
    n_samples = 1000
    rep_no_samples = {
        "rep_no_before": rng.lognormal(mean=0.5, sigma=0.2, size=n_samples),
        "rep_no_after": rng.lognormal(mean=-0.5, sigma=0.2, size=n_samples),
    }
    return {
        "poisson": pd.DataFrame(rep_no_samples),
        "sse": pd.DataFrame(
            rep_no_samples
            | {"dispersion": rng.lognormal(mean=-1.5, sigma=0.5, size=n_samples)}
        ),
    }


@pytest.fixture
def risk_df_by_model():
    days = np.arange(40, 80)
    risk_cases_vec = np.linspace(0.9, 0.1, len(days))
    return {
        # Naive model risks of cases and transmission coincide
        "sse": pd.DataFrame(
            {
                "day": days,
                "risk_cases": risk_cases_vec,
                "risk_transmission": risk_cases_vec,
            }
        ),
        "sse_so": pd.DataFrame(
            {
                "day": days,
                "risk_cases": np.linspace(0.95, 0.2, len(days)),
                "risk_transmission": np.linspace(0.8, 0.05, len(days)),
            }
        ),
    }


@pytest.mark.parametrize(
    ("param_name", "log_x", "expected_labels"),
    [
        ("rep_no_before", False, ["Poisson", "SSE", "prior"]),
        # The Poisson model has no dispersion, so is left out
        ("dispersion", True, ["SSE", "prior"]),
    ],
)
def test_plot_densities(
    *, posterior_df_by_model, toy_inputs, param_name, log_x, expected_labels
):
    priors = toy_inputs["priors"]
    fig = plot_densities(
        posterior_df_by_model=posterior_df_by_model,
        priors=priors,
        param_name=param_name,
        log_x=log_x,
    )
    fig.canvas.draw()
    (ax,) = fig.axes
    lines = ax.get_lines()
    assert [line.get_label() for line in lines] == expected_labels
    assert [text.get_text() for text in ax.get_legend().get_texts()] == expected_labels
    assert ax.get_xscale() == ("log" if log_x else "linear")
    assert ax.get_xlabel().startswith("$") and ax.get_xlabel().endswith("$")

    # Areas under the curves on the plotted axis (per unit log10(x) on a log axis) are
    # probabilities: the prior's is its mass in the plotted range, and each posterior's is
    # close to 1, as the range covers its central 98%
    x_vec = lines[-1].get_xdata()
    axis_x_vec = np.log10(x_vec) if log_x else x_vec
    lognormal_params = parse_lognormal_prior(name=param_name, priors=priors)
    prior_dist = scipy.stats.lognorm(
        s=lognormal_params["sigma"], scale=np.exp(lognormal_params["mu"])
    )
    assert np.trapezoid(lines[-1].get_ydata(), axis_x_vec) == pytest.approx(
        prior_dist.cdf(x_vec[-1]) - prior_dist.cdf(x_vec[0]), rel=1e-3
    )
    for line in lines[:-1]:
        assert 0.9 < np.trapezoid(line.get_ydata(), axis_x_vec) < 1.01


def test_plot_densities_without_legend(*, posterior_df_by_model, toy_inputs):
    fig = plot_densities(
        posterior_df_by_model=posterior_df_by_model,
        priors=toy_inputs["priors"],
        param_name="rep_no_after",
        legend=False,
    )
    assert fig.axes[0].get_legend() is None


def test_plot_model_probabilities():
    fig = plot_model_probabilities(
        model_probability_by_model={"sse": 0.25, "ssi_so": 0.75}
    )
    fig.canvas.draw()
    (ax,) = fig.axes
    wedges = ax.patches
    np.testing.assert_allclose(
        [wedge.theta2 - wedge.theta1 for wedge in wedges], [90, 270]
    )
    assert [wedge.get_facecolor() for wedge in wedges] == [
        matplotlib.colors.to_rgba("tab:blue"),
        matplotlib.colors.to_rgba("tab:orange"),
    ]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "SSE (0.25)",
        "SSI-SO (0.75)",
    ]


@pytest.mark.parametrize(
    ("risk_types", "expected_lines", "expected_ylabel"),
    [
        (
            ("cases",),
            [("sse", "cases", "SSE"), ("sse_so", "cases", "SSE-SO")],
            "Risk of additional cases",
        ),
        (
            ("cases", "transmission"),
            [
                ("sse", "cases", "SSE"),
                ("sse_so", "cases", "SSE-SO (cases)"),
                ("sse_so", "transmission", "SSE-SO (transmission)"),
            ],
            "Risk",
        ),
    ],
)
def test_plot_risk_incidence_overlay(
    *, risk_df_by_model, risk_types, expected_lines, expected_ylabel
):
    incidence_vec = np.arange(60) % 4
    vline_days = [34, 70]
    fig = plot_risk_incidence_overlay(
        risk_df_by_model=risk_df_by_model,
        incidence_vec=incidence_vec,
        first_case_date=_FIRST_CASE_DATE,
        risk_types=risk_types,
        vline_days=vline_days,
    )
    fig.canvas.draw()
    ax, twin_ax = fig.axes

    *risk_lines, vline_before, vline_after = ax.get_lines()
    assert len(risk_lines) == len(expected_lines)
    for line, (model_name, risk_type, label) in zip(
        risk_lines, expected_lines, strict=True
    ):
        risk_df = risk_df_by_model[model_name]
        np.testing.assert_array_equal(line.get_xdata(), risk_df["day"])
        np.testing.assert_array_equal(line.get_ydata(), risk_df[f"risk_{risk_type}"])
        assert line.get_linestyle() == {"cases": "-", "transmission": "--"}[risk_type]
        assert line.get_label() == label
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        label for _, _, label in expected_lines
    ]
    assert [vline_before.get_xdata()[0], vline_after.get_xdata()[0]] == vline_days
    assert ax.get_ylabel() == expected_ylabel

    # Incidence bars in the same days-since-first-case coordinates as the risks, with the
    # x axis covering the months of the risk days only
    bars = twin_ax.patches
    np.testing.assert_allclose(
        [bar.get_x() + bar.get_width() / 2 for bar in bars], np.arange(60)
    )
    np.testing.assert_array_equal([bar.get_height() for bar in bars], incidence_vec)
    assert ax.get_xlim() == (27, 88)


@pytest.mark.parametrize(
    ("plot_days", "interval_months", "expected_ticks", "expected_labels", "xlabel"),
    [
        (
            np.arange(40, 80),
            1,
            [27, 58, 88],
            ["1 May", "1 Jun", "1 Jul"],
            "Date (2018)",
        ),
        (np.arange(40, 80), 2, [27, 58, 88], ["1 May", "", "1 Jul"], "Date (2018)"),
        (
            np.arange(250, 300),
            1,
            [241, 272, 303],
            ["1 Dec 2018", "1 Jan 2019", "1 Feb 2019"],
            "Date",
        ),
    ],
)
def test_month_start_x_axis(
    *, plot_days, interval_months, expected_ticks, expected_labels, xlabel
):
    _, ax = plt.subplots()
    _month_start_x_axis(
        ax=ax,
        day_0_date=_FIRST_CASE_DATE,
        plot_days=plot_days,
        interval_months=interval_months,
    )
    assert list(ax.get_xticks()) == expected_ticks
    assert [label.get_text() for label in ax.get_xticklabels()] == expected_labels
    assert ax.get_xlim() == (expected_ticks[0], expected_ticks[-1])
    assert ax.get_xlabel() == xlabel


def test_save_figure(*, tmp_path):
    fig, _ = plt.subplots()
    paths = [tmp_path / "new_dir" / "fig.svg", tmp_path / "new_dir" / "fig.png"]
    save_figure(fig=fig, paths=paths)
    assert all(path.stat().st_size > 0 for path in paths)
    assert not plt.fignum_exists(fig.number)
