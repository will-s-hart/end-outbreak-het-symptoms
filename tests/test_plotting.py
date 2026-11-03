# Note that AI tools were used to generate tests
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.stats
import seaborn as sns

from endoutbreakhet import DelayDistribution
from endoutbreakhet.plotting import (
    _month_start_x_axis,
    _summarise_unreported_pmf,
    make_panel_figure,
    plot_delay_distribution,
    plot_densities,
    plot_fit_gate,
    plot_incidence,
    plot_model_probabilities,
    plot_model_probabilities_inset,
    plot_risk_incidence_overlay,
    plot_unreported_incidence,
    save_figure,
    set_plot_style,
)
from endoutbreakhet.utils import parse_lognormal_prior

# Days since this date that fall on month starts: 1 May (27), 1 Jun (58), 1 Jul (88), ...,
# 1 Dec (241), 1 Jan 2019 (272), 1 Feb 2019 (303)
_FIRST_CASE_DATE = pd.Timestamp("2018-04-04")


# Seaborn's colorblind palette's blue, orange, green and red
_BLUE, _ORANGE, _GREEN, _RED = sns.color_palette("colorblind")[:4]


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
                "risk_sustained": np.linspace(0.5, 0.0, len(days)),
                "risk_reports": np.linspace(0.85, 0.05, len(days)),
            }
        ),
        "sse_so": pd.DataFrame(
            {
                "day": days,
                "risk_cases": np.linspace(0.95, 0.2, len(days)),
                "risk_transmission": np.linspace(0.8, 0.05, len(days)),
                "risk_sustained": np.linspace(0.6, 0.0, len(days)),
                "risk_reports": np.linspace(0.9, 0.1, len(days)),
            }
        ),
    }


def test_set_plot_style():
    set_plot_style()
    assert plt.rcParams["axes.prop_cycle"].by_key()["color"] == list(
        sns.color_palette("colorblind")
    )
    assert plt.rcParams["svg.fonttype"] == "none"
    assert not plt.rcParams["axes.spines.top"]


def _extent_inches(*, ax):
    # (left, right, bottom, top) of the axes, in inches from the figure's bottom-left
    fig = ax.get_figure(root=True)
    position = ax.get_position()
    return np.array(
        [
            position.x0 * fig.get_figwidth(),
            position.x1 * fig.get_figwidth(),
            position.y0 * fig.get_figheight(),
            position.y1 * fig.get_figheight(),
        ]
    )


@pytest.mark.parametrize(
    ("panel_widths", "spanned_axes_idxs"),
    [
        # A panel filling a row of its own spans the three above it
        ([1, 1, 1, None], (0, 2)),
        # One filling the rest of a row spans the two above it
        ([1, 1, 1, 1, None], (1, 2)),
        ([1, 1, 1, 1, 2], (1, 2)),
    ],
)
def test_make_panel_figure_aligns_spanning_panels(*, panel_widths, spanned_axes_idxs):
    fig, axes = make_panel_figure(panel_widths=panel_widths)
    assert len(axes) == len(panel_widths)
    assert fig.axes == axes
    first_idx, last_idx = spanned_axes_idxs
    left, right, bottom, top = _extent_inches(ax=axes[-1])
    assert left == pytest.approx(_extent_inches(ax=axes[first_idx])[0])
    assert right == pytest.approx(_extent_inches(ax=axes[last_idx])[1])
    # In the second row, as tall as the first
    first_row_extent = _extent_inches(ax=axes[0])
    assert top < first_row_extent[2]
    assert top - bottom == pytest.approx(first_row_extent[3] - first_row_extent[2])
    assert [text.get_text() for text in fig.texts] == [
        f"{chr(ord('A') + idx)}." for idx in range(len(panel_widths))
    ]


def test_make_panel_figure_sizes_by_slots():
    single_fig, (single_ax,) = make_panel_figure(panel_widths=[1])
    single_width, single_height = single_fig.get_size_inches()
    slot_width = make_panel_figure(panel_widths=[1, 1])[0].get_figwidth() - single_width
    # A panel too wide for the rest of the row starts a new one, and the figure is as wide
    # as its widest row
    fig, axes = make_panel_figure(panel_widths=[2, 2])
    np.testing.assert_allclose(
        fig.get_size_inches(), [single_width + slot_width, 2 * single_height]
    )
    # Panels a slot wider by a slot's width, with the same margins
    single_extent = _extent_inches(ax=single_ax)
    np.testing.assert_allclose(
        _extent_inches(ax=axes[1]), single_extent + [0, slot_width, 0, 0]
    )
    # Only labelled with several panels
    assert not single_fig.texts


def test_make_panel_figure_gutters_shift_slots():
    fig, (ax,) = make_panel_figure(panel_widths=[1])
    gutters = {"left": 2, "right": 0.5, "bottom": 1, "top": 0.25}
    gutter_fig, (gutter_ax,) = make_panel_figure(panel_widths=[1], gutters=gutters)
    np.testing.assert_allclose(
        gutter_fig.get_size_inches(), fig.get_size_inches() + [2.5, 1.25]
    )
    np.testing.assert_allclose(
        _extent_inches(ax=gutter_ax), _extent_inches(ax=ax) + [2, 2, 1, 1]
    )


def test_make_panel_figure_sharey():
    fig, axes = make_panel_figure(panel_widths=[1, 1, 1, 1], sharey=True)
    for ax in axes[1:]:
        ax.set_yticks([0, 1], labels=["a", "b"])
    fig.canvas.draw()
    # Tick labels only on the panels starting a row (get_yticklabels leaves out hidden
    # labels)
    for idx, ax in enumerate(axes):
        assert ax.get_shared_y_axes().joined(axes[0], ax)
        assert bool(ax.get_yticklabels()) == (idx in (0, 3))


def test_make_panel_figure_n_columns():
    fig, axes = make_panel_figure(panel_widths=[1, 1, None], n_columns=2)
    # Two slots wide, the last panel filling a row of its own below the first two
    assert fig.get_figwidth() == pytest.approx(
        make_panel_figure(panel_widths=[1, 1])[0].get_figwidth()
    )
    left, right, _, top = _extent_inches(ax=axes[2])
    assert left == pytest.approx(_extent_inches(ax=axes[0])[0])
    assert right == pytest.approx(_extent_inches(ax=axes[1])[1])
    assert top < _extent_inches(ax=axes[0])[2]


def test_make_panel_figure_labels():
    fig, _ = make_panel_figure(panel_widths=[1, 1, 1], labels=["A.", None, "D."])
    assert [text.get_text() for text in fig.texts] == ["A.", "D."]
    with pytest.raises(ValueError, match="2 labels for 3 panels"):
        make_panel_figure(panel_widths=[1, 1, 1], labels=["A.", "B."])


@pytest.mark.parametrize(("n_columns", "width"), [(3, 0), (3, 4), (2, 3)])
def test_make_panel_figure_raises_for_invalid_width(*, n_columns, width):
    with pytest.raises(ValueError, match="Panel widths"):
        make_panel_figure(panel_widths=[1, width], n_columns=n_columns)


@pytest.mark.parametrize(
    ("param_name", "log_x", "expected_labels"),
    [
        ("rep_no_before", False, ["Poisson", "SSE", "Prior"]),
        # The Poisson model has no dispersion, so is left out
        ("dispersion", True, ["SSE", "Prior"]),
    ],
)
def test_plot_densities(
    *, posterior_df_by_model, toy_inputs, param_name, log_x, expected_labels
):
    priors = toy_inputs["priors"]
    fig, (ax,) = make_panel_figure(panel_widths=[1])
    plot_densities(
        ax=ax,
        posterior_df_by_model=posterior_df_by_model,
        priors=priors,
        param_name=param_name,
        log_x=log_x,
    )
    fig.canvas.draw()
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
    _, (ax,) = make_panel_figure(panel_widths=[1])
    plot_densities(
        ax=ax,
        posterior_df_by_model=posterior_df_by_model,
        priors=toy_inputs["priors"],
        param_name="rep_no_after",
        legend=False,
    )
    assert ax.get_legend() is None


def test_plot_model_probabilities():
    fig, (ax,) = make_panel_figure(panel_widths=[1])
    plot_model_probabilities(
        ax=ax, model_probability_by_model={"sse": 0.25, "ssi_so": 0.75}
    )
    fig.canvas.draw()
    (pie_ax,) = ax.child_axes
    wedges = pie_ax.patches
    np.testing.assert_allclose(
        [wedge.theta2 - wedge.theta1 for wedge in wedges], [90, 270]
    )
    assert [wedge.get_facecolor() for wedge in wedges] == [
        matplotlib.colors.to_rgba(_BLUE),
        matplotlib.colors.to_rgba(_ORANGE),
    ]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "SSE (0.25)",
        "SSI-SO (0.75)",
    ]
    assert ax.get_legend().get_title().get_text() == "Model probability"
    assert not ax.get_title()
    # The legend below the pie, clear of it
    assert ax.get_legend().get_window_extent().y1 < pie_ax.get_window_extent().y0


def test_plot_model_probabilities_inset():
    fig, (ax,) = make_panel_figure(panel_widths=[1])
    plot_model_probabilities_inset(
        ax=ax, model_probability_by_model={"sse": 1.0, "ssi_so": 0.0}, label="C."
    )
    fig.canvas.draw()
    # A framed box in the upper right of the panel, labelled as a panel of its own
    (box_ax,) = ax.child_axes
    box_position, position = box_ax.get_position(), ax.get_position()
    assert box_position.x1 < position.x1 and box_position.y1 < position.y1
    assert box_position.x0 + box_position.x1 > position.x0 + position.x1
    assert box_position.y0 + box_position.y1 > position.y0 + position.y1
    assert all(spine.get_visible() for spine in box_ax.spines.values())
    assert [text.get_text() for text in box_ax.texts] == ["C."]
    # The pie over its legend, with edged wedges, so a model of probability 0 still shows
    (pie_ax,) = box_ax.child_axes
    assert all(
        wedge.get_edgecolor() == matplotlib.colors.to_rgba("white")
        for wedge in pie_ax.patches
    )
    assert [text.get_text() for text in box_ax.get_legend().get_texts()] == [
        "SSE (1.00)",
        "SSI-SO (0.00)",
    ]
    assert box_ax.get_legend().get_window_extent().y1 < pie_ax.get_window_extent().y0


def test_plot_fit_gate():
    # Two specs over three days, with the second spec's fit failing on day 11
    statuses = np.array([["pass", "pass", "unassessed"], ["pass", "fail", "pass"]])
    criteria = ("rhat", "ess_bulk", "ess_tail", "divergences", "tree_depth", "risk")
    fig, criterion_axes = make_panel_figure(
        panel_widths=[1] * len(criteria), sharey=True, gutters={"bottom": 1}
    )
    plot_fit_gate(
        axes=criterion_axes,
        status_by_criterion=dict.fromkeys(criteria, statuses),
        fit_status=np.array([["pass", "pass", "pass"], ["pass", "fail", "pass"]]),
        evidence_status=["pass", "fail"],
        spec_names=["sse", "ssi"],
        days=np.array([10, 11, 12]),
        comparison_day=11,
    )
    # Inset in the last panel's right margin
    (evidence_ax,) = criterion_axes[-1].child_axes
    assert evidence_ax.get_position().x0 > criterion_axes[-1].get_position().x1
    colors = {"pass": _GREEN, "fail": _RED, "unassessed": "lightgray"}

    def _cell_colors(*, ax):
        (image,) = ax.get_images()
        return image.to_rgba(image.get_array())

    def _expected_colors(*, statuses):
        return np.array(
            [
                [matplotlib.colors.to_rgba(colors[status]) for status in row]
                for row in statuses
            ]
        )

    assert [ax.get_title() for ax in criterion_axes] == [
        r"$\hat{R}$",
        "Bulk ESS",
        "Tail ESS",
        "Divergences",
        "Max tree depth",
        "Risk",
    ]
    for ax in criterion_axes:
        np.testing.assert_allclose(
            _cell_colors(ax=ax), _expected_colors(statuses=statuses)
        )
        # Days on x, specs on y
        assert ax.get_xlim() == (9.5, 12.5)
        assert ax.get_ylim() == (1.5, -0.5)
        (comparison_line,) = ax.get_lines()
        assert list(comparison_line.get_xdata()) == [11, 11]
    assert [label.get_text() for label in criterion_axes[0].get_yticklabels()] == [
        "sse",
        "ssi",
    ]
    # Only the risk cell on the failed fit is hatched
    for ax in criterion_axes[:-1]:
        assert not ax.patches
    (hatched,) = criterion_axes[-1].patches
    assert hatched.get_hatch() == "////"
    assert hatched.get_xy() == (10.5, 0.5)
    np.testing.assert_allclose(
        _cell_colors(ax=evidence_ax),
        _expected_colors(statuses=[["pass"], ["fail"]]),
    )
    assert [label.get_text() for label in evidence_ax.get_xticklabels()] == ["Evidence"]
    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        "pass",
        "fail",
        "unassessed",
        "risk given a failed fit",
    ]


def test_plot_incidence():
    reported_incidence_vec = np.arange(60) % 4
    _, (ax,) = make_panel_figure(panel_widths=[3])
    plot_incidence(
        ax=ax,
        reported_incidence_vec=reported_incidence_vec,
        first_case_date=_FIRST_CASE_DATE,
        ert_days={"arrival": 20, "departure": 50},
    )
    bars = ax.patches
    np.testing.assert_allclose(
        [bar.get_x() + bar.get_width() / 2 for bar in bars], np.arange(60)
    )
    np.testing.assert_array_equal(
        [bar.get_height() for bar in bars], reported_incidence_vec
    )
    assert [line.get_xdata()[0] for line in ax.get_lines()] == [20, 50]
    assert [text.get_text() for text in ax.texts] == ["ERT arrival", "ERT departure"]
    # The months containing the days since the first case: April to June
    assert ax.get_xlim() == (-3, 88)


def test_plot_delay_distribution():
    delay = DelayDistribution(prob=np.array([0.2, 0.4, 0.3, 0.1]), first_lag=0)
    _, (ax,) = make_panel_figure(panel_widths=[1])
    plot_delay_distribution(ax=ax, delay=delay, delay_name="tost", max_lag=10)
    bars = ax.patches
    np.testing.assert_allclose(
        [bar.get_x() + bar.get_width() / 2 for bar in bars], delay.lags
    )
    np.testing.assert_array_equal([bar.get_height() for bar in bars], delay.prob)
    # Whole bars from lag 0, so the distribution of a delay that can be zero isn't cut
    assert ax.get_xlim() == (-0.5, 10.5)
    assert ax.get_xlabel() == "Time from symptom onset to transmission (days)"
    assert not ax.get_lines()
    assert ax.get_legend() is None


def test_plot_delay_distribution_with_overlay():
    overlay = DelayDistribution(prob=np.array([0.25, 0.45, 0.2, 0.1]), first_lag=1)
    _, (ax,) = make_panel_figure(panel_widths=[1])
    plot_delay_distribution(
        ax=ax,
        delay=DelayDistribution(prob=np.array([0.2, 0.5, 0.3]), first_lag=1),
        delay_name="serial_interval",
        max_lag=10,
        label="Naive models",
        overlay=overlay,
        overlay_label="SO models (implied)",
    )
    (line,) = ax.get_lines()
    np.testing.assert_array_equal(line.get_xdata(), overlay.lags)
    np.testing.assert_array_equal(line.get_ydata(), overlay.prob)
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Naive models",
        "SO models (implied)",
    ]


def test_plot_delay_distribution_colors():
    # Each delay's bars in a colour of its own, translucent so an overlaid line stands out
    colors = []
    for delay_name in ("serial_interval", "inc_period", "tost"):
        _, (ax,) = make_panel_figure(panel_widths=[1])
        plot_delay_distribution(
            ax=ax,
            delay=DelayDistribution(prob=np.array([0.5, 0.5]), first_lag=1),
            delay_name=delay_name,
            max_lag=10,
        )
        (color,) = {bar.get_facecolor() for bar in ax.patches}
        assert 0 < color[3] < 1
        colors.append(color)
    assert len(set(colors)) == len(colors)


def test_reported_incidence_bars_alike_in_every_plot():
    reported_incidence_vec = np.arange(60) % 4
    fig, (incidence_ax, risk_ax, unreported_ax) = make_panel_figure(
        panel_widths=[1, 1, 1]
    )
    plot_incidence(
        ax=incidence_ax,
        reported_incidence_vec=reported_incidence_vec,
        first_case_date=_FIRST_CASE_DATE,
        ert_days={},
    )
    plot_risk_incidence_overlay(
        ax=risk_ax,
        risk_df_by_model={
            "sse": pd.DataFrame({"day": [40, 41], "risk_cases": [0.5, 0.4]})
        },
        reported_incidence_vec=reported_incidence_vec,
        first_case_date=_FIRST_CASE_DATE,
    )
    plot_unreported_incidence(
        ax=unreported_ax,
        unreported_pmf_df_by_model={
            "sse": pd.DataFrame(
                {"day": [1, 2], "n_unreported": [0, 0], "probability": [1.0, 1.0]}
            )
        },
        reported_incidence_vec=reported_incidence_vec,
        first_case_date=_FIRST_CASE_DATE,
        end_date=pd.Timestamp("2018-06-01"),
    )
    # The panels, then the risk panel's twin (the strip is a child of its panel)
    *_, twin_ax = fig.axes
    (strip_ax,) = unreported_ax.child_axes
    bar_colors = [
        {bar.get_facecolor() for bar in ax.patches}
        for ax in (incidence_ax, twin_ax, strip_ax)
    ]
    assert len(bar_colors[0]) == 1
    assert all(colors == bar_colors[0] for colors in bar_colors)


def test_summarise_unreported_pmf():
    # Daily means, and quantiles as the least counts whose cumulative probability reaches
    # each level, even where rounding leaves it just short (on day 2, 44/120 + 73/120 sums
    # to less than 0.975)
    pmf_df = pd.DataFrame(
        {
            "day": [1, 1, 1, 2, 2, 2, 3],
            "n_unreported": [0, 1, 2, 0, 1, 2, 1],
            "probability": [0.5, 0.3, 0.2, 44 / 120, 73 / 120, 3 / 120, 1.0],
        }
    )
    assert np.cumsum([44 / 120, 73 / 120])[-1] < 0.975
    days, mean_vec, quantile_vecs = _summarise_unreported_pmf(
        pmf_df=pmf_df, quantile_levels=(0.025, 0.5, 0.975)
    )
    np.testing.assert_array_equal(days, [1, 2, 3])
    np.testing.assert_allclose(mean_vec, [0.7, 79 / 120, 1])
    np.testing.assert_array_equal(quantile_vecs, [[0, 0, 1], [0, 1, 1], [2, 1, 1]])


def test_plot_unreported_incidence():
    model_names = ["sse", "ssi", "sse_so", "ssi_so"]
    days = np.arange(1, 60)
    # Two unreported cases each day with probability 0.5, so a mean of 1
    pmf_df = pd.DataFrame(
        {
            "day": np.repeat(days, 2),
            "n_unreported": np.tile([0, 2], len(days)),
            "probability": 0.5,
        }
    )
    fig, (ax,) = make_panel_figure(panel_widths=[None], n_columns=2)
    plot_unreported_incidence(
        ax=ax,
        unreported_pmf_df_by_model=dict.fromkeys(model_names, pmf_df),
        reported_incidence_vec=np.arange(60) % 4,
        first_case_date=_FIRST_CASE_DATE,
        end_date=pd.Timestamp("2018-06-01"),
        vline_days=[34],
    )
    fig.canvas.draw()

    # A strip per model, filling a two-column grid row by row, with shared axes
    strip_axes = ax.child_axes
    assert len(strip_axes) == len(model_names)
    positions = [strip_ax.get_position() for strip_ax in strip_axes]
    assert positions[0].x0 == pytest.approx(positions[2].x0)
    assert positions[1].x0 == pytest.approx(positions[3].x0)
    assert positions[0].x1 < positions[1].x0
    assert positions[0].y0 == pytest.approx(positions[1].y0)
    assert positions[2].y0 == pytest.approx(positions[3].y0)
    assert positions[2].y1 < positions[0].y0
    for strip_ax in strip_axes:
        assert strip_ax.get_shared_x_axes().joined(strip_axes[0], strip_ax)
        assert strip_ax.get_shared_y_axes().joined(strip_axes[0], strip_ax)

    # Each strip's mean unreported cases, with the reported cases named in the first
    # legend only
    for strip_idx, strip_ax in enumerate(strip_axes):
        mean_line, vline = strip_ax.get_lines()
        np.testing.assert_array_equal(mean_line.get_xdata(), days)
        np.testing.assert_allclose(mean_line.get_ydata(), 1)
        assert vline.get_xdata()[0] == 34
    assert [
        [text.get_text() for text in strip_ax.get_legend().get_texts()]
        for strip_ax in strip_axes
    ] == [
        ["Reported", "Unreported (SSE)"],
        ["Unreported (SSI)"],
        ["Unreported (SSE-SO)"],
        ["Unreported (SSI-SO)"],
    ]

    # Ticks labelled on the outer strips only (get_*ticklabels leave out hidden labels),
    # over the months before end_date: April and May
    assert [bool(strip_ax.get_xticklabels()) for strip_ax in strip_axes] == [
        False,
        False,
        True,
        True,
    ]
    assert [bool(strip_ax.get_yticklabels()) for strip_ax in strip_axes] == [
        True,
        False,
        True,
        False,
    ]
    assert strip_axes[0].get_xlim() == (-3, 58)
    assert strip_axes[0].get_ylim()[0] == 0
    # One label for each axis, centred on the panel
    assert [strip_ax.get_xlabel() for strip_ax in strip_axes] == [
        "",
        "",
        "Date (2018)",
        "",
    ]
    assert [strip_ax.get_ylabel() for strip_ax in strip_axes] == [
        "",
        "",
        "Number of cases",
        "",
    ]
    panel_extent = ax.get_window_extent()
    xlabel_extent = strip_axes[2].xaxis.label.get_window_extent()
    ylabel_extent = strip_axes[2].yaxis.label.get_window_extent()
    assert (xlabel_extent.x0 + xlabel_extent.x1) / 2 == pytest.approx(
        (panel_extent.x0 + panel_extent.x1) / 2
    )
    assert (ylabel_extent.y0 + ylabel_extent.y1) / 2 == pytest.approx(
        (panel_extent.y0 + panel_extent.y1) / 2
    )


@pytest.mark.parametrize(
    ("risk_types", "expected_lines", "expected_ylabel"),
    # Each line's model, risk type, label and style, which is solid for the panel's first
    # risk type and dashed for its second, whichever they are
    [
        (
            ("cases",),
            [("sse", "cases", "SSE", "-"), ("sse_so", "cases", "SSE-SO", "-")],
            "Risk of additional cases",
        ),
        (
            ("cases", "transmission"),
            [
                ("sse", "cases", "SSE", "-"),
                ("sse_so", "cases", "SSE-SO (cases)", "-"),
                ("sse_so", "transmission", "SSE-SO (transmission)", "--"),
            ],
            "Risk",
        ),
        # Naive models' risks of sustained transmission and reports differ from that of
        # cases, so get their own curves
        (
            ("sustained",),
            [("sse", "sustained", "SSE", "-"), ("sse_so", "sustained", "SSE-SO", "-")],
            "Risk of sustained transmission",
        ),
        (
            ("cases", "reports"),
            [
                ("sse", "cases", "SSE (cases)", "-"),
                ("sse", "reports", "SSE (reports)", "--"),
                ("sse_so", "cases", "SSE-SO (cases)", "-"),
                ("sse_so", "reports", "SSE-SO (reports)", "--"),
            ],
            "Risk",
        ),
        # Without the risk of cases alongside, naive models show that of transmission
        (
            ("transmission",),
            [
                ("sse", "transmission", "SSE", "-"),
                ("sse_so", "transmission", "SSE-SO", "-"),
            ],
            "Risk of additional transmission",
        ),
    ],
)
def test_plot_risk_incidence_overlay(
    *, risk_df_by_model, risk_types, expected_lines, expected_ylabel
):
    reported_incidence_vec = np.arange(60) % 4
    vline_days = [34, 70]
    fig, (ax,) = make_panel_figure(panel_widths=[2])
    plot_risk_incidence_overlay(
        ax=ax,
        risk_df_by_model=risk_df_by_model,
        reported_incidence_vec=reported_incidence_vec,
        first_case_date=_FIRST_CASE_DATE,
        risk_types=risk_types,
        vline_days=vline_days,
    )
    fig.canvas.draw()
    _, twin_ax = fig.axes

    *risk_lines, vline_before, vline_after = ax.get_lines()
    assert len(risk_lines) == len(expected_lines)
    for line, (model_name, risk_type, label, linestyle) in zip(
        risk_lines, expected_lines, strict=True
    ):
        risk_df = risk_df_by_model[model_name]
        np.testing.assert_array_equal(line.get_xdata(), risk_df["day"])
        np.testing.assert_array_equal(line.get_ydata(), risk_df[f"risk_{risk_type}"])
        assert line.get_linestyle() == linestyle
        assert line.get_label() == label
        # Unclipped, so curves at a risk of 0 or 1 show in full
        assert not line.get_clip_on()
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        label for _, _, label, _ in expected_lines
    ]
    assert [vline_before.get_xdata()[0], vline_after.get_xdata()[0]] == vline_days
    assert ax.get_ylabel() == expected_ylabel

    # Incidence bars in the same days-since-first-case coordinates as the risks, with the
    # x axis from the start of the first risk day's month to the last risk day, and risks
    # from 0 to 1
    bars = twin_ax.patches
    np.testing.assert_allclose(
        [bar.get_x() + bar.get_width() / 2 for bar in bars], np.arange(60)
    )
    np.testing.assert_array_equal(
        [bar.get_height() for bar in bars], reported_incidence_vec
    )
    assert ax.get_xlim() == (27, 79)
    assert ax.get_ylim() == (0, 1)


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
