import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

from endoutbreakhet.utils import parse_lognormal_prior

# Figures are rows of panels. A figure has a width, which the panels of each row share
# equally, so every row starts and ends in line with the others. A panel's axes sit within
# its share at margins in inches, the left one holding the y tick labels and axis label, with
# the panel's label above them. The axes are _AXES_HEIGHT tall, or square if that would make
# them taller than wide, so panels sharing a row with many others don't end up as slivers.
# The figure has a margin of its own on the right, for a twin y axis on a row's last panel
_FIGURE_WIDTH = 14.0
_AXES_HEIGHT = 5.3
_AXES_MARGINS = {"left": 0.85, "right": 0.3, "bottom": 0.8, "top": 0.4}
_FIGURE_RIGHT_MARGIN = 0.6
# Below a row of schematics, which have no x axis to label
_SCHEMATIC_BOTTOM_MARGIN = 0.15
# From the top-left corner of a panel to the start of its label's baseline
_PANEL_LABEL_OFFSET = (0.07, 0.28)
_PANEL_LABEL_SIZE = 20
# The box for the model probabilities inset in another panel, in that panel's upper right,
# which R_after's densities leave empty: the box's bounds in the panel's axes fraction, the
# pie's in the box's, the pie's radius, the gap around the legend below the pie (in font
# sizes) and how far across the box its label starts. The compact layout fits a panel in a
# row of three, with a smaller pie beside the label rather than below it
_MODEL_PROBABILITIES_INSET_LAYOUTS = {
    "standard": {
        "box_bounds": (0.33, 0.22, 0.65, 0.76),
        "pie_bounds": (0, 0.46, 1, 0.5),
        "pie_radius": 1,
        "legend_pad": 0.5,
        "label_x": 0.02,
    },
    "compact": {
        "box_bounds": (0.327, 0.176, 0.663, 0.814),
        "pie_bounds": (0.2, 0.61, 0.6, 0.37),
        "pie_radius": 1.2,
        "legend_pad": 0.35,
        "label_x": 0.03,
    },
}

# Seaborn's colorblind palette, by hue
_COLORS = dict(
    zip(
        (
            "blue",
            "orange",
            "green",
            "red",
            "purple",
            "brown",
            "pink",
            "gray",
            "yellow",
            "cyan",
        ),
        sns.color_palette("colorblind"),
        strict=True,
    )
)

_PARAM_LABELS = {
    "rep_no_before": r"$R_\mathrm{before}$",
    "rep_no_after": r"$R_\mathrm{after}$",
    "dispersion": r"$k$",
}

# Public, as are DELAY_COLORS, so figures a script draws itself (such as schematics) match
# the rest. Each naive model and its SO variant are a dark and a light shade of one hue
MODEL_STYLES = {
    "poisson": {"label": "Poisson", "color": "black"},
    "poisson_so": {"label": "Poisson-SO", "color": _COLORS["gray"]},
    "sse": {"label": "SSE", "color": _COLORS["blue"]},
    "sse_so": {"label": "SSE-SO", "color": _COLORS["cyan"]},
    "ssi": {"label": "SSI", "color": _COLORS["red"]},
    "ssi_so": {"label": "SSI-SO", "color": _COLORS["orange"]},
    "dlo": {"label": "DLO", "color": _COLORS["green"]},
}

# Reported cases, in every figure that shows them
_INCIDENCE_COLOR = "darkgray"

# The mean of the unreported cases each day, with this central interval of them, in a grid
# of strips this many columns wide, gaps apart (in the panel's axes fraction): between
# columns, room for the bottom row's outer x tick labels, and between rows, for the legends
_UNREPORTED_INTERVAL = (0.025, 0.975)
_UNREPORTED_N_COLUMNS = 2
_UNREPORTED_GAPS = {"column": 0.06, "row": 0.09}

# By the risk type's place among those in a panel, so the first is always solid
_RISK_TYPE_LINESTYLES = ("-", "--", ":", "-.")

_RISK_TYPE_YLABELS = {
    "cases": "Risk of additional cases",
    "transmission": "Risk of additional transmission",
    "sustained": "Risk of sustained transmission",
    "reports": "Risk of additional reported cases",
}

_GATE_STATUS_COLORS = {
    "pass": _COLORS["green"],
    "fail": _COLORS["red"],
    "unassessed": "lightgray",
}

_GATE_CRITERION_TITLES = {
    "rhat": r"$\hat{R}$",
    "ess_bulk": "Bulk ESS",
    "ess_tail": "Tail ESS",
    "divergences": "Divergences",
    "tree_depth": "Max tree depth",
    "risk": "Risk",
}

_DELAY_XLABELS = {
    "serial_interval": "Serial interval (days)",
    "inc_period": "Incubation period (days)",
    "tost": "Time from symptom onset to transmission (days)",
}

# A serial interval is a TOST then an incubation period, so its colour is the mix of theirs:
# a purple, from blue and a crimson (the palette's own reds are too orange to mix to one).
# The bars are drawn translucent, so a line overlaid on them stands out
DELAY_COLORS = {
    "inc_period": _COLORS["blue"],
    "tost": matplotlib.colors.to_rgb("#d6336c"),
}
DELAY_COLORS["serial_interval"] = tuple(np.mean(list(DELAY_COLORS.values()), axis=0))
_DELAY_BAR_ALPHA = 0.5


def set_plot_style():
    # The style every figure shares, so call it before making any
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=list(_COLORS.values())),
            "savefig.transparent": True,
            # Text stays editable in SVGs
            "svg.fonttype": "none",
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            # Clears x tick labels from y tick labels where axes meet at a labelled corner,
            # as on the risk panels
            "xtick.major.pad": 6,
            "legend.fontsize": 14,
            "legend.title_fontsize": 14,
            "font.family": "sans-serif",
            "font.sans-serif": "Arial",
        }
    )


def make_panel_figure(
    *,
    panels_per_row,
    width=_FIGURE_WIDTH,
    labels=None,
    sharey=False,
    gutters=None,
    axes_heights=None,
    schematic_rows=(),
):
    # A figure width inches wide of rows of panels, panels_per_row giving the number in each
    # row, and the panels' axes in order. A row's panels share the width equally, and their
    # axes are _AXES_HEIGHT tall, or square if narrower than that, unless axes_heights gives
    # the row's height in inches (None for a row keeps that rule). Panels are labelled "A.",
    # "B.", ... unless there is just one, or by labels (None for no label). With sharey, the
    # panels share the first's y axis, labelled only on panels starting a row. gutters adds
    # space (in inches, by side) around the panels for figure-level decorations. The rows
    # in schematic_rows (by index) hold schematics, so leave no room for an x axis's labels
    n_panels = sum(panels_per_row)
    if labels is None:
        labels = (
            [f"{chr(ord('A') + idx)}." for idx in range(n_panels)]
            if n_panels > 1
            else [None]
        )
    if len(labels) != n_panels:
        raise ValueError(f"Got {len(labels)} labels for {n_panels} panels: {labels}")
    gutters = dict.fromkeys(("left", "right", "bottom", "top"), 0) | (gutters or {})
    panel_widths = [
        (width - _FIGURE_RIGHT_MARGIN) / n_row_panels for n_row_panels in panels_per_row
    ]
    axes_widths = [
        panel_width - _AXES_MARGINS["left"] - _AXES_MARGINS["right"]
        for panel_width in panel_widths
    ]
    axes_heights = [
        min(_AXES_HEIGHT, axes_width) if axes_height is None else axes_height
        for axes_width, axes_height in zip(
            axes_widths, axes_heights or [None] * len(axes_widths), strict=True
        )
    ]
    bottom_margins = [
        _SCHEMATIC_BOTTOM_MARGIN
        if row_idx in schematic_rows
        else _AXES_MARGINS["bottom"]
        for row_idx in range(len(panels_per_row))
    ]
    row_heights = [
        _AXES_MARGINS["top"] + axes_height + bottom_margin
        for axes_height, bottom_margin in zip(axes_heights, bottom_margins, strict=True)
    ]
    fig_width = gutters["left"] + width + gutters["right"]
    fig_height = gutters["top"] + sum(row_heights) + gutters["bottom"]
    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = []
    # In inches from the bottom of the figure, as are the positions below
    row_top = fig_height - gutters["top"]
    for (
        n_row_panels,
        panel_width,
        axes_width,
        axes_height,
        bottom_margin,
        row_height,
    ) in zip(
        panels_per_row,
        panel_widths,
        axes_widths,
        axes_heights,
        bottom_margins,
        row_heights,
        strict=True,
    ):
        for row_idx in range(n_row_panels):
            panel_left = gutters["left"] + row_idx * panel_width
            ax = fig.add_axes(
                (
                    (panel_left + _AXES_MARGINS["left"]) / fig_width,
                    (row_top - row_height + bottom_margin) / fig_height,
                    axes_width / fig_width,
                    axes_height / fig_height,
                ),
                sharey=axes[0] if sharey and axes else None,
            )
            if sharey and row_idx > 0:
                ax.tick_params(labelleft=False)
            if (label := labels[len(axes)]) is not None:
                fig.text(
                    panel_left + _PANEL_LABEL_OFFSET[0],
                    row_top - _PANEL_LABEL_OFFSET[1],
                    label,
                    fontsize=_PANEL_LABEL_SIZE,
                    transform=fig.dpi_scale_trans,
                )
            axes.append(ax)
        row_top -= row_height
    return fig, axes


def plot_densities(
    *,
    ax,
    posterior_df_by_model,
    priors,
    param_name,
    log_x=False,
    legend=True,
    legend_kwargs=None,
):
    # legend_kwargs go to the legend, e.g. to fit it in a small panel
    lognormal_params = parse_lognormal_prior(name=param_name, priors=priors)
    prior_dist = scipy.stats.lognorm(
        s=lognormal_params["sigma"], scale=np.exp(lognormal_params["mu"])
    )
    # Models without the parameter (e.g. Poisson models for the dispersion) are left out
    samples_by_model = {
        model_name: posterior_df[param_name].to_numpy()
        for model_name, posterior_df in posterior_df_by_model.items()
        if param_name in posterior_df.columns
    }

    # Limits from central prior/posterior mass, so long tails don't squash the plot
    upper = max(
        [
            prior_dist.ppf(0.975),
            *(np.quantile(samples, 0.99) for samples in samples_by_model.values()),
        ]
    )
    if log_x:
        lower = min(
            [
                prior_dist.ppf(0.025),
                *(np.quantile(samples, 0.01) for samples in samples_by_model.values()),
            ]
        )
        x_vec = np.geomspace(lower, upper, 500)
        # Densities per unit log10(x), so areas on the log axis are probabilities
        kde_eval_vec = np.log10(x_vec)
        prior_density_vec = prior_dist.pdf(x_vec) * x_vec * np.log(10)
        ax.set_xscale("log")
    else:
        lower = 0
        x_vec = np.linspace(lower, upper, 500)
        kde_eval_vec = x_vec
        prior_density_vec = prior_dist.pdf(x_vec)

    for model_name, samples in samples_by_model.items():
        kde = scipy.stats.gaussian_kde(np.log10(samples) if log_x else samples)
        model_style = MODEL_STYLES[model_name]
        ax.plot(
            x_vec,
            kde(kde_eval_vec),
            color=model_style["color"],
            label=model_style["label"],
        )
    ax.plot(x_vec, prior_density_vec, "k--", label="Prior")
    ax.set_xlim(lower, upper)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(_PARAM_LABELS[param_name])
    ax.set_ylabel("Density")

    if legend:
        ax.legend(**(legend_kwargs or {}))


def plot_model_probabilities_inset(
    *, ax, model_probability_by_model, label, compact=False
):
    # A pie of the models' probabilities over its legend, which gives each model's, in a
    # framed box in the upper right of ax's panel, labelled as a panel of its own. compact
    # fits the box to a panel in a row of three. The wedges are edged, so a
    # pie of one model and others of probability near 0 still shows its divisions
    layout = _MODEL_PROBABILITIES_INSET_LAYOUTS["compact" if compact else "standard"]
    box_ax = ax.inset_axes(layout["box_bounds"])
    box_ax.set_xticks([])
    box_ax.set_yticks([])
    for spine in box_ax.spines.values():
        spine.set_visible(True)
    pie = box_ax.inset_axes(layout["pie_bounds"]).pie(
        list(model_probability_by_model.values()),
        colors=[
            MODEL_STYLES[model_name]["color"]
            for model_name in model_probability_by_model
        ],
        wedgeprops={"edgecolor": "white", "linewidth": 1},
        radius=layout["pie_radius"],
    )
    box_ax.legend(
        pie.wedges,
        [
            f"{MODEL_STYLES[model_name]['label']} ({prob:.2f})"
            for model_name, prob in model_probability_by_model.items()
        ],
        title="Model probability",
        loc="lower center",
        borderaxespad=layout["legend_pad"],
    )
    box_ax.text(
        layout["label_x"],
        0.97,
        label,
        transform=box_ax.transAxes,
        va="top",
        fontsize=_PANEL_LABEL_SIZE,
    )


def plot_risk_incidence_overlay(
    *,
    ax,
    risk_df_by_model,
    reported_incidence_vec,
    first_case_date,
    risk_types=("cases",),
    vline_days=(),
):
    # x is in days since the first case, up to the last risk day
    for model_name, risk_df in risk_df_by_model.items():
        model_style = MODEL_STYLES[model_name]
        # Risks of cases and transmission coincide in the naive models, so they get one curve
        model_risk_types = (
            tuple(risk_type for risk_type in risk_types if risk_type != "transmission")
            if not model_name.endswith("_so") and "cases" in risk_types
            else risk_types
        )
        for risk_type in model_risk_types:
            label = model_style["label"]
            if len(model_risk_types) > 1:
                label += f" ({risk_type})"
            ax.plot(
                risk_df["day"],
                risk_df[f"risk_{risk_type}"],
                color=model_style["color"],
                linestyle=_RISK_TYPE_LINESTYLES[risk_types.index(risk_type)],
                label=label,
                # Unclipped, so curves at a risk of 0 or 1 aren't halved by the limits
                clip_on=False,
            )
    for vline_day in vline_days:
        ax.axvline(vline_day, color="black", linestyle="--")
    ax.legend()
    # Every model has the same risk days
    risk_days = next(iter(risk_df_by_model.values()))["day"].to_numpy()
    _month_start_x_axis(
        ax=ax, day_0_date=first_case_date, plot_days=risk_days, interval_months=1
    )
    ax.set_xlim(right=risk_days[-1])
    ax.set_ylim(0, 1)
    ax.set_ylabel(_RISK_TYPE_YLABELS[risk_types[0]] if len(risk_types) == 1 else "Risk")
    twin_ax = ax.twinx()
    twin_ax.bar(
        x=np.arange(len(reported_incidence_vec)),
        height=reported_incidence_vec,
        color=_INCIDENCE_COLOR,
    )
    twin_ax.set_ylim(0, np.max(reported_incidence_vec))
    twin_ax.set_ylabel("Number of cases")
    twin_ax.yaxis.label.set_color("tab:gray")
    twin_ax.tick_params(axis="y", colors="tab:gray")
    twin_ax.spines["right"].set_visible(True)
    twin_ax.spines["right"].set_color("tab:gray")
    twin_ax.spines["left"].set_visible(False)
    twin_ax.spines["bottom"].set_visible(False)
    # Draw the primary axis (curves, legend) above the incidence bars.
    ax.set_zorder(twin_ax.get_zorder() + 1)
    ax.patch.set_visible(False)


def plot_unreported_incidence(
    *,
    ax,
    unreported_pmf_df_by_model,
    reported_incidence_vec,
    first_case_date,
    end_date,
    vline_days=(),
):
    # A strip for each model, filling a grid across ax's panel row by row, of the reported
    # cases by day since the first case, with the mean and central interval of the model's
    # unreported cases each day, from their posterior pmf (in columns day, n_unreported and
    # probability), over the whole months before end_date. The strips share their axes, so
    # their counts can be compared, with ticks labelled on the outer strips only
    ax.axis("off")
    n_rows = -(-len(unreported_pmf_df_by_model) // _UNREPORTED_N_COLUMNS)
    strip_width = (
        1 - (_UNREPORTED_N_COLUMNS - 1) * _UNREPORTED_GAPS["column"]
    ) / _UNREPORTED_N_COLUMNS
    strip_height = (1 - (n_rows - 1) * _UNREPORTED_GAPS["row"]) / n_rows
    strip_axes = []
    for strip_idx, (model_name, pmf_df) in enumerate(
        unreported_pmf_df_by_model.items()
    ):
        row_idx, column_idx = divmod(strip_idx, _UNREPORTED_N_COLUMNS)
        strip_ax = ax.inset_axes(
            (
                column_idx * (strip_width + _UNREPORTED_GAPS["column"]),
                1 - row_idx * _UNREPORTED_GAPS["row"] - (row_idx + 1) * strip_height,
                strip_width,
                strip_height,
            ),
            sharex=strip_axes[0] if strip_axes else None,
            sharey=strip_axes[0] if strip_axes else None,
        )
        strip_axes.append(strip_ax)
        model_style = MODEL_STYLES[model_name]
        days, mean_vec, (lower_vec, upper_vec) = _summarise_unreported_pmf(
            pmf_df=pmf_df, quantile_levels=_UNREPORTED_INTERVAL
        )
        reported_bars = strip_ax.bar(
            np.arange(len(reported_incidence_vec)),
            reported_incidence_vec,
            width=1,
            color=_INCIDENCE_COLOR,
        )
        interval_band = strip_ax.fill_between(
            days,
            lower_vec,
            upper_vec,
            step="mid",
            color=model_style["color"],
            alpha=0.3,
            linewidth=0,
        )
        (mean_line,) = strip_ax.step(
            days, mean_vec, where="mid", color=model_style["color"]
        )
        for vline_day in vline_days:
            strip_ax.axvline(vline_day, color="black", linestyle="--")
        # The reported cases are the same in every strip, so only the first names them
        legend_items = [
            *([(reported_bars, "Reported")] if strip_idx == 0 else []),
            ((interval_band, mean_line), f"Unreported ({model_style['label']})"),
        ]
        strip_ax.legend(*zip(*legend_items, strict=True), loc="upper right")
        strip_ax.yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=4, integer=True)
        )
        strip_ax.tick_params(
            labelbottom=(row_idx == n_rows - 1), labelleft=(column_idx == 0)
        )
    # One label for each axis, on the bottom left strip and centred on the panel. Only
    # their position along the axis is set, as their distance from it is kept clear of the
    # tick labels on each draw
    label_ax = strip_axes[(n_rows - 1) * _UNREPORTED_N_COLUMNS]
    _month_start_x_axis(
        ax=label_ax,
        day_0_date=first_case_date,
        plot_days=days[days < (end_date - first_case_date).days],
    )
    label_ax.set_ylim(bottom=0)
    label_ax.xaxis.label.set_x(0.5 / strip_width)
    label_ax.set_ylabel("Number of cases")
    label_ax.yaxis.label.set_y(0.5 / strip_height)


def plot_incidence(*, ax, reported_incidence_vec, first_case_date, ert_days):
    # Reported cases by day since the first case, with the ERT's arrival and departure
    # (ert_days, by event) marked, each labelled on the side of its line nearer the middle
    # of the days, so a label by a late event stays within the panel
    days = np.arange(len(reported_incidence_vec))
    ax.bar(days, reported_incidence_vec, color=_INCIDENCE_COLOR)
    for event, day in ert_days.items():
        label_on_left = day > days[-1] / 2
        ax.axvline(day, color="black", linestyle="--")
        ax.annotate(
            f"ERT {event}",
            xy=(day, 1),
            xycoords=("data", "axes fraction"),
            xytext=(-4 if label_on_left else 4, 0),
            textcoords="offset points",
            ha="right" if label_on_left else "left",
            va="top",
            fontsize=plt.rcParams["axes.labelsize"],
        )
    _month_start_x_axis(ax=ax, day_0_date=first_case_date, plot_days=days)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Number of cases")


def plot_delay_distribution(
    *, ax, delay, delay_name, max_lag, label=None, overlay=None, overlay_label=None
):
    # Probability mass by lag as bars, optionally with another delay's as a line
    bars = ax.bar(
        delay.lags,
        delay.prob,
        width=1,
        color=DELAY_COLORS[delay_name],
        alpha=_DELAY_BAR_ALPHA,
        label=label,
    )
    if overlay is not None:
        (line,) = ax.plot(
            overlay.lags, overlay.prob, color="black", label=overlay_label
        )
        # Bars first, which matplotlib would otherwise list after lines
        ax.legend(handles=[bars, line])
    ax.set_xlim(-0.5, max_lag + 0.5)
    ax.set_ylim(bottom=0)
    # The default tick steps without 2.5, whose labels take a decimal place more than their
    # neighbours' (0.025 between 0.02 and 0.05) and so more of the margin
    ax.yaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(nbins="auto", steps=[1, 2, 5, 10])
    )
    ax.set_xlabel(_DELAY_XLABELS[delay_name])
    ax.set_ylabel("Probability")


def plot_fit_gate(
    *,
    axes,
    status_by_criterion,
    fit_status,
    evidence_status,
    spec_names,
    days,
    comparison_day,
):
    # A panel on each of axes (sharing y) for each criterion's statuses ("pass", "fail" or
    # "unassessed"), of shape (specs, days) over consecutive days, with the comparison day
    # marked and the risk cells hatched where fit_status is "fail" (their MCSE is then
    # conditional on a failed fit). Each spec's evidence status is a column in the right
    # margin of the last panel, and the legend is at the bottom of the figure, which should
    # leave a gutter for it
    n_specs = len(spec_names)
    criteria = list(status_by_criterion)
    status_codes = {status: code for code, status in enumerate(_GATE_STATUS_COLORS)}
    cmap = matplotlib.colors.ListedColormap(list(_GATE_STATUS_COLORS.values()))

    def _plot_statuses(*, ax, statuses, x_lims):
        ax.imshow(
            np.vectorize(status_codes.__getitem__)(statuses),
            cmap=cmap,
            vmin=-0.5,
            vmax=len(status_codes) - 0.5,
            aspect="auto",
            interpolation="nearest",
            extent=(*x_lims, n_specs - 0.5, -0.5),
        )

    for ax, criterion in zip(axes, criteria, strict=True):
        _plot_statuses(
            ax=ax,
            statuses=np.asarray(status_by_criterion[criterion]),
            x_lims=(days[0] - 0.5, days[-1] + 0.5),
        )
        ax.axvline(comparison_day, color="black", linestyle="--")
        ax.set_title(_GATE_CRITERION_TITLES[criterion])
        ax.set_xlabel("Day")
    risk_ax = axes[criteria.index("risk")]
    for spec_idx, day_idx in zip(
        *np.nonzero(np.asarray(fit_status) == "fail"), strict=True
    ):
        risk_ax.add_patch(
            matplotlib.patches.Rectangle(
                (days[day_idx] - 0.5, spec_idx - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                hatch="////",
                linewidth=0,
            )
        )
    # Inset by 0.1 inches and 0.3 inches wide, whatever the panel's width
    fig = axes[0].get_figure(root=True)
    last_ax_width = axes[-1].get_position().width * fig.get_figwidth()
    evidence_ax = axes[-1].inset_axes(
        (1 + 0.1 / last_ax_width, 0, 0.3 / last_ax_width, 1)
    )
    _plot_statuses(
        ax=evidence_ax,
        statuses=np.asarray(evidence_status)[:, None],
        x_lims=(-0.5, 0.5),
    )
    # Labelled below, as a title would be wider than the column
    evidence_ax.set_xticks([0], labels=["Evidence"], rotation=90)
    evidence_ax.tick_params(left=False, labelleft=False)
    axes[0].set_yticks(np.arange(n_specs), labels=spec_names)
    fig.legend(
        handles=[
            *(
                matplotlib.patches.Patch(facecolor=color, label=status)
                for status, color in _GATE_STATUS_COLORS.items()
            ),
            matplotlib.patches.Patch(
                facecolor="white",
                edgecolor="black",
                hatch="////",
                label="risk given a failed fit",
            ),
        ],
        loc="lower center",
        ncols=len(_GATE_STATUS_COLORS) + 1,
    )


def save_figure(*, fig, paths):
    # Saves fig to each path (one per file format), then closes it
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    plt.close(fig)


def _month_start_x_axis(*, ax, day_0_date, plot_days, interval_months=1):
    # x axis in days since day_0_date, spanning the whole months that contain plot_days, with
    # ticks at the month starts and every interval_months-th one labelled
    plot_dates = day_0_date + pd.to_timedelta(plot_days, unit="D")
    month_start_dates = pd.date_range(
        start=plot_dates.min().replace(day=1),
        end=plot_dates.max().replace(day=1) + pd.offsets.MonthBegin(1),
        freq="MS",
    )
    month_start_days = (month_start_dates - day_0_date).days
    multi_year = month_start_dates.year.nunique() > 1
    labels = [
        f"{d.day} {d:%b}" + (f" {d.year}" if multi_year else "")
        if (i % interval_months == 0)
        else ""
        for i, d in enumerate(month_start_dates)
    ]
    ax.set_xticks(month_start_days, labels=labels)
    ax.set_xlim(month_start_days[0], month_start_days[-1])
    ax.set_xlabel("Date" if multi_year else f"Date ({month_start_dates[0].year})")


def _summarise_unreported_pmf(*, pmf_df, quantile_levels):
    # The days of the unreported cases' pmf by day (in columns day, n_unreported and
    # probability), with their mean on each and their quantiles at each of quantile_levels:
    # the least count whose cumulative probability reaches the level (to within rounding of
    # the probabilities)
    prob_df = pmf_df.pivot(
        index="day", columns="n_unreported", values="probability"
    ).fillna(0)
    counts = prob_df.columns.to_numpy()
    prob_matrix = prob_df.to_numpy()
    cdf_matrix = prob_matrix.cumsum(axis=1)
    return (
        prob_df.index.to_numpy(),
        prob_matrix @ counts,
        [
            counts[np.argmax(cdf_matrix >= level - 1e-9, axis=1)]
            for level in quantile_levels
        ],
    )
