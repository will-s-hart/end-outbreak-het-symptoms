import argparse

import matplotlib.patches
import matplotlib.pyplot as plt

from endoutbreakhet.plotting import (
    DELAY_COLORS,
    MODEL_STYLES,
    make_panel_figure,
    plot_incidence,
    save_figure,
    set_plot_style,
)
from scripts.inputs import get_inputs_overview_plots

# The schematics are drawn in inches, in black with these greys for secondary text and for
# lines that only guide the eye
_MUTED_COLOR = "0.35"
_GUIDE_COLOR = "0.8"

# The superspreading schematic's box for each model: what its name stands for, what
# transmission varies between, the level its superspreading is at and which events (infector
# and day), infectors or days are superspreading, and the transmissions by each infector
# (rows) on each day (columns), 12 in every box so only how they fall differs. DLO's are
# each day's total, since it has no way to say which infector caused a transmission
_SUPERSPREADING_BOXES = {
    "sse": {
        "expansion": "superspreading events",
        "subtitle": "Transmission varies by infector and day",
        "level": "event",
        "superspreading": [(0, 1), (0, 3), (2, 0)],
        "counts": [[0, 4, 0, 3], [1, 0, 0, 0], [3, 0, 1, 0]],
    },
    "ssi": {
        "expansion": "superspreading individuals",
        "subtitle": "Transmission varies between infectors",
        "level": "infector",
        "superspreading": [1],
        "counts": [[0, 1, 0, 0], [3, 2, 3, 2], [0, 0, 1, 0]],
    },
    "dlo": {
        "expansion": "day-level overdispersion",
        "subtitle": "Total transmission varies between days",
        "level": "day",
        "superspreading": [2],
        "counts": [[1, 0, 10, 1]],
    },
}
_SUPERSPREADING_N_INFECTORS = 3
# The dots in each row of a cell's cluster, from the top, by the number of transmissions
_DOT_ROWS = {0: (), 1: (1,), 2: (2,), 3: (1, 2), 4: (2, 2), 10: (3, 4, 3)}


def make_plots():
    set_plot_style()
    inputs = get_inputs_overview_plots()

    # The incidence above the two schematics, each a row of its own, so wide, and as tall
    # (in inches) as its content needs
    fig, (incidence_ax, superspreading_ax, onset_anchoring_ax) = make_panel_figure(
        panels_per_row=[1, 1, 1], axes_heights=[3.0, 3.3, 2.6], schematic_rows=(1, 2)
    )
    plot_incidence(
        ax=incidence_ax,
        reported_incidence_vec=inputs["reported_incidence_vec"],
        first_case_date=inputs["first_case_date"],
        ert_days=inputs["ert_days"],
    )
    _plot_superspreading_schematic(ax=superspreading_ax)
    _plot_onset_anchoring_schematic(
        ax=onset_anchoring_ax, delay_means=inputs["delay_means"]
    )
    save_figure(fig=fig, paths=inputs["fig_paths"]["summary"])


def _plot_superspreading_schematic(*, ax):
    # A box for each superspreading model, of infectors (rows) by day (columns) with a dot
    # for each transmission, tinted where the model's superspreading is: at single events,
    # across an infector's days or, for all infectors together, across a day
    width, height = _schematic_axes(ax=ax)
    gap = 0.4
    box_width = (width - gap * (len(_SUPERSPREADING_BOXES) - 1)) / len(
        _SUPERSPREADING_BOXES
    )
    for box_idx, (model_name, box) in enumerate(_SUPERSPREADING_BOXES.items()):
        _draw_superspreading_box(
            ax=ax,
            left=box_idx * (box_width + gap),
            width=box_width,
            height=height,
            model_name=model_name,
            **box,
        )


def _schematic_axes(*, ax):
    # Makes ax a blank canvas for a schematic, with coordinates in inches from its
    # bottom-left corner, so shapes keep their proportions and sizes are those on the page.
    # Returns its width and height
    width, height = ax.get_position().size * ax.get_figure(root=True).get_size_inches()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return width, height


def _draw_superspreading_box(
    *,
    ax,
    left,
    width,
    height,
    model_name,
    expansion,
    subtitle,
    level,
    superspreading,
    counts,
):
    # A model's box of the superspreading schematic (see _SUPERSPREADING_BOXES), width by
    # height with its bottom-left corner at (left, 0)
    style = MODEL_STYLES[model_name]
    text_size = plt.rcParams["legend.fontsize"]
    ax.add_patch(
        matplotlib.patches.FancyBboxPatch(
            (left + 0.02, 0.02),
            width - 0.04,
            height - 0.04,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor="none",
            edgecolor=_GUIDE_COLOR,
            linewidth=1.5,
        )
    )
    ax.text(
        left + width / 2,
        height - 0.36,
        f"{style['label']}: {expansion}",
        color=style["color"],
        ha="center",
        fontsize=plt.rcParams["axes.titlesize"],
    )
    ax.text(
        left + width / 2,
        height - 0.66,
        subtitle,
        color=_MUTED_COLOR,
        ha="center",
        fontsize=text_size,
    )
    # The grid, between the infectors and their label on its left and a margin on its
    # right, and between the titles above and the days' labels below
    n_days = len(counts[0])
    grid_left, grid_bottom = left + 0.8, 0.68
    day_width = (width - 1.1) / n_days
    grid_width, grid_height = n_days * day_width, height - 1.55
    infector_height = grid_height / _SUPERSPREADING_N_INFECTORS
    # A row of cells for each row of counts, so one spanning every infector if the counts
    # are the days' totals
    cell_height = grid_height / len(counts)
    superspreading_cells = {
        "event": superspreading,
        "infector": [(row, day) for row in superspreading for day in range(n_days)],
        "day": [(row, day) for day in superspreading for row in range(len(counts))],
    }[level]
    for row, day in superspreading_cells:
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (
                    grid_left + day * day_width,
                    grid_bottom + grid_height - (row + 1) * cell_height,
                ),
                day_width,
                cell_height,
                facecolor=style["color"],
                alpha=0.3,
                edgecolor="none",
            )
        )
    for row in range(len(counts) + 1):
        y = grid_bottom + row * cell_height
        ax.plot(
            [grid_left, grid_left + grid_width],
            [y, y],
            color=_GUIDE_COLOR,
            linewidth=1,
        )
    for day in range(n_days + 1):
        x = grid_left + day * day_width
        ax.plot(
            [x, x],
            [grid_bottom, grid_bottom + grid_height],
            color=_GUIDE_COLOR,
            linewidth=1,
        )
    for row, row_counts in enumerate(counts):
        for day, count in enumerate(row_counts):
            _draw_dots(
                ax=ax,
                x=grid_left + (day + 0.5) * day_width,
                y=grid_bottom + grid_height - (row + 0.5) * cell_height,
                dot_rows=_DOT_ROWS[count],
            )
    for infector in range(_SUPERSPREADING_N_INFECTORS):
        _draw_person(
            ax=ax,
            x=grid_left - 0.3,
            y=grid_bottom + grid_height - (infector + 0.5) * infector_height,
            color=(
                style["color"]
                if level == "infector" and infector in superspreading
                else _MUTED_COLOR
            ),
        )
    ax.text(
        grid_left - 0.62,
        grid_bottom + grid_height / 2,
        "Infector",
        rotation=90,
        ha="center",
        va="center",
        fontsize=text_size,
    )
    for day in range(n_days):
        is_superspreading = level == "day" and day in superspreading
        ax.text(
            grid_left + (day + 0.5) * day_width,
            grid_bottom - 0.1,
            str(day + 1),
            color=style["color"] if is_superspreading else "black",
            fontweight="bold" if is_superspreading else "normal",
            ha="center",
            va="top",
            fontsize=text_size,
        )
    ax.text(
        grid_left + grid_width / 2,
        grid_bottom - 0.38,
        "Day",
        ha="center",
        va="top",
        fontsize=text_size,
    )


def _draw_dots(*, ax, x, y, dot_rows):
    # A cluster of dots centred on (x, y), dot_rows giving the number in each of its rows
    # from the top
    spacing = 0.17
    for row, n_row_dots in enumerate(dot_rows):
        for dot in range(n_row_dots):
            ax.add_patch(
                matplotlib.patches.Circle(
                    (
                        x + spacing * (dot - (n_row_dots - 1) / 2),
                        y + 0.9 * spacing * ((len(dot_rows) - 1) / 2 - row),
                    ),
                    0.055,
                    color="0.15",
                    linewidth=0,
                )
            )


def _draw_person(*, ax, x, y, color):
    # A head and shoulders, centred on (x, y)
    ax.add_patch(
        matplotlib.patches.Circle((x, y + 0.09), 0.075, color=color, linewidth=0)
    )
    ax.add_patch(
        matplotlib.patches.Wedge((x, y - 0.16), 0.15, 0, 180, color=color, linewidth=0)
    )


def _plot_onset_anchoring_schematic(*, ax, delay_means):
    # An infector and their infectee on a time axis, as the naive models (above) and the SO
    # models (below) read the same two onsets, with each delay's arrow as long as its mean
    # (delay_means, by delay)
    width, height = _schematic_axes(ax=ax)
    text_size = plt.rcParams["legend.fontsize"]
    # The models' names and descriptions take the left, the time axes the rest, with room
    # at their ends for the outer events' labels
    names_width = 2.75
    axis_start, axis_end = names_width + 0.15, width - 0.1
    infector_onset, infectee_onset = axis_start + 0.85, axis_end - 2
    transmission = infector_onset + (
        (infectee_onset - infector_onset)
        * delay_means["tost"]
        / delay_means["serial_interval"]
    )
    naive_y, so_y = height - 0.6, 0.7
    for y, name, description in (
        (naive_y, "Naive models", "Onset dates treated\nas infection dates"),
        (so_y, "SO models", "Transmission timed from\nthe infector's onset"),
    ):
        ax.text(0, y + 0.3, name, va="center", fontsize=plt.rcParams["axes.titlesize"])
        ax.text(
            0,
            y - 0.02,
            description,
            color=_MUTED_COLOR,
            va="top",
            fontsize=text_size,
            linespacing=1.15,
        )
        _draw_arrow(
            ax=ax,
            start=axis_start,
            end=axis_end,
            y=y,
            color=_GUIDE_COLOR,
            linewidth=1.5,
        )
    ax.text(axis_end, so_y - 0.14, "Time", ha="right", va="top", fontsize=text_size)
    # The onsets are the data, so the same in both readings
    for onset in (infector_onset, infectee_onset):
        ax.plot(
            [onset, onset],
            [so_y, naive_y],
            color=_GUIDE_COLOR,
            linewidth=1.5,
            linestyle=":",
            zorder=0,
        )
    for y, delay_name, label, start, end in (
        (naive_y, "serial_interval", "Serial interval", infector_onset, infectee_onset),
        (so_y, "tost", "Onset to transmission", infector_onset, transmission),
        (so_y, "inc_period", "Incubation period", transmission, infectee_onset),
    ):
        _draw_arrow(
            ax=ax,
            start=start,
            end=end,
            y=y + 0.24,
            color=DELAY_COLORS[delay_name],
            linewidth=2.5,
        )
        ax.text(
            (start + end) / 2,
            y + 0.34,
            label,
            color=DELAY_COLORS[delay_name],
            ha="center",
            va="bottom",
            fontsize=text_size,
        )
    for y, x, label, observed in (
        (naive_y, infector_onset, "Infector's onset", True),
        (naive_y, infectee_onset, "Infectee's onset", True),
        (so_y, infector_onset, "Infector's onset", True),
        (so_y, transmission, "Infectee's infection\n(unobserved)", False),
        (so_y, infectee_onset, "Infectee's onset", True),
    ):
        # In front of the time axis
        ax.plot(
            [x],
            [y],
            marker="o",
            markersize=13,
            markerfacecolor="black" if observed else "white",
            markeredgecolor="black",
            markeredgewidth=2,
            zorder=5,
        )
        ax.text(
            x,
            y - 0.17,
            label,
            ha="center",
            va="top",
            fontsize=text_size,
            linespacing=1.1,
        )


def _draw_arrow(*, ax, start, end, y, color, linewidth):
    # A horizontal arrow from start to end at height y
    ax.annotate(
        "",
        xy=(end, y),
        xytext=(start, y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": linewidth,
            "mutation_scale": 7 * linewidth,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=(
            "Plot the reported incidence, with schematics of the superspreading models "
            "and of the naive and symptom onset-anchored models."
        )
    ).parse_args()

    make_plots()
