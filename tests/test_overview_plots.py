# Note that AI tools were used to generate tests
import matplotlib.colors
import matplotlib.patches
import matplotlib.text
import pytest

from endoutbreakhet.plotting import MODEL_STYLES, make_panel_figure
from scripts import overview_plots
from scripts.inputs import get_inputs_overview_plots


def test_make_plots_writes_figures(*, tmp_path, monkeypatch):
    # Real inputs, with the figure redirected into tmp_path
    inputs = get_inputs_overview_plots()
    inputs["fig_paths"] = {
        fig_name: [tmp_path / path for path in paths]
        for fig_name, paths in inputs["fig_paths"].items()
    }
    monkeypatch.setattr(overview_plots, "get_inputs_overview_plots", lambda: inputs)

    overview_plots.make_plots()

    for paths in inputs["fig_paths"].values():
        for path in paths:
            assert path.stat().st_size > 0


def _schematic_ax(*, height):
    _, (ax,) = make_panel_figure(
        panels_per_row=[1], axes_heights=[height], schematic_rows=(0,)
    )
    return ax


def _assert_canvas_in_inches(*, ax):
    # Nothing of the axes drawn, and its coordinates in inches from its bottom-left corner
    width, height = ax.get_position().size * ax.get_figure(root=True).get_size_inches()
    assert not ax.axison
    assert ax.get_xlim() == pytest.approx((0, width))
    assert ax.get_ylim() == pytest.approx((0, height))


def test_plot_superspreading_schematic():
    ax = _schematic_ax(height=3.3)
    overview_plots._plot_superspreading_schematic(ax=ax)
    _assert_canvas_in_inches(ax=ax)
    width = ax.get_xlim()[1]
    # A box for each model, titled in the model's colour
    colors = [MODEL_STYLES[model_name]["color"] for model_name in ("sse", "ssi", "ssd")]
    titles = dict(
        zip(
            (
                "SSE: superspreading events",
                "SSI: superspreading individuals",
                "SSD: superspreading days",
            ),
            colors,
            strict=True,
        )
    )
    title_texts = [text for text in ax.texts if text.get_text() in titles]
    assert [text.get_text() for text in title_texts] == list(titles)
    for text in title_texts:
        assert matplotlib.colors.same_color(text.get_color(), titles[text.get_text()])
    box_idxs = [int(text.get_position()[0] // (width / 3)) for text in title_texts]
    assert box_idxs == [0, 1, 2]
    # Three infectors (the larger circles, their heads) and 12 transmissions (the smaller)
    # in each box, so the boxes differ only in how the transmissions fall
    circles = [
        patch for patch in ax.patches if type(patch) is matplotlib.patches.Circle
    ]
    head_radius, dot_radius = (
        max(circle.get_radius() for circle in circles),
        min(circle.get_radius() for circle in circles),
    )
    for radius, n_per_box in ((head_radius, 3), (dot_radius, 12)):
        box_idxs = [
            int(circle.center[0] // (width / 3))
            for circle in circles
            if circle.get_radius() == radius
        ]
        assert [box_idxs.count(box_idx) for box_idx in range(3)] == [n_per_box] * 3
    # Tinted: three events, an infector's four days and a day, as one cell for all infectors
    tints = [
        patch for patch in ax.patches if type(patch) is matplotlib.patches.Rectangle
    ]
    tint_box_idxs = [int(tint.get_x() // (width / 3)) for tint in tints]
    assert [tint_box_idxs.count(box_idx) for box_idx in range(3)] == [3, 4, 1]
    assert tints[-1].get_height() == pytest.approx(3 * tints[0].get_height())
    for tint in tints:
        assert matplotlib.colors.same_color(
            tint.get_facecolor()[:3],
            colors[int(tint.get_x() // (width / 3))],
        )


def test_plot_onset_anchoring_schematic():
    ax = _schematic_ax(height=2.6)
    overview_plots._plot_onset_anchoring_schematic(
        ax=ax, delay_means={"serial_interval": 10, "inc_period": 7.5, "tost": 2.5}
    )
    _assert_canvas_in_inches(ax=ax)
    # Arrows from left to right: a time axis for each family of models, then the delays
    arrows = [text for text in ax.texts if isinstance(text, matplotlib.text.Annotation)]
    naive_axis, so_axis, serial_interval, tost, inc_period = (
        (arrow.xyann[0], arrow.xy[0]) for arrow in arrows
    )
    assert naive_axis == so_axis
    assert naive_axis[0] < serial_interval[0] and serial_interval[1] < naive_axis[1]
    # A TOST then an incubation period span the serial interval, each as long as its mean
    assert tost[0] == serial_interval[0]
    assert tost[1] == inc_period[0]
    assert inc_period[1] == serial_interval[1]
    assert tost[1] - tost[0] == pytest.approx(
        0.25 * (serial_interval[1] - serial_interval[0])
    )
    # The naive models' arrow above the SO models', which are level
    arrow_ys = [arrow.xy[1] for arrow in arrows[2:]]
    assert arrow_ys[0] > arrow_ys[1] == arrow_ys[2]
    # Each delay labelled in its arrow's colour, which no other delay has
    label_colors = {
        text.get_text(): matplotlib.colors.to_rgba(text.get_color())
        for text in ax.texts
        if text.get_text()
    }
    arrow_colors = [arrow.arrow_patch.get_edgecolor() for arrow in arrows[2:]]
    assert arrow_colors == [
        label_colors[label]
        for label in ("Serial interval", "Onset to transmission", "Incubation period")
    ]
    assert len(set(arrow_colors)) == 3
    # The same two onsets in both families, filled as observed, with the infection
    # between them in the SO models hollow as not
    events = [line for line in ax.get_lines() if line.get_marker() == "o"]
    assert [event.get_xdata()[0] for event in events] == [
        serial_interval[0],
        serial_interval[1],
        serial_interval[0],
        tost[1],
        serial_interval[1],
    ]
    assert [event.get_markerfacecolor() for event in events] == [
        "black",
        "black",
        "black",
        "white",
        "black",
    ]
    # On their family's time axis, the naive models' above the SO models'
    naive_axis_y, so_axis_y = (arrow.xy[1] for arrow in arrows[:2])
    assert naive_axis_y > so_axis_y
    assert [event.get_ydata()[0] for event in events] == [
        naive_axis_y,
        naive_axis_y,
        so_axis_y,
        so_axis_y,
        so_axis_y,
    ]
