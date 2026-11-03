# Note that AI tools were used to generate tests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scripts import model_comparison_plots
from scripts.inputs import COMPARISON_SPECS, get_inputs_model_comparison_plots


@pytest.mark.parametrize("comparison_spec_name", COMPARISON_SPECS)
def test_make_plots_writes_figures(*, comparison_spec_name, tmp_path, monkeypatch):
    # Real inputs, with the results and figures redirected into tmp_path and fake results
    # written there
    inputs = get_inputs_model_comparison_plots(
        comparison_spec_name=comparison_spec_name
    )
    inputs["results_paths_by_model"] = {
        model_name: {key: tmp_path / path for key, path in paths.items()}
        for model_name, paths in inputs["results_paths_by_model"].items()
    }
    inputs["fig_paths"] = {
        fig_name: [tmp_path / path for path in paths]
        for fig_name, paths in inputs["fig_paths"].items()
    }
    # Only the results the inputs list, so the plots read nothing else
    rng = np.random.default_rng(2)
    risk_days = inputs["risk_days"]
    param_names = {
        param_name
        for figure in inputs["figures"].values()
        for param_name in figure["param_names"]
    }
    days = np.arange(1, len(inputs["reported_incidence_vec"]))
    fake_dfs = {
        "posterior": lambda: pd.DataFrame(
            {
                "chain": np.repeat([0, 1], 100),
                "draw": np.tile(np.arange(100), 2),
                **{
                    param_name: rng.lognormal(sigma=0.3, size=200)
                    for param_name in param_names
                },
            }
        ),
        "evidence": lambda: pd.DataFrame({"log_evidence": [rng.normal()]}),
        # A pmf of 0-2 unreported cases on each day after the first
        "unreported_incidence": lambda: pd.DataFrame(
            {
                "day": np.repeat(days, 3),
                "n_unreported": np.tile([0, 1, 2], len(days)),
                "probability": rng.dirichlet(np.ones(3), size=len(days)).ravel(),
            }
        ),
        "risk": lambda: pd.DataFrame(
            {
                "day": risk_days,
                **{
                    f"risk_{risk_type}": rng.uniform(size=len(risk_days))
                    for risk_type in ("cases", "transmission", "sustained", "reports")
                },
            }
        ),
    }
    for paths in inputs["results_paths_by_model"].values():
        for result_name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            fake_dfs[result_name]().to_csv(path, index=False)
    monkeypatch.setattr(
        model_comparison_plots,
        "get_inputs_model_comparison_plots",
        lambda *, comparison_spec_name: inputs,
    )

    model_comparison_plots.make_plots(comparison_spec_name=comparison_spec_name)

    for paths in inputs["fig_paths"].values():
        for path in paths:
            assert path.stat().st_size > 0


@pytest.mark.parametrize(
    ("figure", "expected_rows", "expected_labels", "expected_calls"),
    [
        # Model probabilities in a box on R_after's densities, labelled in turn
        (
            {
                "param_names": ("rep_no_before", "rep_no_after"),
                "unreported": True,
                "risk_types": ("cases", "transmission"),
            },
            [[0, 1], [2], [3]],
            ["A.", "B.", "D.", "E."],
            [
                ("densities", 0, ("rep_no_before", False, True)),
                ("densities", 1, ("rep_no_after", False, False)),
                ("model_probabilities_inset", 1, ("C.",)),
                ("unreported", 2, ()),
                ("risk", 3, (("cases", "transmission"),)),
            ],
        ),
        # A column per density panel, then model probabilities in a panel of their own
        (
            {
                "param_names": ("rep_no_before", "rep_no_after", "dispersion"),
                "unreported": False,
                "risk_types": ("cases",),
            },
            [[0, 1, 2], [3, 4]],
            ["A.", "B.", "C.", "D.", "E."],
            [
                ("densities", 0, ("rep_no_before", False, True)),
                ("densities", 1, ("rep_no_after", False, False)),
                ("densities", 2, ("dispersion", True, False)),
                ("model_probabilities", 3, ()),
                ("risk", 4, (("cases",),)),
            ],
        ),
        (
            {"param_names": (), "unreported": True, "risk_types": ("cases",)},
            [[0], [1]],
            ["A.", "B."],
            [("unreported", 0, ()), ("risk", 1, (("cases",),))],
        ),
        # A lone panel is unlabelled
        (
            {"param_names": (), "unreported": False, "risk_types": ("sustained",)},
            [[0]],
            [],
            [("risk", 0, (("sustained",),))],
        ),
    ],
)
def test_make_figure_layout(
    *, figure, expected_rows, expected_labels, expected_calls, monkeypatch
):
    # The panels' plotting is recorded rather than done, as (panel kind, index of its axes,
    # what distinguishes it)
    calls = []

    def _record(*, kind, arg_names):
        def _plot(*, ax, **kwargs):
            calls.append((kind, ax, tuple(kwargs[arg_name] for arg_name in arg_names)))

        return _plot

    for plot_name, kind, arg_names in (
        ("plot_densities", "densities", ("param_name", "log_x", "legend")),
        ("plot_model_probabilities", "model_probabilities", ()),
        ("plot_model_probabilities_inset", "model_probabilities_inset", ("label",)),
        ("plot_unreported_incidence", "unreported", ()),
        ("plot_risk_incidence_overlay", "risk", ("risk_types",)),
    ):
        monkeypatch.setattr(
            model_comparison_plots,
            plot_name,
            _record(kind=kind, arg_names=arg_names),
        )

    fig = model_comparison_plots._make_figure(
        **figure,
        df_by_model_by_result={
            "posterior": {},
            "evidence": {"sse": pd.DataFrame({"log_evidence": [0.0]})},
            "unreported_incidence": {},
            "risk": {},
        },
        priors={},
        unreported_plot_end_date=None,
        incidence_kwargs={},
    )

    # Axes are added row by row, top to bottom
    rows = {}
    for ax_idx, ax in enumerate(fig.axes):
        rows.setdefault(round(ax.get_position().y0, 6), []).append(ax_idx)
    assert list(rows.values()) == expected_rows
    # Every row is filled, the unreported cases and risks taking whatever the rest leave
    row_right = max(ax.get_position().x1 for ax in fig.axes)
    for row in expected_rows:
        assert fig.axes[row[-1]].get_position().x1 == pytest.approx(row_right)
    assert [text.get_text() for text in fig.texts] == expected_labels
    assert [
        (kind, fig.axes.index(ax), args) for kind, ax, args in calls
    ] == expected_calls
    plt.close(fig)
