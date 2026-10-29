# Note that AI tools were used to generate tests
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
    rng = np.random.default_rng(2)
    risk_days = inputs["risk_days"]
    for paths in inputs["results_paths_by_model"].values():
        paths["risk"].parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "chain": np.repeat([0, 1], 100),
                "draw": np.tile(np.arange(100), 2),
                **{
                    param_name: rng.lognormal(sigma=0.3, size=200)
                    for param_name in inputs["param_names"]
                },
            }
        ).to_csv(paths["posterior"], index=False)
        pd.DataFrame({"log_evidence": [rng.normal()]}).to_csv(
            paths["evidence"], index=False
        )
        pd.DataFrame(
            {
                "day": risk_days,
                **{
                    f"risk_{risk_type}": rng.uniform(size=len(risk_days))
                    for risk_type in ("cases", "transmission", "sustained", "reports")
                },
            }
        ).to_csv(paths["risk"], index=False)
    monkeypatch.setattr(
        model_comparison_plots,
        "get_inputs_model_comparison_plots",
        lambda *, comparison_spec_name: inputs,
    )

    model_comparison_plots.make_plots(comparison_spec_name=comparison_spec_name)

    for paths in inputs["fig_paths"].values():
        for path in paths:
            assert path.stat().st_size > 0
