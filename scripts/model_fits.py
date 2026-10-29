import argparse

import pandas as pd

from endoutbreakhet import DelayDistribution, run_pipeline
from scripts.inputs import MODEL_SPECS, get_inputs_model_fits


def run_analyses(*, model_spec_name):
    inputs = get_inputs_model_fits(model_spec_name=model_spec_name)
    # round_trip reads back exactly the values discretised
    delay_distributions = {
        delay_name: DelayDistribution.from_df(
            df=pd.read_csv(path, float_precision="round_trip")
        )
        for delay_name, path in inputs["delay_distributions_results_paths"].items()
    }
    outputs = run_pipeline(
        model_name=inputs["model_name"],
        reported_incidence_vec=inputs["reported_incidence_vec"],
        switch_day=inputs["switch_day"],
        comparison_day=inputs["comparison_day"],
        risk_days=inputs["risk_days"],
        delay_distributions=delay_distributions,
        priors=inputs["priors"],
        fit_kwargs=inputs["fit_kwargs"],
        n_evidence_proposal_samples=inputs["n_evidence_proposal_samples"],
        rng=inputs["rng"],
        reporting_prob=inputs["reporting_prob"],
    )

    results_paths = inputs["results_paths"]
    results_paths["risk"].parent.mkdir(parents=True, exist_ok=True)
    outputs["risk_df"].to_csv(results_paths["risk"], index=False)
    pd.DataFrame([outputs["evidence"]]).to_csv(results_paths["evidence"], index=False)
    # Scalar parameter samples, tracked so plots can be remade without the (untracked)
    # full trace
    posterior = outputs["trace"].posterior.to_dataset()
    scalar_names = [
        name for name, values in posterior.items() if values.dims == ("chain", "draw")
    ]
    posterior[scalar_names].to_dataframe().to_csv(results_paths["posterior"])
    outputs["trace"].to_netcdf(results_paths["trace"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Fit a model spec to the data available up to each day, saving the risks, "
            "and the posterior and evidence on the comparison day."
        )
    )
    parser.add_argument("--spec", required=True, choices=MODEL_SPECS)
    args = parser.parse_args()

    run_analyses(model_spec_name=args.spec)
