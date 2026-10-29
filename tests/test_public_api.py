# Note that AI tools were used to generate tests
import endoutbreakhet


def test_public_api():
    assert set(endoutbreakhet.__all__) == {
        "DELAY_FIRST_LAGS",
        "DelayDistribution",
        "compute_log_evidence",
        "compute_model_probabilities",
        "compute_risk",
        "discretise_cori",
        "fit_model",
        "gamma_dist_from_mean_sd",
        "lognormal_params_from_median_quantile_025",
        "run_pipeline",
    }
