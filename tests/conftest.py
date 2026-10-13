# Note that AI tools were used to generate tests
import matplotlib
import numpy as np
import pytest

# Non-interactive backend, so plotting tests don't need a display
matplotlib.use("Agg")

MODEL_NAMES = ["poisson", "sse", "ssi", "dlo", "poisson_so", "sse_so", "ssi_so"]


@pytest.fixture
def toy_inputs():
    # Last case on day 11, so force of infection (lags 1-4) is zero from day 16 and force
    # of disease (lags 1-5) is zero from day 17
    return {
        "incidence_vec": np.array(
            [2, 1, 3, 2, 4, 1, 2, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        ),
        "switch_day": 6,
        "delay_distributions": {
            "serial_interval": np.array([0.2, 0.4, 0.3, 0.1]),
            "inc_period": np.array([0.5, 0.3, 0.2]),
            "tost": np.array([0.3, 0.5, 0.2]),
        },
        "priors": {
            "rep_no_before": {"dist": "lognormal", "median": 1.0, "quantile_025": 0.2},
            "rep_no_after": {"dist": "lognormal", "median": 1.0, "quantile_025": 0.2},
            "dispersion": {"dist": "lognormal", "median": 0.18, "quantile_025": 0.018},
        },
    }


@pytest.fixture
def toy_inputs_fixed_dispersion(*, toy_inputs):
    return toy_inputs | {
        "priors": toy_inputs["priors"]
        | {"dispersion": {"dist": "fixed", "value": 0.18}}
    }


@pytest.fixture(params=MODEL_NAMES)
def model_name(*, request):
    return request.param
