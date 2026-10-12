# Note that AI tools were used to generate tests
import pytest

from endoutbreakhet.inference import _build_pymc_model
from scripts.inputs import MODEL_SPECS


@pytest.mark.parametrize(
    "spec_name",
    [
        pytest.param(
            spec_name,
            marks=pytest.mark.xfail(
                raises=ValueError,
                strict=True,
                reason="Fixed dispersion prior not yet supported",
            ),
        )
        if spec_name.endswith("_fixed_dispersion")
        else spec_name
        for spec_name in MODEL_SPECS
    ],
)
def test_model_specs_build(*, spec_name, toy_inputs):
    spec = MODEL_SPECS[spec_name]
    _build_pymc_model(
        model_name=spec["model_name"],
        incidence_vec=toy_inputs["incidence_vec"],
        switch_day=toy_inputs["switch_day"],
        delay_distributions=toy_inputs["delay_distributions"],
        priors=spec["priors"],
    )
