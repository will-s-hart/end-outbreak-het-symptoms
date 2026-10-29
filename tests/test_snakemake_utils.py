# Note that AI tools were used to generate tests
import numpy as np
import pytest

from scripts.snakemake_utils import as_param


def test_as_param_tracks_values_a_repr_would_hide():
    # numpy reprs arrays to 8 significant digits, and a Generator's repr is its address
    assert repr(as_param(inputs={"x": np.array([1.0])})) != repr(
        as_param(inputs={"x": np.array([1.0 + 1e-12])})
    )
    assert repr(as_param(inputs={"rng": np.random.default_rng(seed=1)})) != repr(
        as_param(inputs={"rng": np.random.default_rng(seed=2)})
    )
    assert repr(as_param(inputs={"rng": np.random.default_rng(seed=1)})) == repr(
        as_param(inputs={"rng": np.random.default_rng(seed=1)})
    )


def test_as_param_ignores_dict_order():
    assert repr(as_param(inputs={"a": 1, "b": {"c": 2, "d": 3}})) == repr(
        as_param(inputs={"b": {"d": 3, "c": 2}, "a": 1})
    )


def test_as_param_raises_for_unknown_type():
    # Snakemake would silently skip it, leaving the value untracked
    with pytest.raises(TypeError):
        as_param(inputs={"x": object()})
