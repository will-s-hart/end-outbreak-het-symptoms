# Note that AI tools were used to generate tests
import multiprocessing
import os
import types

import numpy as np
import pandas as pd
import pytensor
import pytest
import xarray as xr
from pytensor.configparser import parse_config_string

from endoutbreakhet import pipeline

# Seeded with the year of the first Pipe Masters
_SEED = 1971


@pytest.fixture
def pipeline_inputs(*, toy_inputs):
    # Toy data run to day 17. The comparison day (10) is not a risk day, and the risk days
    # are first, second and third reef: third reef (day 19) only breaks once the flat spell
    # after the data is padded in
    return toy_inputs | {
        "model_name": "poisson",
        "comparison_day": 10,
        "risk_days": np.array([13, 16, 19]),
        "fit_kwargs": {
            "chains": 2,
            "tune": 50,
            "draws": 50,
            "compute_convergence_checks": False,
        },
        "n_evidence_proposal_samples": 200,
        "rng": np.random.default_rng(_SEED),
    }


@pytest.mark.parametrize(
    ("model_name", "risk_types"),
    [
        ("poisson", ("cases", "transmission", "sustained", "reports")),
        # DLO risks of sustained transmission and reports aren't implemented
        ("dlo", ("cases", "transmission")),
    ],
)
def test_fits_data_through_each_day(
    *, pipeline_inputs, model_name, risk_types, monkeypatch
):
    fits = []
    risk_calls = []
    evidence_calls = []
    risk_offsets = {
        "cases": 0.25,
        "transmission": 0.5,
        "sustained": 0.125,
        "reports": 0.375,
    }

    def _paddle_out(*, reported_incidence_vec, rng, reporting_prob, **kwargs):
        # Stub fit that paddles out but never gets barrelled, just noting the set it saw
        fits.append(
            {
                "reported_incidence_vec": reported_incidence_vec,
                "rng": rng,
                "reporting_prob": reporting_prob,
            }
        )
        return types.SimpleNamespace(n_days=len(reported_incidence_vec))

    def _fake_risk(*, trace, risk_type, reporting_prob, **kwargs):
        risk_calls.append({"risk_type": risk_type, "reporting_prob": reporting_prob})
        risk = trace.n_days + risk_offsets[risk_type]
        return {"risk": risk, "risk_mcse": risk / 100, "risk_rhat": 1 + risk / 1000}

    def _fake_fit_diagnostics(*, trace):
        return {"n_divergences": trace.n_days, "max_rhat_rep_no_before": 1.0}

    def _fake_log_evidence(*, trace, reporting_prob, **kwargs):
        evidence_calls.append({"trace": trace, "reporting_prob": reporting_prob})
        return {"log_evidence": -trace.n_days}

    monkeypatch.setattr(pipeline, "fit_model", _paddle_out)
    monkeypatch.setattr(pipeline, "compute_risk", _fake_risk)
    monkeypatch.setattr(pipeline, "compute_fit_diagnostics", _fake_fit_diagnostics)
    monkeypatch.setattr(pipeline, "compute_log_evidence", _fake_log_evidence)
    outputs = pipeline.run_pipeline(
        **(pipeline_inputs | {"model_name": model_name}), reporting_prob=0.7, n_jobs=1
    )

    days = [10, 13, 16, 19]
    padded_incidence_vec = np.concatenate(
        [pipeline_inputs["reported_incidence_vec"], [0, 0]]
    )
    # Latest day first, as the longest fits shouldn't be the last out, each day riding its
    # own stream of random numbers, the one spawned for it in day order
    assert len(fits) == len(days)
    day_rngs = np.random.default_rng(_SEED).spawn(len(days))
    for fit, day, day_rng in zip(fits, days[::-1], day_rngs[::-1], strict=True):
        np.testing.assert_array_equal(
            fit["reported_incidence_vec"], padded_incidence_vec[: day + 1]
        )
        assert (
            fit["reported_incidence_vec"].dtype
            == pipeline_inputs["reported_incidence_vec"].dtype
        )
        assert fit["rng"].random() == day_rng.random()

    # Each risk with its Monte Carlo standard error and R-hat beside it
    risk_vecs = {
        risk_type: np.array(days) + 1 + risk_offsets[risk_type]
        for risk_type in risk_types
    }
    pd.testing.assert_frame_equal(
        outputs["risk_df"],
        pd.DataFrame(
            {
                "day": days,
                **{
                    column: values
                    for risk_type, risk_vec in risk_vecs.items()
                    for column, values in [
                        (f"risk_{risk_type}", risk_vec),
                        (f"risk_mcse_{risk_type}", risk_vec / 100),
                        (f"risk_rhat_{risk_type}", 1 + risk_vec / 1000),
                    ]
                },
            }
        ),
        check_dtype=False,
    )
    # And every day's fit diagnostics
    pd.testing.assert_frame_equal(
        outputs["diagnostics_df"],
        pd.DataFrame(
            {
                "day": days,
                "n_divergences": [day + 1 for day in days],
                "max_rhat_rep_no_before": 1.0,
            }
        ),
        check_dtype=False,
    )
    # Like Gerry Lopez at Pipe, the evidence is taken once, on the comparison day only
    assert len(evidence_calls) == 1
    assert evidence_calls[0]["trace"] is outputs["trace"]
    assert outputs["trace"].n_days == 11
    assert outputs["evidence"] == {"log_evidence": -11}
    # The fits, risks and evidence all see the same reporting probability
    assert {
        call["reporting_prob"] for call in [*fits, *risk_calls, *evidence_calls]
    } == {0.7}


def test_parallel_matches_serial(*, pipeline_inputs):
    # Pipe and Backdoor break left and right off the same peak: from the same seed, serial
    # and parallel runs must ride the identical wave
    outputs_serial = pipeline.run_pipeline(
        **(pipeline_inputs | {"rng": np.random.default_rng(_SEED)}), n_jobs=1
    )
    outputs_parallel = pipeline.run_pipeline(
        **(pipeline_inputs | {"rng": np.random.default_rng(_SEED)}), n_jobs=2
    )

    pd.testing.assert_frame_equal(
        outputs_serial["risk_df"], outputs_parallel["risk_df"]
    )
    pd.testing.assert_frame_equal(
        outputs_serial["diagnostics_df"], outputs_parallel["diagnostics_df"]
    )
    assert outputs_serial["evidence"] == outputs_parallel["evidence"]
    xr.testing.assert_equal(
        outputs_serial["trace"].posterior.to_dataset(),
        outputs_parallel["trace"].posterior.to_dataset(),
    )
    risk_types = ["cases", "transmission", "sustained", "reports"]
    assert list(outputs_serial["risk_df"].columns) == [
        "day",
        *(
            f"{name}_{risk_type}"
            for risk_type in risk_types
            for name in ("risk", "risk_mcse", "risk_rhat")
        ),
    ]
    risks = outputs_serial["risk_df"][
        [f"risk_{risk_type}" for risk_type in risk_types]
    ].to_numpy()
    assert np.all((risks >= 0) & (risks <= 1))
    assert list(outputs_serial["diagnostics_df"]["day"]) == [10, 13, 16, 19]


def test_parallel_maps_isolate_real_numba_caches(*, tmp_path, monkeypatch):
    # Workers mustn't share a Numba cache with the caller, other runs or each other: on a
    # cluster network filesystem, reading a cache index that another node is replacing fails
    # with ESTALE, and on any filesystem, workers saving different signatures of one op at
    # once can leave one's entry pointing at the other's code. The caller's base is set in a
    # .pytensorrc, as on the cluster
    rc_base_compiledir = tmp_path / "rc_base_compiledir"
    rc_path = tmp_path / "pytensorrc"
    rc_path.write_text(f"[global]\nbase_compiledir = {rc_base_compiledir}\n")
    monkeypatch.setenv("PYTENSORRC", str(rc_path))
    monkeypatch.setenv("PYTENSOR_FLAGS", "exception_verbosity=high,numba__cache=True")

    def _compile(*, barrier):
        import pytensor.tensor as pt
        from pytensor.link.numba import cache

        # Neither task can finish until both workers have started
        barrier.wait(timeout=60)

        x = pt.dvector("x")
        fn = pytensor.function([x], pt.exp(x) + 1, mode="NUMBA")
        np.testing.assert_allclose(fn([0.0, 1.0]), np.exp([0.0, 1.0]) + 1)

        return {
            "pid": os.getpid(),
            "base_compiledir": pytensor.config.base_compiledir,
            "compiledir": pytensor.config.compiledir,
            "numba_cache_path": cache.NUMBA_CACHE_PATH,
            "has_cache_files": any(cache.NUMBA_CACHE_PATH.glob("*.nbi")),
            "has_base_cache_files": any(
                (pytensor.config.base_compiledir / "numba").glob("*.nbi")
            ),
            "exception_verbosity": pytensor.config.exception_verbosity,
        }

    with multiprocessing.Manager() as manager:
        barrier = manager.Barrier(2)

        def _map():
            return pipeline._parallel_map(
                func=_compile,
                kwargs_list=[{"barrier": barrier}] * 2,
                n_jobs=2,
                desc="Checking Numba caches",
            )

        first = _map()
        second = _map()

    base_compiledirs = []
    for caches in (first, second):
        (base_compiledir,) = {cache["base_compiledir"] for cache in caches}
        base_compiledirs.append(base_compiledir)
        assert len({cache["pid"] for cache in caches}) == 2
        assert base_compiledir != pytensor.config.base_compiledir
        assert not base_compiledir.exists()
        # Each worker caches in a dir of its own within the base, leaving the base's default
        # unused
        assert len({cache["numba_cache_path"].parent for cache in caches}) == 2
        for cache in caches:
            worker_dir = cache["numba_cache_path"].parent
            assert worker_dir.parent == base_compiledir
            assert cache["has_cache_files"]
            assert not cache["has_base_cache_files"]
            # PyTensor resolves the compile dir
            assert cache["compiledir"] == worker_dir.resolve() / "compiledir"
            # The caller's other flags reach the workers
            assert cache["exception_verbosity"] == "high"
    assert not rc_base_compiledir.exists()
    # The next session paddles out with fresh workers, not ones whose caches are gone
    assert base_compiledirs[0] != base_compiledirs[1]
    assert {cache["pid"] for cache in first}.isdisjoint(
        cache["pid"] for cache in second
    )
    assert os.environ["PYTENSOR_FLAGS"] == "exception_verbosity=high,numba__cache=True"


@pytest.mark.parametrize("flags", [None, "base_compiledir=/elsewhere"])
@pytest.mark.parametrize(
    "base_compiledir", ["/tmp/a,b#c", r"C:\Users\a b\Temp", "/tmp/it's"]
)
def test_new_workers_flags_keep_awkward_base_compiledirs_whole(
    *, flags, base_compiledir, monkeypatch
):
    if flags is None:
        monkeypatch.delenv("PYTENSOR_FLAGS", raising=False)
    else:
        monkeypatch.setenv("PYTENSOR_FLAGS", flags)

    with (
        pytest.raises(RuntimeError, match="Wipeout"),
        pipeline._base_compiledir_for_new_workers(base_compiledir=base_compiledir),
    ):
        # As PyTensor will read the flags in a new worker, overriding any base the caller
        # set
        assert (
            parse_config_string(os.environ["PYTENSOR_FLAGS"])["base_compiledir"]
            == base_compiledir
        )
        raise RuntimeError("Wipeout")
    # The caller's flags are restored, even after a wipeout
    assert os.environ.get("PYTENSOR_FLAGS") == flags


def test_serial_map_keeps_the_callers_pytensor_caches():
    def _compiledir():
        return pytensor.config.compiledir

    assert pipeline._parallel_map(
        func=_compiledir, kwargs_list=[{}], n_jobs=1, desc="Checking caches"
    ) == [pytensor.config.compiledir]
