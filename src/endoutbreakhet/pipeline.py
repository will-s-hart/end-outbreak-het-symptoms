import contextlib
import functools
import logging
import os
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytensor
from joblib import Parallel, delayed, parallel_config
from pytensor.link.numba import cache as pytensor_numba_cache
from tqdm import tqdm

from endoutbreakhet.evidence import compute_log_evidence
from endoutbreakhet.inference import compute_fit_diagnostics, fit_model
from endoutbreakhet.risk import compute_risk
from endoutbreakhet.utils import validate_delay_distributions


def run_pipeline(
    *,
    model_name,
    reported_incidence_vec,
    switch_day,
    comparison_day,
    risk_days,
    delay_distributions,
    priors,
    fit_kwargs,
    n_evidence_proposal_samples,
    rng,
    reporting_prob=1.0,
    n_jobs=-1,
):
    # Fits the model to the onsets up to and including each day in risk_days and
    # comparison_day (taking onsets after the end of reported_incidence_vec to be zero),
    # computing the fit's diagnostics and every risk the model has after that day (DLO risks
    # of sustained transmission and reports aren't implemented), each with its Monte Carlo
    # standard error and R-hat. The trace and log evidence are kept for comparison_day only,
    # where specs are compared
    validate_delay_distributions(
        model_name=model_name, delay_distributions=delay_distributions
    )
    days = np.union1d(risk_days, [comparison_day])
    reported_incidence_vec = np.pad(
        reported_incidence_vec, (0, max(0, days[-1] + 1 - len(reported_incidence_vec)))
    )
    risk_types = (
        ("cases", "transmission")
        if model_name == "dlo"
        else ("cases", "transmission", "sustained", "reports")
    )
    analyse_day = functools.partial(
        _analyse_day,
        model_name=model_name,
        switch_day=switch_day,
        delay_distributions=delay_distributions,
        priors=priors,
        reporting_prob=reporting_prob,
        risk_types=risk_types,
        # Chains run sequentially within each fit, as the fits run in parallel
        fit_kwargs=fit_kwargs | {"cores": 1, "progressbar": False},
        n_evidence_proposal_samples=n_evidence_proposal_samples,
    )
    day_outputs = _parallel_map(
        func=analyse_day,
        kwargs_list=[
            {
                "reported_incidence_vec": reported_incidence_vec[: day + 1],
                "rng": day_rng,
                "keep_trace_and_evidence": day == comparison_day,
            }
            for day, day_rng in zip(days, rng.spawn(len(days)), strict=True)
        ],
        n_jobs=n_jobs,
        desc="Fitting model to data available up to each day",
    )
    comparison_output = day_outputs[np.searchsorted(days, comparison_day)]

    risk_df = pd.DataFrame(
        [
            {"day": day}
            | {
                f"{name}_{risk_type}": value
                for risk_type, risk in output["risk_by_type"].items()
                for name, value in risk.items()
            }
            for day, output in zip(days, day_outputs, strict=True)
        ]
    )
    diagnostics_df = pd.DataFrame(
        [
            {"day": day} | output["diagnostics"]
            for day, output in zip(days, day_outputs, strict=True)
        ]
    )
    return {
        "risk_df": risk_df,
        "diagnostics_df": diagnostics_df,
        "trace": comparison_output["trace"],
        "evidence": comparison_output["evidence"],
    }


def _analyse_day(
    *,
    model_name,
    reported_incidence_vec,
    switch_day,
    delay_distributions,
    priors,
    reporting_prob,
    risk_types,
    fit_kwargs,
    n_evidence_proposal_samples,
    rng,
    keep_trace_and_evidence,
):
    trace = fit_model(
        model_name=model_name,
        reported_incidence_vec=reported_incidence_vec,
        switch_day=switch_day,
        delay_distributions=delay_distributions,
        priors=priors,
        rng=rng,
        reporting_prob=reporting_prob,
        **fit_kwargs,
    )
    output = {
        "diagnostics": compute_fit_diagnostics(trace=trace),
        "risk_by_type": {
            risk_type: compute_risk(
                model_name=model_name,
                reported_incidence_vec=reported_incidence_vec,
                switch_day=switch_day,
                delay_distributions=delay_distributions,
                trace=trace,
                reporting_prob=reporting_prob,
                risk_type=risk_type,
            )
            for risk_type in risk_types
        },
    }
    if not keep_trace_and_evidence:
        return output | {"trace": None, "evidence": None}

    evidence = compute_log_evidence(
        model_name=model_name,
        reported_incidence_vec=reported_incidence_vec,
        switch_day=switch_day,
        delay_distributions=delay_distributions,
        priors=priors,
        trace=trace,
        n_proposal_samples=n_evidence_proposal_samples,
        rng=rng,
        reporting_prob=reporting_prob,
    )
    return output | {"trace": trace, "evidence": evidence}


def _parallel_map(*, func, kwargs_list, n_jobs, desc):
    # func(**kwargs) for each kwargs in kwargs_list, in order, over n_jobs worker processes
    # with single-threaded numeric libraries and PyTensor caches of their own, in a temporary
    # dir (node-local on the cluster) removed afterwards. With n_jobs=1, joblib runs tasks in
    # the calling process, which keeps its caches
    with (
        tempfile.TemporaryDirectory(
            prefix="endoutbreakhet_pytensor_", ignore_cleanup_errors=True
        ) as base_compiledir,
        _base_compiledir_for_new_workers(base_compiledir=base_compiledir),
        parallel_config(backend="loky", inner_max_num_threads=1),
    ):
        return list(
            tqdm(
                Parallel(
                    n_jobs=n_jobs,
                    return_as="generator",
                    # A new initializer for each map also stops loky reusing an earlier
                    # map's workers, whose caches have been removed
                    initializer=functools.partial(
                        _prepare_worker, base_compiledir=base_compiledir
                    ),
                )(delayed(func)(**kwargs) for kwargs in kwargs_list),
                total=len(kwargs_list),
                desc=desc,
            )
        )


@contextlib.contextmanager
def _base_compiledir_for_new_workers(*, base_compiledir):
    # PyTensor's caches default to dirs under `config.base_compiledir`, its Numba linker's
    # (created on import) to `base_compiledir/numba`. Each worker moves its caches to dirs of
    # its own (see _prepare_worker), but only once PyTensor is imported, so as a safeguard
    # workers get the map's base too, keeping anything cached by default out of the
    # caller's. That may be on
    # a cluster network filesystem, where a worker reading a cache index that a worker on
    # another node is replacing fails with ESTALE. The base is fixed when PyTensor is
    # imported, which in a worker comes before any initializer runs (unpickling one imports
    # this package), so it's passed in PYTENSOR_FLAGS, which workers inherit when spawned.
    # PYTENSOR_FLAGS overrides .pytensorrc and its later flags override earlier ones, so this
    # overrides the base however the caller set it, keeping other flags. PyTensor splits the
    # flags with a POSIX shlex, so the path is single-quoted (closing and reopening the
    # quotes around any quote in it) to keep backslashes, commas and #s
    flags = os.environ.get("PYTENSOR_FLAGS")
    quoted_base_compiledir = "'" + str(base_compiledir).replace("'", "'\"'\"'") + "'"
    os.environ["PYTENSOR_FLAGS"] = ",".join(
        filter(None, [flags, f"base_compiledir={quoted_base_compiledir}"])
    )
    try:
        yield
    finally:
        if flags is None:
            del os.environ["PYTENSOR_FLAGS"]
        else:
            os.environ["PYTENSOR_FLAGS"] = flags


def _prepare_worker(*, base_compiledir):
    # Give this worker PyTensor caches of its own within the map's base. The compile FileLock
    # is taken on `config.compiledir/.lock` (read dynamically per compilation), so sharing
    # one compile dir across workers would serialise the C-compilation phase and defeat the
    # parallelism. Numba saves a function's cache index by an unlocked read-modify-write
    # (numba#10862), so workers saving different signatures of one op at once (e.g. Alloc
    # with int8 and int16 shapes) can leave one signature's entry pointing at the other's
    # compiled code, which a later compile loads and fails to type. PyTensor's cache locator
    # hands NUMBA_CACHE_PATH to each cached function as it's jitted, so this precedes any
    # compilation, and never creates it. It sits beside the compile dir, as PyTensor removes
    # empty dirs within that. The worker's dir is made unique by mkdtemp, not named by pid,
    # since loky replaces workers mid-map (e.g. on memory growth) and a pid can recur.
    # Quieten PyMC's per-fit INFO logging, which would otherwise be repeated for every task
    worker_dir = pathlib.Path(tempfile.mkdtemp(prefix="worker_", dir=base_compiledir))
    (worker_dir / "compiledir").mkdir()
    (worker_dir / "numba").mkdir()
    pytensor.config.compiledir = worker_dir / "compiledir"
    pytensor_numba_cache.NUMBA_CACHE_PATH = worker_dir / "numba"
    logging.getLogger("pymc").setLevel(logging.WARNING)
