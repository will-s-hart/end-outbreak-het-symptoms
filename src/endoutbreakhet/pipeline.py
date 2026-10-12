import atexit
import functools
import logging
import os
import pathlib
import shutil

import numpy as np
import pandas as pd
import pytensor
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

from endoutbreakhet.evidence import compute_log_evidence
from endoutbreakhet.inference import fit_model
from endoutbreakhet.risk import compute_risk


def run_pipeline(
    *,
    model_name,
    incidence_vec,
    switch_day,
    comparison_day,
    risk_days,
    delay_distributions,
    priors,
    fit_kwargs,
    n_evidence_proposal_samples,
    rng,
    n_jobs=-1,
):
    # Fits the model to the onsets up to and including each day in risk_days and
    # comparison_day (taking onsets after the end of incidence_vec to be zero), and computes
    # the risks of cases and transmission after that day. The trace and log evidence are
    # kept for comparison_day only, where specs are compared
    days = np.union1d(risk_days, [comparison_day])
    incidence_vec = np.pad(
        incidence_vec, (0, max(0, days[-1] + 1 - len(incidence_vec)))
    )
    analyse_day = functools.partial(
        _analyse_day,
        model_name=model_name,
        switch_day=switch_day,
        delay_distributions=delay_distributions,
        priors=priors,
        # Chains run sequentially within each fit, as the fits run in parallel
        fit_kwargs=fit_kwargs | {"cores": 1, "progressbar": False},
        n_evidence_proposal_samples=n_evidence_proposal_samples,
    )
    day_outputs = _parallel_map(
        func=analyse_day,
        kwargs_list=[
            {
                "incidence_vec": incidence_vec[: day + 1],
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
        {
            "day": days,
            "risk_cases": [output["risk_cases"] for output in day_outputs],
            "risk_transmission": [
                output["risk_transmission"] for output in day_outputs
            ],
        }
    )
    return {
        "risk_df": risk_df,
        "trace": comparison_output["trace"],
        "evidence": comparison_output["evidence"],
    }


def _analyse_day(
    *,
    model_name,
    incidence_vec,
    switch_day,
    delay_distributions,
    priors,
    fit_kwargs,
    n_evidence_proposal_samples,
    rng,
    keep_trace_and_evidence,
):
    trace = fit_model(
        model_name=model_name,
        incidence_vec=incidence_vec,
        switch_day=switch_day,
        delay_distributions=delay_distributions,
        priors=priors,
        rng=rng,
        **fit_kwargs,
    )
    output = {
        f"risk_{risk_type}": compute_risk(
            model_name=model_name,
            incidence_vec=incidence_vec,
            switch_day=switch_day,
            delay_distributions=delay_distributions,
            trace=trace,
            risk_type=risk_type,
        )
        for risk_type in ("cases", "transmission")
    }
    if not keep_trace_and_evidence:
        return output | {"trace": None, "evidence": None}

    evidence = compute_log_evidence(
        model_name=model_name,
        incidence_vec=incidence_vec,
        switch_day=switch_day,
        delay_distributions=delay_distributions,
        priors=priors,
        trace=trace,
        n_proposal_samples=n_evidence_proposal_samples,
        rng=rng,
    )
    return output | {"trace": trace, "evidence": evidence}


def _parallel_map(*, func, kwargs_list, n_jobs, desc):
    # func(**kwargs) for each kwargs in kwargs_list, in order, over n_jobs worker processes
    # with single-threaded numeric libraries
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return list(
            tqdm(
                Parallel(n_jobs=n_jobs, return_as="generator")(
                    delayed(_call_in_worker)(
                        func=func, kwargs=kwargs, parent_pid=os.getpid()
                    )
                    for kwargs in kwargs_list
                ),
                total=len(kwargs_list),
                desc=desc,
            )
        )


def _call_in_worker(*, func, kwargs, parent_pid):
    # With n_jobs=1, joblib runs tasks in the calling process, whose environment is left alone
    if os.getpid() != parent_pid:
        _prepare_worker()
    return func(**kwargs)


# Guards the one-time per-worker environment setup below.
_WORKER_ENV_READY = False


def _prepare_worker():
    # Isolate this worker's PyTensor compile dir: the compile FileLock is taken on
    # `config.compiledir/.lock` (read dynamically per compilation), so sharing one dir across
    # workers would serialise the C-compilation phase and defeat the parallelism. Quieten
    # PyMC's per-fit INFO logging, which would otherwise be repeated for every task
    global _WORKER_ENV_READY
    if _WORKER_ENV_READY:
        return
    worker_compiledir = (
        pathlib.Path(pytensor.config.base_compiledir) / f"worker_{os.getpid()}"
    )
    worker_compiledir.mkdir(parents=True, exist_ok=True)
    pytensor.config.compiledir = worker_compiledir
    # Remove the per-worker compile dir when the worker process exits so they don't accumulate
    # under base_compiledir across runs.
    atexit.register(shutil.rmtree, worker_compiledir, ignore_errors=True)
    logging.getLogger("pymc").setLevel(logging.WARNING)
    _WORKER_ENV_READY = True
