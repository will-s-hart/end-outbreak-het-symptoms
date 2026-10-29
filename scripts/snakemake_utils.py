import pathlib

import numpy as np
import pandas as pd


def as_param(*, inputs):
    # A script's inputs as builtins, for its rule's params, so the rule reruns when (and only
    # when) a value it is given changes. Snakemake records params by repr and silently skips
    # values that aren't builtins (or pandas objects), and arrays' reprs are abbreviated.
    return _to_builtins(obj=inputs)


def _to_builtins(*, obj):
    if isinstance(obj, dict):
        # Sorted, so that merely reordering a dict doesn't trigger a rerun
        return {key: _to_builtins(obj=obj[key]) for key in sorted(obj)}
    if isinstance(obj, list | tuple):
        return [_to_builtins(obj=item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Before the builtin scalars, since np.float64 subclasses float but has its own repr
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.random.Generator):
        # Its state, so that the seed counts
        return _to_builtins(obj=obj.bit_generator.state)
    if isinstance(obj, pathlib.Path | pd.Timestamp):
        return str(obj)
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    # Rather than pass it through, since Snakemake would silently not track it
    raise TypeError(f"Can't convert {type(obj).__name__} to builtins for a rule param")
