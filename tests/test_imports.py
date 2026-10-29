# Note that AI tools were used to generate tests
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[1]
ANALYSIS_SCRIPT_NAMES = sorted(
    path.stem
    for path in (REPO_ROOT / "scripts").glob("*.py")
    if not path.stem.endswith("_plots")
    and path.stem not in ("inputs", "snakemake_utils", "__init__")
)


@pytest.mark.parametrize("script_name", ANALYSIS_SCRIPT_NAMES)
def test_analysis_scripts_dont_import_plotting(*, script_name):
    # The Snakefile leaves the plotting module out of analysis rules' inputs, which is only
    # right if they never import it. Checked in a subprocess, since the plotting tests have
    # already imported it into this one.
    code = (
        f"import sys, scripts.{script_name}; "
        "assert 'endoutbreakhet.plotting' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
