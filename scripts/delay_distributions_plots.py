import argparse

import numpy as np
import pandas as pd

from endoutbreakhet import DelayDistribution
from endoutbreakhet.plotting import plot_delay_distribution, save_figure
from scripts.inputs import get_inputs_delay_distributions_plots


def make_plots():
    inputs = get_inputs_delay_distributions_plots()

    fig_paths = inputs["fig_paths"]
    delay_by_name = {
        delay_name: DelayDistribution.from_df(df=pd.read_csv(path))
        for delay_name, path in inputs["delay_distributions_results_paths"].items()
    }
    for delay_name in ("inc_period", "tost"):
        delay_fig = plot_delay_distribution(
            delay=delay_by_name[delay_name],
            delay_name=delay_name,
            max_lag=inputs["max_plot_lag"],
        )
        save_figure(fig=delay_fig, paths=fig_paths[delay_name])
    # Serial interval used by the naive models, with that implied by the incubation
    # period and TOST used by the SO models
    serial_interval_fig = plot_delay_distribution(
        delay=delay_by_name["serial_interval"],
        delay_name="serial_interval",
        max_lag=inputs["max_plot_lag"],
        label="Naive models",
        overlay=_implied_serial_interval(
            tost=delay_by_name["tost"], inc_period=delay_by_name["inc_period"]
        ),
        overlay_label="SO models (implied)",
    )
    save_figure(fig=serial_interval_fig, paths=fig_paths["serial_interval"])


def _implied_serial_interval(*, tost, inc_period):
    # An infectee's onset follows its infector's by a TOST then an incubation period, so
    # the serial interval's distribution is their convolution
    return DelayDistribution(
        prob=np.convolve(tost.prob, inc_period.prob),
        first_lag=tost.first_lag + inc_period.first_lag,
    )


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=(
            "Plot the discretised delay distributions, with the serial interval implied "
            "by the incubation period and TOST."
        )
    ).parse_args()

    make_plots()
