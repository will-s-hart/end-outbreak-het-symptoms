import argparse

import numpy as np
import pandas as pd

from endoutbreakhet import DelayDistribution
from endoutbreakhet.plotting import (
    make_panel_figure,
    plot_delay_distribution,
    plot_incidence,
    save_figure,
    set_plot_style,
)
from scripts.inputs import get_inputs_model_inputs_plots


def make_plots():
    set_plot_style()
    inputs = get_inputs_model_inputs_plots()

    delay_by_name = {
        delay_name: DelayDistribution.from_df(df=pd.read_csv(path))
        for delay_name, path in inputs["delay_distributions_results_paths"].items()
    }
    # The incidence across the top, above the delay distributions
    fig, (incidence_ax, serial_interval_ax, inc_period_ax, tost_ax) = make_panel_figure(
        panel_widths=[None, 1, 1, 1]
    )
    plot_incidence(
        ax=incidence_ax,
        reported_incidence_vec=inputs["reported_incidence_vec"],
        first_case_date=inputs["first_case_date"],
        ert_days=inputs["ert_days"],
    )
    # Serial interval used by the naive models, with that implied by the incubation
    # period and TOST used by the SO models
    plot_delay_distribution(
        ax=serial_interval_ax,
        delay=delay_by_name["serial_interval"],
        delay_name="serial_interval",
        max_lag=inputs["max_plot_lag"],
        label="Naive models",
        overlay=_implied_serial_interval(
            tost=delay_by_name["tost"], inc_period=delay_by_name["inc_period"]
        ),
        overlay_label="SO models (implied)",
    )
    for ax, delay_name in ((inc_period_ax, "inc_period"), (tost_ax, "tost")):
        plot_delay_distribution(
            ax=ax,
            delay=delay_by_name[delay_name],
            delay_name=delay_name,
            max_lag=inputs["max_plot_lag"],
        )
    save_figure(fig=fig, paths=inputs["fig_paths"]["summary"])


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
            "Plot the reported incidence and the discretised delay distributions, with the "
            "serial interval implied by the incubation period and TOST."
        )
    ).parse_args()

    make_plots()
