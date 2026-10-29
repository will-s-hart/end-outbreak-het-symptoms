import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from endoutbreakhet.utils import parse_lognormal_prior

_PARAM_LABELS = {
    "rep_no_before": r"$R_\mathrm{before}$",
    "rep_no_after": r"$R_\mathrm{after}$",
    "dispersion": r"$k$",
}

_MODEL_STYLES = {
    "poisson": {"label": "Poisson", "color": "black"},
    "sse": {"label": "SSE", "color": "tab:blue"},
    "ssi": {"label": "SSI", "color": "tab:green"},
    "dlo": {"label": "DLO", "color": "tab:red"},
    "sse_so": {"label": "SSE-SO", "color": "tab:pink"},
    "ssi_so": {"label": "SSI-SO", "color": "tab:orange"},
    "poisson_so": {"label": "Poisson-SO", "color": "tab:gray"},
}

_RISK_TYPE_LINESTYLES = {
    "cases": "-",
    "transmission": "--",
    "sustained": ":",
    "reports": "-.",
}

_RISK_TYPE_YLABELS = {
    "cases": "Risk of additional cases",
    "transmission": "Risk of additional transmission",
    "sustained": "Risk of sustained transmission",
    "reports": "Risk of additional reported cases",
}


def plot_densities(
    *, posterior_df_by_model, priors, param_name, log_x=False, legend=True
):
    fig, ax = plt.subplots()

    lognormal_params = parse_lognormal_prior(name=param_name, priors=priors)
    prior_dist = scipy.stats.lognorm(
        s=lognormal_params["sigma"], scale=np.exp(lognormal_params["mu"])
    )
    # Models without the parameter (e.g. Poisson models for the dispersion) are left out
    samples_by_model = {
        model_name: posterior_df[param_name].to_numpy()
        for model_name, posterior_df in posterior_df_by_model.items()
        if param_name in posterior_df.columns
    }

    # Limits from central prior/posterior mass, so long tails don't squash the plot
    upper = max(
        [
            prior_dist.ppf(0.975),
            *(np.quantile(samples, 0.99) for samples in samples_by_model.values()),
        ]
    )
    if log_x:
        lower = min(
            [
                prior_dist.ppf(0.025),
                *(np.quantile(samples, 0.01) for samples in samples_by_model.values()),
            ]
        )
        x_vec = np.geomspace(lower, upper, 500)
        # Densities per unit log10(x), so areas on the log axis are probabilities
        kde_eval_vec = np.log10(x_vec)
        prior_density_vec = prior_dist.pdf(x_vec) * x_vec * np.log(10)
        ax.set_xscale("log")
    else:
        lower = 0
        x_vec = np.linspace(lower, upper, 500)
        kde_eval_vec = x_vec
        prior_density_vec = prior_dist.pdf(x_vec)

    for model_name, samples in samples_by_model.items():
        kde = scipy.stats.gaussian_kde(np.log10(samples) if log_x else samples)
        model_style = _MODEL_STYLES[model_name]
        ax.plot(
            x_vec,
            kde(kde_eval_vec),
            color=model_style["color"],
            label=model_style["label"],
        )
    ax.plot(x_vec, prior_density_vec, "k--", label="prior")
    ax.set_xlim(lower, upper)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(_PARAM_LABELS[param_name])
    ax.set_ylabel("Density")

    if legend:
        ax.legend()
    return fig


def plot_model_probabilities(*, model_probability_by_model):
    fig, ax = plt.subplots()
    pie = ax.pie(
        list(model_probability_by_model.values()),
        colors=[
            _MODEL_STYLES[model_name]["color"]
            for model_name in model_probability_by_model
        ],
    )
    ax.legend(
        pie.wedges,
        [
            f"{_MODEL_STYLES[model_name]['label']} ({prob:.2f})"
            for model_name, prob in model_probability_by_model.items()
        ],
    )
    ax.set_title("Model probabilities")
    return fig


def plot_risk_incidence_overlay(
    *,
    risk_df_by_model,
    reported_incidence_vec,
    first_case_date,
    risk_types=("cases",),
    vline_days=(),
):
    # x is in days since the first case
    fig, ax = plt.subplots()
    for model_name, risk_df in risk_df_by_model.items():
        model_style = _MODEL_STYLES[model_name]
        # Risks of cases and transmission coincide in the naive models, so they get one curve
        model_risk_types = (
            tuple(risk_type for risk_type in risk_types if risk_type != "transmission")
            if not model_name.endswith("_so") and "cases" in risk_types
            else risk_types
        )
        for risk_type in model_risk_types:
            label = model_style["label"]
            if len(model_risk_types) > 1:
                label += f" ({risk_type})"
            ax.plot(
                risk_df["day"],
                risk_df[f"risk_{risk_type}"],
                color=model_style["color"],
                linestyle=_RISK_TYPE_LINESTYLES[risk_type],
                label=label,
            )
    for vline_day in vline_days:
        ax.axvline(vline_day, color="black", linestyle="--")
    ax.legend()
    _month_start_x_axis(
        ax=ax,
        day_0_date=first_case_date,
        # Every model has the same risk days
        plot_days=next(iter(risk_df_by_model.values()))["day"].to_numpy(),
        interval_months=1,
    )
    ax.set_ylabel(_RISK_TYPE_YLABELS[risk_types[0]] if len(risk_types) == 1 else "Risk")
    twin_ax = ax.twinx()
    twin_ax.bar(
        x=np.arange(len(reported_incidence_vec)),
        height=reported_incidence_vec,
        color="black",
        alpha=0.5,
    )
    twin_ax.set_ylim(0, np.max(reported_incidence_vec))
    twin_ax.set_ylabel("Number of cases")
    twin_ax.yaxis.label.set_color("tab:gray")
    twin_ax.tick_params(axis="y", colors="tab:gray")
    twin_ax.spines["right"].set_visible(True)
    twin_ax.spines["right"].set_color("tab:gray")
    twin_ax.spines["left"].set_visible(False)
    twin_ax.spines["bottom"].set_visible(False)
    # Draw the primary axis (curves, legend) above the incidence bars.
    ax.set_zorder(twin_ax.get_zorder() + 1)
    ax.patch.set_visible(False)
    return fig


def save_figure(*, fig, paths):
    # Saves fig to each path (one per file format), then closes it
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    plt.close(fig)


def _month_start_x_axis(*, ax, day_0_date, plot_days, interval_months=1):
    # x axis in days since day_0_date, spanning the whole months that contain plot_days, with
    # ticks at the month starts and every interval_months-th one labelled
    plot_dates = day_0_date + pd.to_timedelta(plot_days, unit="D")
    month_start_dates = pd.date_range(
        start=plot_dates.min().replace(day=1),
        end=plot_dates.max().replace(day=1) + pd.offsets.MonthBegin(1),
        freq="MS",
    )
    month_start_days = (month_start_dates - day_0_date).days
    multi_year = month_start_dates.year.nunique() > 1
    labels = [
        f"{d.day} {d:%b}" + (f" {d.year}" if multi_year else "")
        if (i % interval_months == 0)
        else ""
        for i, d in enumerate(month_start_dates)
    ]
    ax.set_xticks(month_start_days, labels=labels)
    ax.set_xlim(month_start_days[0], month_start_days[-1])
    ax.set_xlabel("Date" if multi_year else f"Date ({month_start_dates[0].year})")
