import numpy as np
import scipy.stats


def lognormal_params_from_median_percentile_2_5(median, percentile_2_5):
    mu = np.log(median)
    sigma = (mu - np.log(percentile_2_5)) / scipy.stats.norm.ppf(0.975)
    return {"mu": mu, "sigma": sigma}
