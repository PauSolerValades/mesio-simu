"""
This module contains all functions
which generate Matplotlib figures
given a sample
"""

from typing import Optional, Tuple, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_histogram_weibull(
    sample: np.ndarray, params: Optional[Dict] = None, save_plot: bool = True
):
    fig, ax = plt.subplots()
    sns.histplot(x=sample, kde=False, stat="density", color="white", ax=ax)
    sns.kdeplot(x=sample, color="black", ax=ax)
    ax.set_xlabel("Failure time")
    fig.suptitle("Weibull Distribution sample", fontsize=14)
    if params is not None:
        ax.set_title(f"(α = {params['alpha']}, β = {params['beta']})")
    if save_plot:
        plt.savefig("foo.png", dpi=500)
    else:
        return ax


def plot_grid_weibull(sample_grid: Dict[Tuple, np.ndarray]):
    n_scenarios = len(list(sample_grid.keys()))
    n_rows = int(n_scenarios / 3)
    fig, ax = plt.subplots(nrows=n_rows, ncols=3)
    for i, (params, sample) in enumerate(sample_grid):
        ax[i] = plot_histogram_weibull(sample, 
        params={"alpha": params[0], "beta": params[1]}, save_plot=False)
    #TODO