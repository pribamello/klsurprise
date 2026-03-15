from getdist import plots, MCSamples
import matplotlib.pyplot as plt
import numpy as np


##############################################################################################
### Plotting utilities
##############################################################################################


def freedman_diaconis_bin_width(data):
    """Calculate the optimal bin width using the Freedman-Diaconis rule."""
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr * len(data) ** (-1 / 3)
    return bin_width


def plot_histograms(
    data_arrays,
    data_labels,
    cumulative=False,
    marginalize=True,
    num_bins=None,
    title=None,
    alphas=None,
    xlabel="S nats",
    ylabel="PDF",
    xrange=None,
    stat="density",
    legend_fontsize=16,
    label_fontsize=20,
    tick_fontsize=14,
    figsize=(8, 4),
):
    """
    Plot histograms for multiple data arrays with options for cumulative distribution, marginalization,
    varying number of bins, and alpha values.

    Parameters:
    - data_arrays (list of ndarray): A list of data arrays to plot.
    - data_labels (list of str): A list of labels corresponding to each data array.
    - cumulative (bool, optional): If True, plots the cumulative histogram. Defaults to False.
    - marginalize (bool, optional): If True, normalizes the histogram to form a probability density. Defaults to True.
    - num_bins (list of int, optional): Number of bins for each histogram. Defaults to None.
    - alphas (list of float, optional): Alpha values for each histogram. Defaults to None.
    - xlabel (str, optional): Label for the x-axis.
    - ylabel (str, optional): Label for the y-axis.
    - xrange (tuple, optional): Range for the x-axis.
    - stat (str, optional): If 'density', the histogram is normalized to form a probability density function.
    - legend_fontsize (int, optional): Font size for the legend. Defaults to 16.
    - label_fontsize (int, optional): Font size for the x and y labels. Defaults to 20.
    - tick_fontsize (int, optional): Font size for the ticks on both x and y axes. Defaults to 14.
    - figsize (tuple, optional): Figure size. Defaults to (8, 4).

    Returns:
    - ax (matplotlib.axes._subplots.AxesSubplot): The plot object for further modification.
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if num_bins is None:
        num_bins = []
        for data in data_arrays:
            bin_width = freedman_diaconis_bin_width(data)
            bins = int((np.max(data) - np.min(data)) / bin_width)
            num_bins.append(bins)
            print("nbins = ", bins)

    if alphas is None:
        alphas = np.linspace(1, 0.4, len(data_arrays))  # Decreasing alpha values

    for data, label, bins, alpha in zip(data_arrays, data_labels, num_bins, alphas):
        weights = np.ones_like(data) / (len(data) if marginalize else 1)
        ax.hist(
            data,
            bins=bins,
            cumulative=cumulative,
            density=stat,
            weights=weights,
            label=label,
            alpha=alpha,
            histtype="stepfilled",
        )

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if xrange is not None:
        ax.set_xlim(xrange)
    ax.legend(fontsize=legend_fontsize)

    # Set tick parameters for both axes
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    return ax


def create_triangle_plot(
    chain_1,
    chain_2=None,
    D1="Chain 1",
    D2="Chain 2",
    domain=None,
    names=None,
    labels=None,
    lgl1=None,
    lgl2=None,
    smooth=(0.3, 0.3),
    save_plot=False,
    burnin=0.0,
    MLE1=None,
    MLE2=None,
    width_inch=8,
    subplot_size=1,
    label_size=20,
    tick_label_size=16,
    pad_inches=0.2,
    plot_colors=None,
):
    """
    Generate a triangle plot for one or two sets of Monte Carlo samples with the given parameters and optional likelihoods.

    Parameters:
    - chain_1, chain_2 (ndarray): The MCMC samples for datasets.
    - domain (ndarray): [min, max] ranges for each parameter.
    - lgl1, lgl2 (ndarray, optional): Log-likelihoods for the chains.
    - D1, D2 (str, optional): Identifiers for the datasets.
    - smooth (tuple, optional): Smoothing scales (smooth1D, smooth2D).
    - save_plot (bool, optional): If True, saves the plot to a file.
    - burnin (float, optional): Fraction of samples to discard as burn-in.
    - names, labels (list, optional): Parameter names and labels.
    - MLE1, MLE2 (array, optional): Maximum Likelihood Estimators for the datasets.
    - plot_colors (list, optional): List containing colors for 'chain_1', 'chain_2', 'MLE1', and 'MLE2'.
      Must have 4 elements: [chain1_color, chain2_color, MLE1_color, MLE2_color].
      If None, defaults to ['C0', 'C1', 'C2', 'C3'].

    Returns:
    - Plot object (getdist subplot plotter).
    """
    # Infer names and labels if not provided
    num_params = chain_1.shape[1]
    if names is None:
        names = [f"x_{i + 1}" for i in range(num_params)]
    if labels is None:
        labels = names
    if domain is None:
        # infer domain from data
        delta_domain = 1 * np.sqrt(chain_1.var(axis=0))
        domain = np.array(
            [
                chain_1.min(axis=0) - 4 * delta_domain,
                chain_1.max(axis=0) + 4 * delta_domain,
            ]
        ).T

    # Create ranges dictionary from domain array
    ranges = {name: domain[i] for i, name in enumerate(names)}

    # Create MCSamples instance for chain_1
    samples_args_1 = {
        "samples": chain_1,
        "ranges": ranges,
        "ignore_rows": burnin,
        "names": names,
        "labels": labels,
        "label": D1,
    }
    if lgl1 is not None:
        samples_args_1["loglikes"] = lgl1
    samples_1 = MCSamples(**samples_args_1)

    # Unpack smoothness settings
    smooth1D, smooth2D = smooth
    settings = {
        "contours": [0.68, 0.95, 0.99],
        "smooth_scale_1D": smooth1D,
        "smooth_scale_2D": smooth2D,
    }

    # Update plot settings for chain_1
    samples_1.updateSettings(settings)

    # Prepare the list of samples for plotting
    samples_list = [samples_1]
    if chain_2 is not None:
        samples_args_2 = {
            "samples": chain_2,
            "ranges": ranges,
            "ignore_rows": burnin,
            "names": names,
            "labels": labels,
            "label": D2,
        }
        if lgl2 is not None:
            samples_args_2["loglikes"] = lgl2
        samples_2 = MCSamples(**samples_args_2)

        # Update plot settings for chain_2
        samples_2.updateSettings(settings)

        # Add samples_2 to the list for plotting
        samples_list.append(samples_2)

    # Set plot colors — fixed bug: default list had only 2 elements but indices 2-3 were accessed
    if plot_colors is None:
        plot_colors = ["C0", "C1", "C2", "C3"]

    MLE1_color = plot_colors[2]
    MLE2_color = plot_colors[3]
    contour_colors = plot_colors[:2]

    # Generate the triangle plot
    mp = plots.get_subplot_plotter(width_inch=width_inch, subplot_size=subplot_size)
    mp.alpha_filled_add = 0.5
    mp.settings.figure_legend_frame = False
    mp.settings.legend_fontsize = 20
    mp.triangle_plot(
        samples_list,
        filled=True,
        normalized=True,
        legend_labels=[D1, D2],
        legend_loc="upper right",
        contour_colors=contour_colors,
    )

    ################### add MLE point in plot ###################
    # Plot MLE points if provided
    if MLE1 is not None:
        for i in range(len(names)):
            ax = mp.subplots[i, i]  # Access the diagonal subplot for each parameter
            if ax is not None:
                ax.axvline(MLE1[i], color=MLE1_color, ls="--")
            for j in range(i):
                ax2 = mp.subplots[i, j]
                if ax2 is not None:
                    ax2.scatter(MLE1[j], MLE1[i], color=MLE1_color, marker="x")

    if MLE2 is not None:
        for i in range(len(names)):
            ax = mp.subplots[i, i]
            if ax is not None:
                ax.axvline(MLE2[i], color=MLE2_color, ls="--")
            for j in range(i):
                ax2 = mp.subplots[i, j]
                if ax2 is not None:
                    ax2.scatter(MLE2[j], MLE2[i], color=MLE2_color, marker="x")
    #############################################################

    # Adjust font sizes individually
    for ax in mp.subplots.flatten():
        if ax:
            ax.tick_params(axis="both", which="major", labelsize=tick_label_size)
            ax.xaxis.label.set_size(label_size)
            ax.yaxis.label.set_size(label_size)

    # Determine the file name based on whether chain_2 is provided and save if requested
    fname = f"TrianglePlot_{D1}" + (f"-{D2}" if chain_2 is not None else "") + ".pdf"
    if save_plot:
        plt.savefig(fname, bbox_inches="tight", pad_inches=pad_inches)
    return mp
