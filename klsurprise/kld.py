import numpy as np
import jax.numpy as jnp
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .utils import calculate_flat_prior_volume
from .sampling import run_nested_sampling, process_batch
from .io import save_dict_to_hdf5


def KLD_numerical(
    res_p,
    logP,
    res_q,
    logQ,
    domain=None,
    clip_range=[-1e16, 5000],
    clip_values=True,
    progress=True,
    batch_size=1000,
    prior_transform="flat",
):
    """
    Computes the Kullback-Leibler Divergence between two distributions: KLD(p|q).

    Parameters
    ----------
    res_p : dynesty NS results object
        Dynesty nested sampling results for the first distribution.
    logP : function
        Log-probability function for the first distribution.
    res_q : dynesty NS results object
        Dynesty nested sampling results for the second distribution.
    logQ : function
        Log-probability function for the second distribution.
    domain : array-like, optional
        The domain over which the prior is defined. This is required if using
        a flat prior (default is None).
    prior_transform : str, optional
        Type of prior transform applied. Default is 'flat'.
    clip_range : list, optional
        Range for clipping log-probability values to avoid overflow (default is [-1e16, 5000]).
    clip_values : bool, optional
        Whether to clip the log-probability values (default is True).
    progress : bool, optional
        Whether to display a progress bar during processing (default is True).
    batch_size : int, optional
        The size of batches for processing samples (default is 1000).

    Returns
    -------
    kld : float
        The computed Kullback-Leibler Divergence between the two distributions.

    Notes
    -----
    - The function currently assumes a flat prior volume if `prior_transform` is 'flat'.
    - A more efficient way to obtain `samples_p` and corresponding `logP` values exists,
      but is not implemented yet.
    - This function assumes that logP is jax compatible.
    """

    # Compute the prior volume if domain is provided
    if (domain is not None) and (prior_transform == "flat"):
        prior_volume = calculate_flat_prior_volume(domain)
    # else assume it to be one
    else:
        prior_volume = 1

    # Obtain equally weighted samples of distribution p(theta)
    samples_p = res_p.samples_equal()

    # Compute the evidence-normalized log-probability functions for both distributions
    logZp = res_p["logz"][-1] + jnp.log(prior_volume)

    # @jit
    def logP_norm(x):
        return logP(x) - logZp

    logZq = res_q["logz"][-1] + jnp.log(prior_volume)

    # @jit
    def logQ_norm(x):
        return logQ(x) - logZq

    # Process the samples to obtain normalized log-probabilities
    log_prob_p = process_batch(
        logP_norm, samples_p, progress=progress, batch_size=batch_size
    )
    log_prob_q = process_batch(
        logQ_norm, samples_p, progress=progress, batch_size=batch_size
    )

    # Clip values to avoid overflow if specified
    if clip_values:
        log_prob_p = np.clip(log_prob_p, a_min=clip_range[0], a_max=clip_range[1])
        log_prob_q = np.clip(log_prob_q, a_min=clip_range[0], a_max=clip_range[1])

    # Compute the Kullback-Leibler Divergence
    kld = (log_prob_p - log_prob_q).mean()

    return kld


def kld_worker(
    sample,
    logL2,
    logL1,
    ndim,
    domain,
    logL_mock=None,
    mock1_NS_result=None,
    logP_1=None,
    prior_transform="flat",
    n_effective=20000,
    clip_range=[-1e16, 50000],
):
    """
    Worker function for parallelizing the evaluation of the Kullback-Leibler Divergence (KLD) distribution.

    Parameters:
    -----------
    sample : array-like
        A sample from the Posterior Predictive Distribution (PPD) for which KLD distribution is to be evaluated.

    logL2 : callable
        Function to compute the log-likelihood for data 2. It should accept two inputs: (parameters, data).

    logL1 : callable
        Log-likelihood function for the first dataset.

    ndim : int
        The number of dimensions in the parameter space.

    domain : array-like
        Parameter space domain.

    logL_mock : callable, optional
        Function to compute the log-likelihood for mock data. If None, logL2 is used.

    mock1_NS_result : object, optional
        Result object from the first Nested Sampling run (output from a Dynesty nested sampling).

    logP_1 : callable, optional
        Log-posterior function for the data from the first dataset.

    prior_transform : callable or str, optional
        Transformation function for the prior distribution, which maps a uniform distribution to the
        parameter space. Can be a custom function or the string 'flat' for flat priors. Defaults to 'flat'.

    n_effective : int, optional
        The number of effective samples to target for the nested sampling run. Defaults to 20,000.

    clip_range : list, optional
        Range for clipping log-likelihood values to avoid numerical overflow or underflow. Defaults to [-1e16, 50000].

    Returns:
    --------
    tuple
        kld, sample:
        - kld: The value KLD(p_mock, p1) where p_mock was created using sample as data vector for likelihood 2.
        - sample: The input PPD sample used to generate the value of kld returned.

    Notes:
    ------
    - If `domain` is provided, it will override the domain information in `prior_transform`.
    - The domain information is only required when using a flat prior.
    - Currently only works for flat priors.
    """

    # create mock data and run nested sampling
    # @jit
    def logpMock_2(theta):
        return jnp.nan_to_num(
            logL2(theta, sample), nan=1e-32
        )  # create full posterior distribution 2

    results2 = run_nested_sampling(
        logpMock_2,
        ndim=ndim,
        prior_transform=prior_transform,
        domain=domain,
        n_effective=n_effective,
    )  # functions arguments are the best for SNIa chain.

    kld_return = KLD_numerical(
        results2,
        logpMock_2,
        mock1_NS_result,
        logL1,
        domain=domain,
        clip_range=clip_range,
        clip_values=True,
        progress=False,
        batch_size=1000,
        prior_transform="flat",
    )

    return kld_return, sample


def compute_kld_distribution(
    PPDsamples,
    logL2,
    logL1,
    ndim,
    domain,
    mock1_NS_result,
    logP_1=None,
    n_jobs=4,
    result_path=None,
    prior_transform="flat",
    n_effective=20000,
    clip_range=[-1e16, 50000],
):
    """
    Parallel computation of KLD for PPD samples and saving results to HDF5. Will compute distribution Dkl(p2i, p1)

    Parameters:
    - PPDsamples: Collection of PPD samples to process.
    - logL2: Function for log-likelihood computation for mock data. A function that contains two inputs> (param, data)
    - logL1: Log-likelihood function for the first dataset.
    - ndim: Number of dimensions in the parameter space.
    - domain: Parameter space domain.
    - mock1_NS_result: result object from Dynesty Nested Sampling for mock 1.
    - logP_1 (optional): Log-posterior function for dataset 1. If None, logL1 is used.
    - n_jobs (optional): Number of parallel jobs. Defaults to 2.
    - result_path (optional): Path to HDF5 file for results. Defaults to 'results.hdf5'.
    - prior_transform: currently a flat prior transform defined by domain.
    """
    kld_results, ppdsample_results = [], []

    if logP_1 is None:
        logP_1 = logL1

    results = Parallel(n_jobs=n_jobs)(
        delayed(kld_worker)(
            sample,
            logL2,
            logL1,
            ndim,
            domain,
            None,
            mock1_NS_result,
            logP_1,
            prior_transform,
            n_effective,
            clip_range,
        )
        for i, sample in enumerate(tqdm(PPDsamples, desc="Iterating over the PPD"))
    )

    for kld, ppdsample in results:
        kld_results.append(kld)
        ppdsample_results.append(ppdsample)

    results_dict = {
        "kld_dist": np.array(kld_results),
        "ppd_sample": np.array(ppdsample_results),
    }

    if result_path is not None:
        save_dict_to_hdf5(result_path, results_dict)

    return kld_results
