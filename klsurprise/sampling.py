import numpy as np
import dynesty
from dynesty import utils as dyfunc

import jax.numpy as jnp
from jax import vmap

from tqdm.auto import tqdm
from joblib import load, dump


def run_nested_sampling(
    loglikelihood,
    ndim,
    domain,
    prior_transform="flat",
    static_NS=False,
    dynamic_NS=True,
    dlogz=0.5,
    print_progress=False,
    nlive=1000,
    nlive_batch=200,
    maxiter=600000,
    maxbatch=50,
    use_stop=True,
    bootstrap=0,
    n_effective=20000,
    **kwargs,
):
    """
    Performs a Nested Sampling run to estimate the Bayesian evidence and obtain posterior samples for a given loglikelihood function and parameter space.
    This function is designed to be flexible, allowing for customization of the domain, prior transformation, and other parameters.

    Parameters:
        loglikelihood (callable): The log-likelihood function that accepts an array of parameter values and returns the log-likelihood.
                                This function should be defined by the user and is a critical component of the Nested Sampling algorithm.
        ndim (int): The number of dimensions of the parameter space.
        domain (np.array, optional): A 2D numpy array specifying the domain (parameter ranges) for each dimension of the parameter space.
                                    The shape of this array should be (ndim, 2), where each row corresponds to the minimum and maximum
                                    values for a parameter. If `None`, it is assumed that domain information is provided by the `prior_transform` argument.
                                    Defaults to `None`.
        prior_transform (str or callable, optional): A function that transforms samples from the unit cube to the parameter space defined by the domain.
                                                    If set to "flat", a uniform prior over the specified domain is used. Custom transformations can be provided
                                                    by passing a callable that implements the desired transformation. Defaults to "flat".
        static_NS (bool, optional): If `True`, performs a static Nested Sampling run, which accurately estimates the evidence but may not sample the posterior as effectively. Defaults to `False`.
        dynamic_NS (bool, optional): If `True`, performs a dynamic Nested Sampling run, which is better at sampling the posterior. Defaults to `True`.
        dlogz (float, optional): The stopping criterion for the Nested Sampling run, based on the estimated remaining evidence. Defaults to `0.5`.
        print_progress (bool, optional): If `True`, progress information is printed during the Nested Sampling run. Defaults to `False`.
        nlive (int, optional): The number of live points used in the Nested Sampling run. Higher values can improve accuracy but increase computational cost. Defaults to `1000`.
        nlive_batch (int, optional): The number of live points to add in each batch during dynamic Nested Sampling. Defaults to `200`.
        maxiter (int, optional): The maximum number of iterations allowed in the Nested Sampling run. Defaults to `600000`.
        maxbatch (int, optional): The maximum number of batches allowed during dynamic Nested Sampling. Defaults to `50`.
        use_stop (bool, optional): If `True`, the dynamic Nested Sampling will attempt to stop when it detects convergence. Defaults to `True`.
        bootstrap (int, optional): Specifies the number of bootstrapping iterations for error estimation of the evidence. Defaults to `0`, indicating no bootstrapping.
        n_effective (int, optional): The target number of effective posterior samples for dynamic Nested Sampling. Defaults to `20000`.
        **kwargs: Additional keyword arguments to be passed directly to the `dynesty.NestedSampler` or `dynesty.DynamicNestedSampler` constructors. This allows for further customization of the sampler.

    Returns:
        results (object): An object containing the results of the Nested Sampling run. This object includes attributes such as posterior samples,
                        log evidence, and information on the sampling efficiency. The exact contents can vary depending on the version and configuration of `dynesty`.

    Note:
        This function requires the `dynesty` package for Nested Sampling. Ensure that `dynesty` is installed and properly configured in your environment.

    WARNING:
        If a "prior_transform" is provided, the "domain" should generally be set to `None` to avoid conflicts.
    """
    # Ensure the domain is a numpy array with the right shape
    domain = np.asarray(domain)
    assert domain.shape == (ndim, 2), "Domain must be an array with shape (ndim, 2)."

    if prior_transform == "flat":
        # Define the prior transform function
        def prior_transform_fun(utheta):
            """Transforms samples `utheta` drawn from the unit cube to samples from the domain."""
            return domain[:, 0] + utheta * (domain[:, 1] - domain[:, 0])
    else:
        prior_transform_fun = prior_transform
        raise NotImplementedError("Only flat prior is currently implemented")

    if static_NS:
        # "Static" nested sampling.
        # Accurately measures evidence but it's not as effective in sampling the posterior.
        sampler = dynesty.NestedSampler(
            loglikelihood,
            prior_transform_fun,
            ndim,
            nlive=nlive,
            bootstrap=bootstrap,
            **kwargs,
        )
        sampler.run_nested(dlogz=dlogz, print_progress=print_progress)
        sresults = sampler.results
        results = sresults

    if dynamic_NS:
        # "Dynamic" nested sampling.
        # Effectively samples the posterior.
        dsampler = dynesty.DynamicNestedSampler(
            loglikelihood, prior_transform_fun, ndim, bootstrap=bootstrap
        )
        dsampler.run_nested(
            dlogz_init=dlogz,
            print_progress=print_progress,
            nlive_init=nlive,
            nlive_batch=nlive_batch,
            maxbatch=maxbatch,
            maxiter=maxiter,
            use_stop=use_stop,
            n_effective=n_effective,
        )
        dresults = dsampler.results
        results = dresults

    if dynamic_NS and static_NS:
        # Combine results from "Static" and "Dynamic" runs.
        results = dyfunc.merge_runs([sresults, dresults])

    return results


def process_batch(logpost, parameter_array, batch_size=1000, progress=False):
    """
    Apply a function logpost over parameters in batches.
    This makes distributes the computation of logP and makes the evaluation of KLD faster, but limits the code usage
    to jax compatible logposteriors.

    Parameters:
        logpost (callable): The log-posterior function to apply to the coordinates. This function should be jax compatible.
        parameter_array (array-like): A 2D array where each row represents a parameter vector.
                                The function will be applied to each row.
        batch_size (int, optional): The number of coordinates to process in each batch. Defaults to 1000.
        progress (bool, optional): If `True`, displays a progress bar during processing. Defaults to `False`.

    Returns:
        logpost_matrix (array-like): A 1D array containing the log-posterior values for each coordinate,
                                    concatenated from the results of processing each batch.

    Note:
        This function uses `vmap` from JAX to vectorize the `logpost` function, allowing for efficient batch processing.
    """
    # Vectorize the logpost function to apply it over batches of coordinates
    vmap_logpost = vmap(logpost)

    # Prepare to process the coordinates in batches
    flat_coordinates = parameter_array
    logpost_matrix = []

    # Iterate over the coordinates in batches
    for i in (
        tqdm(range(0, flat_coordinates.shape[0], batch_size), desc="Processing batches")
        if progress
        else range(0, flat_coordinates.shape[0], batch_size)
    ):
        # Select the current batch of coordinates
        batch_coords = flat_coordinates[i : i + batch_size]
        # Apply the vectorized logpost function to the batch and store the results
        logpost_matrix.append(vmap_logpost(batch_coords))

    # Concatenate the results from all batches into a single array
    logpost_matrix = jnp.concatenate(logpost_matrix)

    return logpost_matrix


def load_create_NS_file(data_1_name, logL1, ndim, domain, n_effective=15000, dlogz=0.5):
    """
    This function checks for existing NS results and if they don't exist, it creates one using logL1 and domain.
    """
    print(70 * "-")
    print("Loading posterior data")
    print(70 * "-")
    try:
        ## loading pre-made Nested Sampling run
        res_1 = load(data_1_name)
        print("Data loaded sucessfully!")
    except (FileNotFoundError, OSError, TypeError):
        print(70 * "-")
        print("Loading failed!")
        print(70 * "-")

        print("Running Nested Sampling...")
        print(70 * "-")
        # any nested sampling routine will be fine, as long as it has the method .samples_equal() or equivalent, to obtain equally weighted samples.
        res_1 = run_nested_sampling(
            logL1,
            ndim,
            domain=domain,
            print_progress=True,
            n_effective=n_effective,
            dlogz=dlogz,
        )
        print("Run completed sucessfully.")

        # if data_1_name is not None:
        try:
            print("Saving ", data_1_name)
            dump(res_1, data_1_name)
        except OSError:
            pass
    return res_1
