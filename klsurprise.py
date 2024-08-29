import numpy as np
import dynesty
from dynesty import utils as dyfunc

import jax.numpy as jnp
from jax import vmap
from jax import jit

from tqdm.auto import tqdm

class surprise_statistics:
    """
    Computes the surprise statistics between two datasets or models using Nested Sampling.

    This function calculates the surprise (S) distribution and, optionally, the S value between two datasets or models
    by performing Nested Sampling. It allows for comparing the likelihoods and data models within a specified parameter
    domain using a top-hat prior.

    Computes the Surprise statistics S(p(x|D2)||p(x|D1))

    Args:
        logL1 (callable): 
            A function that computes the log-likelihood of data_1 given parameters (theta).
            Signature: logL1(theta) -> float
        data_2_model_fun (callable): 
            A function that maps parameters (theta) to the data space (D) for data_2.
            Signature: data_2_model_fun(theta) -> array_like
        covariance_matrix_2 (array_like): 
            The covariance matrix of data_2 in the data space. The likelihood is assumed to be Gaussian.
        domain (array_like): 
            The parameter space boundaries defining the top-hat prior for Nested Sampling.
            Should be a Nx2 array, e.g. np.array([[0.6,0.9],[0.2, 1.5]])
        data_2 (array_like, optional): 
            The data vector for data_2. If provided, the function will perform Nested Sampling on likelihood 2 
            using a top-hat prior defined by domain. Necessary to compute S.If None, only the S distribution 
            will be provided. Default is None.
        data_1_name (str, optional): 
            The name identifier for data_1. If provided, the Nested Sampling results for data_1 will be saved
            using this name. If None, results will not be saved. Default is None.
        data_2_name (str, optional): 
            The name identifier for data_2. If provided, the Nested Sampling results for data_2 will be saved
            using this name. If None, results will not be saved. Default is None.

    Notes:
        - The code assumes Gaussian likelihoods.
        - The code assumes top-hat priors defined by domain.
        - For now, it's necessary that the likelihoods provided are jax compatible. So the code can't work with some  
          external likelihoods like Planck.  
    """


    def __init__(self, logL1, data_2_model_fun, covariance_matrix_2, domain, data_2=None, data_1_name=None, data_2_name=None):
            """
            Initializes the SurpriseStatistics class with the provided parameters.
            """
            self.logL1 = logL1
            self.data_2_model_fun = data_2_model_fun
            self.covariance_matrix_2 = covariance_matrix_2
            self.domain = domain
            self.data_2 = data_2
            self.data_1_name = data_1_name
            self.data_2_name = data_2_name
            self.ndim = domain.shape[0]

    def calculate_flat_prior_volume(self, domain=None):
        """
        Calculate the volume of a flat (uniform) prior distribution over a specified multidimensional domain.

        Parameters:
        domain (numpy array): A 2D array where each row corresponds to a different dimension of the parameter space.
                            The first column contains the lower bounds, and the second column contains the upper bounds for each dimension.

        Returns:
        float: The calculated volume of the domain, which is the product of the lengths of the intervals for each dimension.
        """
        if domain is None:
            domain = self.domain
        lengths = np.diff(domain, axis=1).T
        volume = np.prod(lengths)
        return volume


    def run_nested_sampling(self, loglikelihood, ndim=None, domain=None, prior_transform="flat", static_NS=False, dynamic_NS=True, dlogz=0.5, 
                            print_progress=False, nlive=1000, nlive_batch=200, maxiter=600000, maxbatch=50, use_stop=True,
                            bootstrap=0, n_effective=20000, **kwargs):
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
        if ndim is None:
            ndim = self.ndim
        if domain is None:
            domain = self.domain
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
            raise("Only flat prior is currently implemented")

        if static_NS:
            # "Static" nested sampling.
            # Accurately measures evidence but it's not as effective in sampling the posterior.
            sampler = dynesty.NestedSampler(loglikelihood, prior_transform_fun, ndim, nlive=nlive, bootstrap=bootstrap, **kwargs)
            sampler.run_nested(dlogz=dlogz, print_progress=print_progress)
            sresults = sampler.results
            results = sresults
        
        if dynamic_NS:
            # "Dynamic" nested sampling.
            # Effectively samples the posterior.
            dsampler = dynesty.DynamicNestedSampler(loglikelihood, prior_transform_fun, ndim, bootstrap=bootstrap)
            dsampler.run_nested(dlogz_init=dlogz, print_progress=print_progress, nlive_init=nlive, 
                                nlive_batch=nlive_batch, maxbatch=maxbatch, maxiter=maxiter, use_stop=use_stop, n_effective=n_effective)
            dresults = dsampler.results
            results = dresults
        
        if dynamic_NS and static_NS:
            # Combine results from "Static" and "Dynamic" runs.
            results = dyfunc.merge_runs([sresults, dresults])
            
        return results

    def process_batch(self, logpost, parameter_array, batch_size=1000, progress=False):
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
        for i in tqdm(range(0, flat_coordinates.shape[0], batch_size), desc="Processing batches") if progress else range(0, flat_coordinates.shape[0], batch_size):
            # Select the current batch of coordinates
            batch_coords = flat_coordinates[i:i + batch_size]
            # Apply the vectorized logpost function to the batch and store the results
            logpost_matrix.append(vmap_logpost(batch_coords))
        
        # Concatenate the results from all batches into a single array
        logpost_matrix = jnp.concatenate(logpost_matrix)
        
        return logpost_matrix


    def KLD_numerical(self, res_p, logP, res_q, logQ, domain=None, 
                        clip_range=[-1e16, 5000], clip_values=True, progress=True, 
                        batch_size=1000):
        """
        Computes the Kullback-Leibler Divergence between two distributions: KLD(p|q).

        Parameters:
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
        
        Returns:
        -------
        kld : float
            The computed Kullback-Leibler Divergence between the two distributions.

        Notes:
        -----
        - The function currently assumes a flat prior volume if `prior_transform` is 'flat'.
        - A more efficient way to obtain `samples_p` and corresponding `logP` values exists, 
        but is not implemented yet.
        - This function assumes that logP is jax compatible.
        """

        # Compute the prior volume if domain is provided
        prior_transform='flat'
        if (domain is not None) and (prior_transform == 'flat'):
            prior_volume = self.calculate_flat_prior_volume(domain)
        # else assume it to be one
        else:
            prior_volume = 1

        # Obtain equally weighted samples of distribution p(theta)
        samples_p = res_p.samples_equal()
        
        # Compute the evidence-normalized log-probability functions for both distributions
        logZp = res_p['logz'][-1] + np.log(prior_volume)
        
        @jit
        def logP_norm(x):
            return logP(x) - logZp    

        logZq = res_q['logz'][-1] + np.log(prior_volume)
        
        @jit
        def logQ_norm(x):
            return logQ(x) - logZq 
            
        # Process the samples to obtain normalized log-probabilities
        log_prob_p = self.process_batch(logP_norm, samples_p, progress=progress, batch_size=batch_size)
        log_prob_q = self.process_batch(logQ_norm, samples_p, progress=progress, batch_size=batch_size)

        # Clip values to avoid overflow if specified
        if clip_values:
            log_prob_p = np.clip(log_prob_p, a_min=clip_range[0], a_max=clip_range[1])
            log_prob_q = np.clip(log_prob_q, a_min=clip_range[0], a_max=clip_range[1])

        # Compute the Kullback-Leibler Divergence
        kld = (log_prob_p - log_prob_q).mean()

        return kld


    def kld_worker(self):
        # this function is used so the code can compute the Surprise statistics in parallel usnig joblib
        pass

    def write_results_to_hdf5(self):
        # this function is currently beeing used to save the results.
        # but it's badly writen and I already have a new version.
        # I need to integrate it here.
        pass

    def save_dict_to_hdf5(self):
        # this is the better implemented function.
        pass

    def compute_kld_distribution(self):
        # old main_parallel
        # this function computes the KLD distribution
        # this is the most costly function, which is used to compute the KLD distribution.
        ## currently it has an internal routine that saves the results. I want to change that so the results are saved outside this function.
        ## another possibility would be to implement this function in a way that it iterativelly saves the results. 
        pass

    def find_pval(self):
        # this function is used in post-processing of main_parallel results.
        pass

    def sigma_discordance(self):
        # this function is used in post-processing of main_parallel results.
        pass

    def load_create_NS_file(self):
        # this function is used to load data_1,2_name if provided
        pass

    def surprise_function_call(self):
        # this is the main routine which also computes the PPD.
        pass

    ############# maybe I should also create a function that computes the PPD here. Instead of leaving it as a very short outside module.

    '''
    # These functions can be used to process nested sampling results but might not be needed for now
    def get_random_generator(self):
        pass

    def importance_weights(self):
        pass

    def resample_equal(self):
        # function to resample NS results
        pass
    '''

    # def analytical_kld(self):
        # might not be needed here
        # pass
