import numpy as np
import dynesty
from dynesty import utils as dyfunc

import jax.numpy as jnp
from jax import vmap
from jax import jit
from jax.scipy.linalg import solve_triangular, cholesky

from joblib import Parallel, delayed
from joblib import load, dump

from tqdm.auto import tqdm
import h5py
from scipy import special

# might add to this code...
import PPD

@jit
def logL2_jitted(data_2_model, chol_cov_2, D):
    """
    JIT-compatible log-likelihood computation.
    
    Args:
    - data_2_model: Model data (already computed from theta).
    - chol_cov_2: Cholesky factor of the covariance matrix.
    - D: Data space vector.

    Returns:
    - The log-likelihood.
    """
    n = D.shape[0]
    diff = data_2_model - D
    solve = solve_triangular(chol_cov_2, diff, lower=True)
    log_det_cov = 2 * jnp.sum(jnp.log(jnp.diagonal(chol_cov_2)))
    log_likelihood = -0.5 * (n * jnp.log(2 * jnp.pi) + log_det_cov + jnp.dot(solve, solve))
    
    return log_likelihood


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


    def __init__(self, logL1, data_2_model_fun, covariance_matrix_2, domain, data_2=None, data_1_name=None, data_2_name=None, init_NS = False, Nppd = None):
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
        
            self.chol_cov_2 = jnp.linalg.cholesky(covariance_matrix_2)
            self.res_1, self.res_2 = None, None
            self.PPD_chain = None

    def __initialize_NS__(self):
        self.res_1 = self.load_create_NS_file(self.data_1_name, self.logL1, self.ndim, self.domain)
        try:
            self.res_2 = self.load_create_NS_file(self.data_2_name, self.logP2, self.ndim, self.domain)
        except:
            print("Could neither load or create the posterior NS estimate for data D2.")

    def __initialize_PPD__(self, Nppd, n_jobs = 1, sample_size = 1):
        if self.res_1 is None:
            self.res_1 = self.load_create_NS_file(self.data_1_name, self.logL1, self.ndim, self.domain)
        self.th1_samples = self.sampler(self.res_1.samples_equal(), Nppd) # we take a subset of samples with size Nkld
        self.PPD_chain = PPD.create_ppd_chain(th1_samples=self.th1_samples, data_model_fun=self.data_2_model_fun, 
                                              cov_matrix = self.covariance_matrix_2, sample_size=sample_size, n_jobs=n_jobs)

    def logL2(self, theta, D):
        """
        Compute the log-likelihood of theta for the multivariate normal distribution.
        
        Args:
        - theta: parameter space vector (n-dimensional vector).
        - D: Data space vector

        Returns:
        - The log-likelihood of theta, D.
        """
        data_2_model = self.data_2_model_fun(theta)
        return logL2_jitted(data_2_model, self.chol_cov_2, D)
    
    def logP2(self, theta):
        return self.logL2(theta, self.data_2)

    def sampler(self, chain, nsample):
        '''
        Sample coordinate points on a Markov Chain Monte Carlo (MCMC) chain.

        This function is used to randomly select a specified number of samples from a given chain of equally weighted points. 
        It is useful in scenarios where you need a subset of samples from a larger MCMC chain for analysis 
        or further processing.

        Parameters:
        chain (array-like): A collection or array representing the MCMC chain. 
                            Each element in this array is a state or point in the MCMC chain.
        nsample (int): The number of samples to be drawn from the chain. This value should be 
                    a positive integer and less than or equal to the length of the chain.

        Returns:
        array-like: A subset of the chain, containing the randomly selected samples.
        '''
        index = np.arange(0, len(chain))
        rnd_el = np.random.choice(index, nsample)
        sampled = chain[rnd_el]
        return sampled

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
                        batch_size=1000, prior_transform='flat'):
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
        if (domain is not None) and (prior_transform == 'flat'):
            prior_volume = self.calculate_flat_prior_volume(domain)
        # else assume it to be one
        else:
            prior_volume = 1

        # Obtain equally weighted samples of distribution p(theta)
        samples_p = res_p.samples_equal()
        
        # Compute the evidence-normalized log-probability functions for both distributions
        logZp = res_p['logz'][-1] + jnp.log(prior_volume)
        
        # @jit
        def logP_norm(x):
            return logP(x) - logZp    

        logZq = res_q['logz'][-1] + jnp.log(prior_volume)
        
        # @jit
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

    def kld_worker(self, sample, logL_mock = None, mock1_NS_result= None, logP_1=None, ndim=None, 
                prior_transform='flat', n_effective = 20000, clip_range = [-1e16, 50000]):
        '''
        Worker function for parallelizing the evaluation of the Kullback-Leibler Divergence (KLD) distribution.

        Parameters:
        -----------
        sample : array-like
            A sample from the Posterior Predictive Distribution (PPD) for which KLD distribution is to be evaluated.
        
        logL_mock : callable
            Function to compute the log-likelihood for mock data. It should accept two inputs: (parameters, data).
        
        mock1_NS_result : object
            Result object from the first Nested Sampling run (output from a Dynesty nested sampling).
        
        logP_1 : callable, optional
            Log-posterior function for the data from the first dataset.
        
        domain : array-like, optional
            Parameter space domain. Required if `prior_transform` is 'flat', otherwise inferred from the `prior_transform` function.
        
        ndim : int, optional
            The number of dimensions in the parameter space. If not provided, it defaults to `self.ndim`.
        
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
        '''  

        if (ndim is None):
            ndim = self.ndim
        if logL_mock is None:
            logL_mock = self.logL2
        if mock1_NS_result is None:
            mock1_NS_result = self.res_1 

        # create mock data and run nested sampling
        # @jit
        def logpMock_2(theta):
            return self.logL2(theta, sample) # create full posterior distribution 2
        results2 = self.run_nested_sampling(logpMock_2, ndim=ndim, prior_transform=prior_transform, 
                                        domain=self.domain, n_effective=n_effective) # functions arguments are the best for SNIa chain.
        
        kld_return = self.KLD_numerical(results2, logpMock_2, mock1_NS_result, self.logL1, domain=self.domain, 
                        clip_range=clip_range, clip_values=True, progress=False, batch_size=1000, prior_transform='flat')
        
        return kld_return, sample

   
    def save_dict_to_hdf5(self, file_name, data_dict):
        """
        Save the contents of a dictionary to an HDF5 file, handling different data types.

        Parameters:
        - file_name: The name of the HDF5 file to be created.
        - data_dict: The dictionary to save, where keys are dataset names and values are the data.
        """
        with h5py.File(file_name, 'w') as hdf:
            for key, value in data_dict.items():
                # Check the type of the value to handle it appropriately
                if isinstance(value, np.ndarray):
                    # Save NumPy arrays directly as datasets
                    hdf.create_dataset(key, data=value)
                elif isinstance(value, jnp.ndarray):
                    # Convert JAX array to NumPy array and save it
                    hdf.create_dataset(key, data=np.array(value))
                elif isinstance(value, (float, int, np.float32, np.float64, np.int32, np.int64)):
                    # Save floats or integers as attributes of a dataset
                    # If the value is a NumPy scalar, convert to a native Python type
                    if hasattr(value, 'item'):
                        value = value.item()
                    dset = hdf.create_dataset(key, data=[])
                    dset.attrs['value'] = value
                else:
                    raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")
    

    def compute_kld_distribution(self, PPDsamples=None, logL_mock=None, mock1_NS_result=None, logP_1=None,  n_jobs=4, 
                  result_path=None, prior_transform='flat', n_effective=20000, clip_range = [-1e16, 50000]):
        """
        Parallel computation of KLD for PPD samples and saving results to HDF5. Will compute distribution Dkl(p2i, p1)

        Parameters:
        - PPDsamples: Collection of PPD samples to process.
        - logL_mock: Function for log-likelihood computation for mock data. A function that contains two inputs> (param, data)
        - mock1_NS_result: result object from Dynesty Nested Sampling for mock 1.
        - domain: Parameter space domain.
        - n_jobs (optional): Number of parallel jobs. Defaults to 2.
        - result_path (optional): Path to HDF5 file for results. Defaults to 'results.hdf5'.
        - prior_transform: currently a flat prior transform defined by domain. 
        """
        kld_results, ppdsample_results = [], []

        if PPDsamples is None:
            PPDsamples = self.PPD_chain
        if logL_mock is None:
            logL_mock = self.logL2
        if mock1_NS_result is None:
            mock1_NS_result = self.res_1
        if logP_1 is None:
            logP_1 = self.logL1

        results = Parallel(n_jobs=n_jobs)(delayed(self.kld_worker)
                                        (sample, logL_mock, mock1_NS_result, logP_1, self.ndim, prior_transform, n_effective, clip_range)
                                        for i, sample in enumerate(tqdm(PPDsamples, desc="Iterating over the PPD")))

        for kld, ppdsample in results:
            kld_results.append(kld)
            ppdsample_results.append(ppdsample) 

        results_dict = {"kld_dist" : np.array(kld_results), "ppd_sample" : np.array(ppdsample_results)}

        if result_path is not None:
            self.save_dict_to_hdf5(result_path, results_dict)
        
        return kld_results   

    def find_pval(self, Sdist, S, verbose = 0):
        """
        Calculate the p-value from a distribution of surprise values given an observed surprise.

        Parameters:
        - Sdist (ndarray): An array of surprise values from simulations or a distribution.
        - S (float): The observed surprise value for which the p-value is to be calculated.

        Returns:
        - pval (float): The calculated p-value indicating the probability of observing a surprise at least as extreme as S.
        """
        pval = Sdist[Sdist > S].size/Sdist.size
        if verbose>0:
            print("p-value = {:.1f} %".format(100*pval))
        return pval

    def sigma_discordance(self, p_value):
        return np.sqrt(2)*special.erfinv(1-p_value)

    def load_create_NS_file(self, data_1_name, logL1, ndim, domain,  n_effective=15000, dlogz=0.5):
        '''
        This function checks for existing NS results and if they don't exist, it creates one using logL1 and domain.
        '''
        print(70*'-')
        print("Loading posterior data")
        print(70*'-')
        try: 
            ## loading pre-made Nested Sampling run
            res_1 = load(data_1_name)
            print("Data loaded sucessfully!")
        except:
            print(70*'-')
            print("Loading failed!")
            print(70*'-')

            print("Running Nested Sampling...")
            print(70*'-')
            # any nested sampling routine will be fine, as long as it has the method .samples_equal() or equivalent, to obtain equally weighted samples. 
            res_1 = self.run_nested_sampling(logL1, ndim, domain=domain, print_progress=True, n_effective=n_effective, dlogz=dlogz)
            print("Run completed sucessfully.")
            
            # if data_1_name is not None:
            try:
                print("Saving ", data_1_name)
                dump(res_1, data_1_name)
            except:
                pass
        return res_1

    def surprise_function_call(self, Nkld, result_path = None, n_effective= 15000, n_jobs=-1, verbose=1):
        '''
        Compute the Kullback-Leibler Divergence (KLD) distribution and optionally calculate the surprise statistic 
        if a second dataset is provided.

        This function loads or creates posterior samples using Nested Sampling (NS) for a first dataset (D1) 
        and then load or create a Posterior Predictive Distribution (PPD). If a second dataset (D2) is provided, 
        it also computes the KLD between the posterior distributions of D1 and D2, returning the surprise statistic.

        Parameters:
        -----------
        Nkld : int
            The number of KLD samples to be drawn from the Posterior Predictive Distribution (PPD).
        
        result_path : str
            The path where the results will be saved (in HDF5 format).
        
        n_effective : int, optional (default=15000)
            The number of effective samples to target for the nested sampling run.

        n_jobs : int, optional (default=-1)
            The number of parallel jobs to run. Set to -1 to use all available cores.

        verbose : int, optional (default=1)
            Level of verbosity. Set to 0 for silent mode, higher values for more verbose output.

        Returns:
        --------
        results_dic : dict
            A dictionary containing the computed KLD values and the surprise statistic (if applicable). 
            The contents of the dictionary vary depending on whether a second dataset is provided. The keys include:
            - 'S': The computed surprise statistic (if D2 is provided).
            - 'S_dist': The distribution of surprise statistics.
            - 'kld21': KLD(p2 | p1), i.e., the KLD between the posterior distributions of D1 and D2 (if D2 is provided).
            - 'kld_exp': The expected KLD (mean value of the KLD distribution).
            - 'kld_dist': The distribution of KLD samples.
            - 'p_value': The p-value associated with the surprise statistic (if D2 is provided).
            - 'sigma_discordance': The sigma-level discordance between the two datasets (if D2 is provided).

        Notes:
        ------
        - The function first computes the KLD between the posterior distribution and the PPD for dataset D1. 
        - If a second dataset (D2) is provided, the KLD between the posterior distributions of D1 and D2 is also computed.
        - The function calculates the surprise statistic using the expected KLD and compares it to the KLD of D2.
        '''
        
        print("Handling dataset 1...")
        print(70*"_")
        ############ loading/creating mock 1 and 2 ############
        if self.res_1 is None:
            self.__initialize_NS__()
        print("Done!")
        
        print("")
        print("Handling posterior predictive distribution PPD(D2|D1) ...")
        print(70*"_")
        
        ############ create posterior predictive distribution ############
        if self.PPD_chain is None:
            self.__initialize_PPD__(Nkld)
        else:
            Nkld = self.PPD_chain.shape[0]
            print("Will sample KLD the same size as PPD.\nNkld = ", Nkld)

        print("Handling KLD distribution...")
        print(70*"_")
        kld_samples = self.compute_kld_distribution(self.PPD_chain, self.logL2, self.res_1, logP_1=self.logL1, 
                                                    n_jobs=n_jobs, n_effective=n_effective)

        kld_array = jnp.array(kld_samples)
        # kld_array = np.array(kld_samples)
        kld_exp = kld_array.mean()
        S_dist = kld_array - kld_exp

        # if data 2 is provided
        if self.res_2 is not None:
            kld_value = self.KLD_numerical(self.res_2, self.logP2, self.res_1, self.logL1, domain = self.domain)
            S = kld_value - kld_exp
            p_value = self.find_pval(S_dist, S, verbose = 0)
            sigma_disc = self.sigma_discordance(p_value)
            if verbose>0:
                print("S = {:.2f} nats".format(S))
                print("<KLD> = {:.2f} nats".format(kld_exp))
                print("KLD = {:.2f} nats".format(kld_value))
                print("p-val = {:.2f} nats".format(p_value))
            results_dic = {"domain" : self.domain, "S" : S, "S_dist": S_dist, "kld21" : kld_value, "kld_exp":kld_exp, "kld_dist":kld_array, "p_value":p_value, 'sigma_discordance':sigma_disc}
        else:
            if verbose>0:
                print("<KLD> = {:.2f} nats".format(kld_exp))
            results_dic = {"domain" : self.domain, "S_dist": S_dist, "kld_exp":kld_exp, "kld_dist":kld_array}
        if result_path is not None:
            print("Saving results to ", result_path)
            self.save_dict_to_hdf5(result_path, results_dic)
        return results_dic
    

    ################################################ create PPD ################################################
    ############################################################################################################
    def generate_samples(self, theory_vec, L, nz, sample_size):
        """
        Generate Gaussian samples for a given theory vector.
        """
        z = np.random.normal(size=(sample_size, nz))
        gaussian_samples = theory_vec + np.dot(z, L.T)
        return gaussian_samples

    def create_ppd_chain(self, th1_samples, data_model_fun = None, cov_matrix = None, sample_size=10, n_jobs=4):
        """
        Create the Posterior Predictive Distribution (PPD) chain. This function assumes a Gaussian likelihood for each observed datavector, it then takes 
        'sample_size' samples from each Gaussian and join them togheter in a matrix where each line is a data sample from the PPD.
        
        Parameters:
        - th1_samples: The samples from the parameter space of the posterior distribution. Given as input to create_mock function. Should be (h, Om, Ok, w)
        - data_model_fun: A function that takes an input the cosmological parameters theta as an array and returns a data vector.
        - cov_matrix: Covariance matrix of predicted distribution. Standard matrix is Pantheon covariance.
        - sample_size: The number of samples to be taken from the Gaussian distribution for each theory vector.
        - n_jobs: The number of parallel jobs to run.
        
        Returns:
        - The PPD chain as a numpy array.
        """

        # if th1_samples is None:
            # th1_samples = self.res_1.samples_equal()
        if data_model_fun is None:
            data_model_fun = self.data_2_model_fun
        if cov_matrix is None:
            cov_matrix = self.covariance_matrix_2

        print("Evaluating theory from sample distribution p1")
        D1_th1_samples = np.zeros((th1_samples.shape[0], cov_matrix.shape[0]))
        for i, th in enumerate(tqdm(th1_samples, desc="Generating theory vectors")):
            D1_th = data_model_fun(th)
            D1_th1_samples[i] = D1_th

        print("Sampling the Posterior Predictive Distribution...")
        ntheta, nz = D1_th1_samples.shape

        # Perform Cholesky decomposition once, outside the loop
        L = cholesky(cov_matrix, lower=True)

        # Parallel execution
        samples_list = Parallel(n_jobs=n_jobs)(
            delayed(self.generate_samples)(theory_vec, L, nz, sample_size) for theory_vec in tqdm(D1_th1_samples, desc="Sampling PPD")
        )

        # Concatenate all the sample arrays into one array
        PPD_chain = np.vstack(samples_list)
        return PPD_chain