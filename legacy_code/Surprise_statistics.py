import h5py
import numpy as np
from scipy.stats import norm
from tqdm.auto import tqdm ########################## this line might raise an error when using this outside jupyter lab
import dynesty
from dynesty import utils as dyfunc

from numpy.linalg import inv, slogdet
from joblib import Parallel, delayed
import os
import warnings

import joblib
from jax import vmap
from jax import jit
import jax.numpy as jnp

# custom functions 
import ferramentas as fe # used in sample_logP

homedir = os.getcwd()


################### draw random samples from chain
def sampler(chain, nsample):
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

################### prior volume
from scipy.stats import norm

def calculate_flat_prior_volume(domain):
    # Calculate the lengths of each dimension
    lengths = np.diff(domain, axis = 1).T
    # The volume is the product of the lengths of the sides
    volume = np.prod(lengths)
    return volume


# for the BAO+BBN theoretical prior
mu_wb = 0.02218
sigma_wb = 0.00055
def calculate_gaussian_prior_volume(mu, sigma, num_sigma=3):
    # Calculate the CDF for both the upper and lower bounds
    lower_bound_cdf = norm.cdf(mu - num_sigma * sigma, loc=mu, scale=sigma)
    upper_bound_cdf = norm.cdf(mu + num_sigma * sigma, loc=mu, scale=sigma)
    
    # The volume under the Gaussian curve within ± num_sigma * sigma
    return upper_bound_cdf - lower_bound_cdf

def run_nested_sampling(loglikelihood, ndim, domain=None, prior_transform="flat", static_NS = False, dynamic_NS = True, dlogz=0.5, 
                        print_progress = False, ignore_warnings=False, nlive=1000, nlive_batch=200, maxiter=600000, maxbatch=50, use_stop = True,
                        bootstrap=0, n_effective=20000, **kwargs):
    """
    Performs a Nested Sampling run to estimate the Bayesian evidence and obtain posterior samples for a given loglikelihood function and parameter space. 
    This function is designed to be flexible, allowing for customization of the domain, prior transformation, and handling of warnings.

    Parameters:
        loglikelihood (callable): The log-likelihood function that accepts an array of parameter values and returns the log-likelihood. 
                                  This function should be defined by the user and is a critical component of the Nested Sampling algorithm.
        ndim (int): The number of dimensions of the parameter space, i.e., the number of parameters being estimated.
        domain (np.array, optional): A 2D numpy array specifying the domain (parameter ranges) for each dimension of the parameter space. 
                                     The shape of this array should be (ndim, 2), where each row corresponds to the minimum and maximum 
                                     values for a parameter. If `None`, it is assumed that domain information is provided by the `prior_transform` argument.
        prior_transform (str or callable, optional): A function that transforms samples from the unit cube to the parameter space defined by the domain. 
                                                     If set to "flat", a uniform prior over the specified domain is used. Custom transformations can be provided 
                                                     by passing a callable that implements the desired transformation. Defaults to "flat". 
        print_progress (bool, optional): If `True`, progress information is printed during the Nested Sampling run. Defaults to `False`.
        ignore_warnings (bool, optional): If `True`, suppresses warning messages. This can be useful for clean output, but caution is advised. 
                                          Defaults to `False`.
        bootstrap (int, optional): Specifies the number of bootstrapping iterations for error estimation. Defaults to `0`, indicating no bootstrapping.
        **kwargs: Additional keyword arguments to be passed directly to the `dynesty.NestedSampler` constructor. This allows for further customization of the sampler.

    Returns:
        results (object): An object containing the results of the Nested Sampling run. This object includes attributes such as posterior samples, 
                          log evidence, and information on the sampling efficiency. The exact contents can vary depending on the version and configuration of `dynesty`.

    Note:
        This function requires the `dynesty` package for Nested Sampling. Ensure that `dynesty` is installed and properly configured in your environment.
    WARNING!!!!!!!:
        By general rule, if a "prior_transform" is provided, "domain" should be set to None.
    """
    if ignore_warnings:
        warnings.filterwarnings('ignore') 
    else:
        pass
    if domain is not None:
        # Ensure the domain is a numpy array with the right shape
        domain = np.asarray(domain)
        assert domain.shape == (ndim, 2), "Domain must be an array with shape (ndim, 2)."

        if prior_transform=="flat":
            # Define the prior transform function
            def prior_transform(utheta):
                """Transforms samples `u` drawn from the unit cube to samples from the domain."""
                return domain[:, 0] + utheta * (domain[:, 1] - domain[:, 0])
    if static_NS:
        # "Static" nested sampling.
        # accuratelly measures evidence but it's not good in sampling the posterior.
        sampler = dynesty.NestedSampler(loglikelihood, prior_transform, ndim, nlive=nlive, bootstrap=bootstrap, **kwargs)
        sampler.run_nested(dlogz=dlogz, print_progress = print_progress)
        sresults = sampler.results
        results = sresults
    if dynamic_NS:
        # "Dynamic" nested sampling.
        # correctly sample the posterior
        dsampler = dynesty.DynamicNestedSampler(loglikelihood, prior_transform, ndim, bootstrap=bootstrap)
        dsampler.run_nested(dlogz_init=dlogz, print_progress = print_progress, nlive_init=nlive, 
                            nlive_batch=nlive_batch, maxbatch=maxbatch, maxiter=maxiter, use_stop=use_stop, n_effective=n_effective)
        dresults = dsampler.results
        results = dresults
    if dynamic_NS & static_NS:
        # Combine results from "Static" and "Dynamic" runs.
        results = dyfunc.merge_runs([sresults, dresults])
        
    return results


def analytical_kld(mu1, sigma1, mu2, sigma2):
    """
    Compute the Kullback-Leibler Divergence between two multivariate Gaussian distributions.

    Parameters:
        mu1 (array): Mean of the first Gaussian.
        sigma1 (array): Covariance matrix of the first Gaussian.
        mu2 (array): Mean of the second Gaussian.
        sigma2 (array): Covariance matrix of the second Gaussian.

    Returns:
        kld (float): The analytical KLD from the first Gaussian to the second Gaussian.
    """
    sigma2_inv = inv(sigma2)
    k = len(mu1)
    term1 = np.trace(sigma2_inv.dot(sigma1))
    term2 = (mu2 - mu1).T.dot(sigma2_inv).dot(mu2 - mu1)
    term3 = -k
    term4 = slogdet(sigma2)[1] - slogdet(sigma1)[1]
    kld = 0.5 * (term1 + term2 + term3 + term4)
    return kld


'''
###### Not working for some reason
def compute_kld_MCMC(res_p, logpost_q, prior_volume, clip_range = [-1e8, 100]):
    """
    Compute the Kullback-Leibler Divergence (KLD) between two distributions,
    where the second distribution's normalized log-posterior function is available.
    
    Parameters:
    - res_1: Dynesty Nested sampling result object for distribution p.
    - logpost_q: Function to compute log-posterior for samples under distribution q. This function must be normalized in parameter space!
    - prior_volume: used to evaluate evidence of logpost_p.
    - clip_range: min-max clip range used to avoid overflow.
    
    Returns:
    - KLD from distribution p to q.
    """
    # Extract samples and log weights from res_1
    samples_p, log_prob_p = resample_equal(res_p)
    logZ = res_p['logz'][-1]+np.log(prior_volume)
    log_prob_p = log_prob_p - logZ # normalization of loglikelihood
    
    # Evaluate log-likelihood under p for each sample in p
    # log_prob_p = np.array([logpost_p(sample) for sample in samples_p])

    # Evaluate log-likelihood under q for each sample in p
    log_prob_q = np.array([logpost_q(sample) for sample in samples_p])

    # clip values to avoid overflow
    log_prob_p = np.clip(log_prob_p, a_min=clip_range[0], a_max = clip_range[1])
    log_prob_q = np.clip(log_prob_q, a_min=clip_range[0], a_max = clip_range[1])

    # Calculate KLD
    kld = np.mean(log_prob_p - log_prob_q)
    return kld
'''

def process_batch(logpost, coordinates, batch_size = 1000, progress = False):
    vmap_logpost = vmap(logpost)
    flat_coordinates = coordinates
    # Define batch size, choose a size that fits in your memory
    logpost_matrix = []
    for i in tqdm(range(0, flat_coordinates.shape[0], batch_size), desc="Processing batches") if progress else range(0, flat_coordinates.shape[0], batch_size):
        batch_coords = flat_coordinates[i:i + batch_size]
        logpost_matrix.append(vmap_logpost(batch_coords))
    logpost_matrix = jnp.concatenate(logpost_matrix)
    return logpost_matrix

def compute_KLD_MCMC(res_p, logP, res_q, logQ, domain=None, prior_transform = 'flat', clip_range = [-1e16, 5000], clip_values = True, progress=True, batch_size = 1000, samples_p = None):

    # there is a better way to sample_p and get the values logP with the sampling process, 
    # but I can't seem to get it working so this is a dumb and less efficient way.

    #########################################################################
    if (domain is not None) and (prior_transform=='flat'):
        ############ FLAT PRIOR VOLUME
        prior_volume = calculate_flat_prior_volume(domain)
    elif callable(prior_transform):
        ############ HARD CODED. Only works for BAO+BBN flat+gaussian prior.
        prior_volume = calculate_flat_prior_volume(domain)*calculate_gaussian_prior_volume(mu_wb, sigma_wb, num_sigma = 5)

    if samples_p is None:
        samples_p = res_p.samples_equal()
    else:
        samples_p = samples_p # this allow us to provide a marginalized function call LogP and a marginalized chain samples_p.
        # the evidence will still be the same so the results res_p,q can be of a run with say (h, Om, Ok, w) and the function call 
        # can be of (Om, Ok, w), marginalized over h.
    logZp = res_p['logz'][-1]+np.log(prior_volume)
    @jit
    def logP_norm(x):
        return logP(x) - logZp    

    logZq = res_q['logz'][-1]+np.log(prior_volume)
    @jit
    def logQ_norm(x):
        return logQ(x) - logZq 
        
    log_prob_p = process_batch(logP_norm, samples_p, progress=progress, batch_size=batch_size)
    log_prob_q = process_batch(logQ_norm, samples_p, progress=progress, batch_size=batch_size)

    if clip_values:
        # clip values to avoid overflow
        log_prob_p = np.clip(log_prob_p, a_min=clip_range[0], a_max = clip_range[1])
        log_prob_q = np.clip(log_prob_q, a_min=clip_range[0], a_max = clip_range[1])

    kld = (log_prob_p-log_prob_q).mean()

    return kld
##### used to resample Dynesty Nested Sampling run for equal-weighted particles. Also returns the loglikelihood of the sample instead of just the sample
##### this function is really the same as .samples_equal() method but it also returns logL value for a faster evaluation of KLD

def get_random_generator(seed=None):
    """
    Return a random generator (using the seed provided if available)
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.Generator(np.random.PCG64(seed))

def importance_weights(logwt):
    """
    Return the importance weights for the each sample.
    """
    wt = np.exp(logwt)
    wt = wt / wt.sum()
    return wt

def resample_equal(results, rstate=None):
    """
    Resample a set of points from the weighted set of inputs
    such that they all have equal weight. The points are also
    randomly shuffled.

    Each input sample appears in the output array either
    `floor(weights[i] * nsamples)` or `ceil(weights[i] * nsamples)` times,
    with `floor` or `ceil` randomly selected (weighted by proximity).

    Parameters
    ----------
    samples : `~numpy.ndarray` with shape (nsamples,)
        Set of unequally weighted samples.

    weights : `~numpy.ndarray` with shape (nsamples,)
        Corresponding weight of each sample.

    rstate : `~numpy.random.Generator`, optional
        `~numpy.random.Generator` instance.

    Returns
    -------
    equal_weight_samples : `~numpy.ndarray` with shape (nsamples,)
        New set of samples with equal weights in random order.

    Examples
    --------
    x = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    w = np.array([0.6, 0.2, 0.15, 0.05])
    utils.resample_equal(x, w)
    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.],
           [ 3.,  3.]])

    Notes
    -----
    Implements the systematic resampling method described in `Hol, Schon, and
    Gustafsson (2006) <doi:10.1109/NSSPW.2006.4378824>`_.
   """
    
    if rstate is None:
        rstate = get_random_generator()
    if type(rstate) is int:
        rstate = get_random_generator(rstate)

    samples = results.samples
    logL    = results.logl
    weights = importance_weights(results.logwt)

    SQRTEPS = np.sqrt(float(np.finfo(np.float64).eps))
    cumulative_sum = np.cumsum(weights)
    if abs(cumulative_sum[-1] - 1.) > SQRTEPS:
        # same tol as in numpy's random.choice.
        # Guarantee that the weights will sum to 1.
        warnings.warn("Weights do not sum to 1 and have been renormalized.")
    cumulative_sum /= cumulative_sum[-1]
    # this ensures that the last element is strictly == 1

    # Make N subdivisions and choose positions with a consistent random offset.
    nsamples = len(weights)
    positions = (rstate.random() + np.arange(nsamples)) / nsamples
    
    # Resample the data.
    idx = np.zeros(nsamples, dtype=int)
    i, j = 0, 0
    while i < nsamples and j < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    idx = rstate.permutation(idx)
    return samples[idx], logL[idx]


############################################################################################
############################################################################################
########################## routine to run the code in parallel #############################
################ nova implementação focada em Pantheon+&SH0ES vs. BAO+BBN ##################
############################################################################################
def kld_worker(sample, logL_mock, mock1_NS_result, logP_1=None, domain=None, ndim=None, 
               prior_transform='flat', mode = 'MCMC', ignore_warnings=False, 
               save_mock = False, save_mock_path = '../PPD_mocks_SNIa_BAO/', save_mock_prefix = 'Mock_PPD_generic_', save_mock_iterator = 0,
               n_effective = 20000):
    '''
    Worker function for parallelizing the evaluation of KLD distribution.

    Parameters:
    - sample: The sample from the PPD for which KLD is to be evaluated.
    - logL_mock: Function to compute log-likelihood for mock data. A function that contains two inputs> (param, data)
    - mock1_NS_result: result object from Dynesty Nested Sampling.
    - domain: Parameter space domain.
    - n_components (optional): Number of components for Gaussian mixture model. If None, no GMM is evaluated.
    - mode: string, 'MCMC', 'GMM' or 'both'.
    Returns:
    - Tuple of (kld, kld_single, kdl_analytical, sample).

    Notes: domain information can be contained inside the prior_transform function. 
           domain information is only requested for a flat prior. 
    '''     
    if (ndim is None)&(domain is not None):
        ndim = domain.shape[0]
        domain_pass = domain
    if callable(prior_transform):
        domain_pass = None
    # create mock data and run nested sampling 
    logpMock_2 = lambda u : logL_mock(u, sample) # create full posterior distribution 2
    results2 = run_nested_sampling(logpMock_2, ndim=ndim, prior_transform=prior_transform, 
                                     domain=domain_pass, ignore_warnings=ignore_warnings, n_effective=n_effective) # functions arguments are the best for SNIa chain.
    if save_mock:
        save_name = save_mock_path + save_mock_prefix + '{}.pkl'.format(save_mock_iterator)
        joblib.dump(results2, save_name)

    if mode=='MCMC':
        if logP_1 is None:
            print("Please make sure the log-posterior of data-1 is a valid function for MCMC mode!")
        
        kld_mcmc = compute_KLD_MCMC(results2, logpMock_2, mock1_NS_result, logP_1, domain = domain, prior_transform=prior_transform,
                                    clip_range = [-1e8, 1500], clip_values = True, progress=False, batch_size = 1000)
        kld_return = kld_mcmc
    
    if not save_mock:
        return kld_return, sample
    else: # se salvar o mock, salva também uma lista única estruturada com um identificador de arquivo.
        return kld_return, sample, save_mock_iterator
               

def write_results_to_hdf5(file_path, kld_data, kld_analytical_data=None, 
                          ppdsample_data=None, domain=None, mock1_chain=None, save_mock_iterator_list = None):
    """
    Saves KLD results, PPD samples, domain, mock1_chain, and n_components to an HDF5 file.
    Makes use of optional arguments to only save provided datasets.

    Parameters:
    - file_path (str): Path to the HDF5 file for storing results.
    - kld_data (list): List of Kullback-Leibler Divergence results.
    - kld_gmm_data (list, optional): List of KLD with GMM results.
    - kld_analytical_data (list, optional): List of KLD Analytical results.
    - ppdsample_data (list, optional): List of corresponding PPD samples.
    - domain (array, optional): Parameter space domain.
    - mock1_chain (array, optional): MCMC chain for the first mock dataset.
    - n_components (int, optional): Number of components for Gaussian mixture model.
    """
    with h5py.File(file_path, 'a') as hdf_file:
        # Always save kld_data
        if 'kld' not in hdf_file:
            hdf_file.create_dataset('kld', shape=(0,), maxshape=(None,), dtype=np.float32)
        hdf_file['kld'].resize((hdf_file['kld'].shape[0] + len(kld_data),))
        hdf_file['kld'][-len(kld_data):] = kld_data

        if kld_analytical_data is not None:
            if 'kld_analytical' not in hdf_file:
                hdf_file.create_dataset('kld_analytical', shape=(0,), maxshape=(None,), dtype=np.float32)
            hdf_file['kld_analytical'].resize((hdf_file['kld_analytical'].shape[0] + len(kld_analytical_data),))
            hdf_file['kld_analytical'][-len(kld_analytical_data):] = kld_analytical_data

        if ppdsample_data is not None:
            ppdsize = ppdsample_data[0].shape[0]
            if 'ppdsample' not in hdf_file:
                hdf_file.create_dataset('ppdsample', shape=(0, ppdsize), maxshape=(None, ppdsize), dtype=np.float32, chunks=True)
            hdf_file['ppdsample'].resize((hdf_file['ppdsample'].shape[0] + len(ppdsample_data), ppdsize))
            hdf_file['ppdsample'][-len(ppdsample_data):] = ppdsample_data

        try:
            # Save domain, mock1_chain, n_components, and z1_list if provided
            if domain is not None:
                # Assuming 'hdf_file' is your HDF5 file object and 'domain' is the dataset you want to create
                if 'domain' in hdf_file:
                    del hdf_file['domain']  # Delete the existing dataset
                hdf_file.create_dataset('domain', data=np.array(domain), overwrite=True)

       
            if mock1_chain is not None:
                if 'mock1_chain' in hdf_file:
                    del hdf_file['mock1_chain']  # Delete the existing dataset    
                hdf_file.create_dataset('mock1_chain', data=np.array(mock1_chain), overwrite=True)
            
            if save_mock_iterator_list is not None:
                hdf_file.create_dataset('save_mock_iterator_list', data=np.array(save_mock_iterator_list))
        except: 
            pass

def main_parallel(PPDsamples, logL_mock, mock1_NS_result, logP_1=None, domain=None, mode='MCMC', ndim=None, n_jobs=4, 
                  result_path='results.hdf5', prior_transform='flat', ignore_warnings=False, 
                  save_mock=False, save_mock_path = '../PPD_mocks_SNIa/', save_mock_prefix = 'Mock_PPD_generic_', n_effective=20000):
    """
    Parallel computation of KLD for PPD samples and saving results to HDF5.

    Parameters:
    - PPDsamples: Collection of PPD samples to process.
    - logL_mock: Function for log-likelihood computation for mock data. A function that contains two inputs> (param, data)
    - mock1_NS_result: result object from Dynesty Nested Sampling.
    - domain: Parameter space domain.
    - n_jobs (optional): Number of parallel jobs. Defaults to 2.
    - result_path (optional): Path to HDF5 file for results. Defaults to 'results.hdf5'.
    - n_components (optional): Number of components for Gaussian mixture model. Defaults to 10.
    """
    kld_results, ppdsample_results = [], []
    # if domain is not None and ndim is None:
        # domain_pass = np.atleast_2d(domain)
    # else:
    domain_pass = domain
    
    mock_1_chain = mock1_NS_result.samples_equal()

    '''
    results = Parallel(n_jobs=n_jobs)(delayed(kld_worker)
                                      (sample, logL_mock, mock1_NS_result, logP_1, gmm1, domain_pass, ndim, prior_transform, mode, n_components, ignore_warnings) 
                                      for sample in tqdm(PPDsamples, desc="Iterating over the PPD"))
    '''
    
    results = Parallel(n_jobs=n_jobs)(delayed(kld_worker)
                                      (sample, logL_mock, mock1_NS_result, logP_1, domain_pass, ndim, prior_transform, mode, 
                                       ignore_warnings, save_mock, save_mock_path, save_mock_prefix, i, n_effective)
                                      for i, sample in enumerate(tqdm(PPDsamples, desc="Iterating over the PPD")))

    if not save_mock:
        for kld, ppdsample in results:
            kld_results.append(kld)
            ppdsample_results.append(ppdsample) 

        if result_path is not None:
            write_results_to_hdf5(result_path, kld_results, ppdsample_data=ppdsample_results, domain=domain, 
                                mock1_chain=mock_1_chain)
    else:
        save_mock_iterator_list = []
        for kld, ppdsample, save_mock_iterator in results:
            kld_results.append(kld)
            ppdsample_results.append(ppdsample)
            save_mock_iterator_list.append(save_mock_iterator)

        if result_path is not None:
            write_results_to_hdf5(result_path, kld_results, ppdsample_data=ppdsample_results, domain=domain, 
                                mock1_chain=mock_1_chain, save_mock_iterator_list=save_mock_iterator_list)   
        
    return kld_results
    # return results
    

##################################### wrap up functions
def find_pval(Sdist, S, verbose = 0):
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

from scipy import special
def sigma_discordance(p_value):
    return np.sqrt(2)*special.erfinv(1-p_value)

def save_dict_to_hdf5(file_name, data_dict):
    """
    Save the contents of a dictionary to an HDF5 file.

    Parameters:
    - file_name: The name of the HDF5 file to be created.
    - data_dict: The dictionary to save, where keys are dataset names and values are the data.
    """
    with h5py.File(file_name, 'w') as hdf:
        for key, value in data_dict.items():
            # Save each item in the dictionary as a dataset
            hdf.create_dataset(key, data=value)
            
def load_create_NS_file(data_1_name, logL1, ndim, domain,  n_effective=15000, dlogz=0.5):
    print(70*'-')
    print("Loading posterior data")
    print(70*'-')
    try: 
        ## loading pre-made Nested Sampling run
        res_1 = joblib.load(data_1_name)
        print("Data loaded sucessfully!")
    except:
        print(70*'-')
        print("Loading failed!")
        print(70*'-')

        print("Running Nested Sampling...")
        print(70*'-')
        # any nested sampling routine will be fine, as long as it has the method .samples_equal() or equivalent, to obtain equally weighted samples. 
        res_1 = run_nested_sampling(logL1, ndim, domain=domain, print_progress=True, n_effective=n_effective, dlogz=dlogz)
        print("Run completed sucessfully.")
        
        # if data_1_name is not None:
        try:
            print("Saving ", data_1_name)
            joblib.dump(res_1, data_1_name)
        except:
            pass
    return res_1

import PPD
def surprise_function_call(logL1, logL2, data2_model_fun, covariance_matrix_2, domain, Nkld, 
                           result_path, data_1_name = None, n_effective= 15000, n_jobs=-1, data_2_vec = None, data_2_name = None, verbose=1):
    # logL1 --> a callable function of theta (parameter)
    # logL2 --> a callable function of theta (parameter) and D (data).
    # data_2_vec is yet to be added. If provided then function should also compute KLD(p2|p1) 
    # and return the surprise statistic value  
    ndim = domain.shape[0]

    if verbose>0:
        print_progress = True
    else:
        print_progress = False

    ############ loading/creating mock 1 ############
    res_1 = load_create_NS_file(data_1_name, logL1, ndim, domain)
    
    # This method is a particularity of Dynesty, but can be easily implemented for any other NS package.
    # Equal weighted samples 
    eq_samples_1 = res_1.samples_equal()
    print("Done!")
    
    ############ create posterior predictive distribution ############
    # parameter space samples of posterior distribution p(theta|D1)
    th1_samples = sampler(eq_samples_1, Nkld) # we take a subset of samples with size Nkld
    PPD_chain = PPD.create_ppd_chain(th1_samples=th1_samples, data_model_fun=data2_model_fun, cov_matrix = covariance_matrix_2, sample_size=1, n_jobs=n_jobs)

    kld_samples = main_parallel(PPD_chain, logL2, res_1, logP_1=logL1,  domain=domain, mode='MCMC', n_jobs=n_jobs, 
              result_path=None, ignore_warnings=True, 
              n_effective=n_effective)

    kld_array = np.array(kld_samples)
    kld_exp = kld_array.mean()
    S_dist = kld_array - kld_exp

    # if data 2 is provided
    if data_2_vec is not None:
        logP2 = lambda theta : logL2(theta, data_2_vec)

        # load or create data_2 posterior chain with NS
        if data_2_name is not None:
            res_2 = load_create_NS_file(data_2_name, logP2, ndim, domain)
        else:
            res_2 = run_nested_sampling(logP2, ndim, domain=domain, print_progress=True)
        
        kld_value = compute_KLD_MCMC(res_2, logP2, res_1, logL1, domain = domain)
        S = kld_value - kld_exp
        p_value = find_pval(S_dist, S, verbose = verbose)
        sigma_disc = sigma_discordance(p_value)
        if verbose>0:
            print("S = {:.2f} nats".format(S))
            print("<KLD> = {:.2f} nats".format(kld_exp))
            print("KLD = {:.2f} nats".format(kld_value))
        results_dic = {"S" : S, "S_dist": S_dist, "kld21" : kld_value[0], "kld_exp":kld_exp, "kld_dist":kld_array, "p_value":p_value, 'sigma_discordance':sigma_discordance}
    else:
        if verbose>0:
            print("<KLD> = {:.2f} nats".format(kld_exp))
        results_dic = {"S_dist": S_dist, "kld_exp":kld_exp, "kld_dist":kld_array}
    if result_path is not None:
        save_dict_to_hdf5(result_path, results_dic)
    return results_dic