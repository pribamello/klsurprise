def surprise_statistics(logL1, data_2_model_fun, covariance_matrix_2, domain, data_2=None, data_1_name=None, data_2_name=None):
    """
    Computes the surprise statistics between two datasets or models using Nested Sampling.

    This function calculates the surprise (S) distribution and, optionally, the S value between two datasets or models
    by performing Nested Sampling. It allows for comparing the likelihoods and data models within a specified parameter
    domain using a top-hat prior.

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
        # Initialize the class with the provided parameters
        self.logL1 = logL1
        self.data_2_model_fun = data_2_model_fun
        self.covariance_matrix_2 = covariance_matrix_2
        self.domain = domain
        self.data_2 = data_2
        self.data_1_name = data_1_name
        self.data_2_name = data_2_name

def sampler(self):
    # maybe in a helper file or outisde this class
    pass

def calculate_flat_prior_volume(self): ################################## ok
    # necessary to normalize the NS run
    pass

# This is really not necessary. It's just one...
# def calculate_gaussian_prior_volume(self):
    # necessary to normalize the NS run.
    # pass

def prior_transform_flat(self):
    # implemented already
    # flat prior transform
    pass

def prior_transform_gaussian(self):
    # not implemented
    # shoudn't be hard to implement 
    # flat prior transform
    pass

def run_nested_sampling(self): ################################## ok
    # already implemented, working fine.
    pass

# def analytical_kld(self): 
    # might not be needed here
    # pass

def process_batch(self):
    # this function will be used to compute kld by means of monte carlo integration.
    # it processes batches of information and serialize computation using jax
    # this function speeds up the computation of KLD but limits the code usage to only jax compatible functions.
    pass

def KLD_numerical(self)
    # old version was: compute_KLD_MCMC():
    # currently limited to likelihoods that use jax
    # If I can fix this I can make this code work with any likelihood
    # there might be a way to optmize this function in parallel without jax.
    # maybe using mpi 
    pass

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