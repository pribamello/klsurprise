import numpy as np
from jax.scipy.linalg import cholesky
from joblib import Parallel, delayed
from tqdm.auto import tqdm

def generate_samples(theory_vec, L, nz, sample_size):
    """
    Generate Gaussian samples for a given theory vector.
    """
    z = np.random.normal(size=(sample_size, nz))
    gaussian_samples = theory_vec + np.dot(z, L.T)
    return gaussian_samples

def create_ppd_chain(th1_samples, data_model_fun, cov_matrix, sample_size=10, n_jobs=4):
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
        delayed(generate_samples)(theory_vec, L, nz, sample_size) for theory_vec in tqdm(D1_th1_samples, desc="Sampling PPD")
    )

    # Concatenate all the sample arrays into one array
    PPD_chain = np.vstack(samples_list)
    return PPD_chain

