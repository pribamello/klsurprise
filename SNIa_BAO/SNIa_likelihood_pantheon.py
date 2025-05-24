import pandas as pd
import numpy as np
import time

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.linalg import cholesky
import jax_cosmo as jc
from jax import jit

from tqdm.auto import tqdm
from joblib import Parallel, delayed

'''
This code is an implementation of the Bayesian quantities necessary to perform an analysis on Pantheon+ data.
Data is available at: https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/
For reference see paper: https://arxiv.org/pdf/2202.04077.pdf
'''

dataname = "Pantheon+SH0ES.dat"
covname  = 'Pantheon+SH0ES_STAT+SYS.cov'

def read_cov_file(filename):
    """
    Reads a covariance matrix file.
    
    Parameters:
    - filename: The name of the file to read.
    
    Returns:
    - A numpy array containing the covariance matrix.
    """
    with open(filename, 'r') as file:
        # Read the first line to get the size of the matrix
        n = int(file.readline().strip())
        # Read the rest of the file in one go - because the first line was read already (i=0), this command starts from the second (i=1)
        data = np.fromfile(file, sep="\n")
    
    # Check if the data length matches n*n
    if data.size != n*n:
        raise ValueError("File size does not match expected dimensions (n*n).")
    
    # Reshape the data into an n x n matrix
    matrix = data.reshape((n, n))
    return matrix

# read data
data=pd.read_csv(dataname, delimiter=' ')
mu_SNIa = jnp.array(data['MU_SH0ES'].to_numpy())
z_SNIa = jnp.array(data['zHD'].to_numpy())

# read covarance file
# print("Reading Covariance Matrix file...")
time0 = time.time()
covariance_pantheon = read_cov_file(covname)
# print("Elapsed time = {:.2f} s\n".format(time.time()-time0))
# perform Cholesky decomposition of covariance matrix
chol_cov = cholesky(covariance_pantheon, lower=True)
log_det_cov = 2 * jnp.sum(jnp.log(jnp.diagonal(chol_cov)))

#################################################### The code #################################################### 
# function that returns a the loglikelihood for a normal distribution with Pantheon+ covariance matrix
@jit
def log_normal_cholesky(x, mean):
    """
    Calculates the log-likelihood of data assuming a normal distribution, using the Cholesky decomposition.
    
    Parameters:
    - x: The data points.
    - mean: The mean values of the distribution.
    
    Returns:
    - The log-likelihood of the data.
    """
    n = x.shape[0]
    diff = x - mean
    solve = solve_triangular(chol_cov, diff, lower=True)
    log_likelihood = -0.5 * (n * jnp.log(2 * jnp.pi) + log_det_cov + jnp.dot(solve, solve))
    return log_likelihood

# function that evaluates theoretical model for SNIa observations given Pantheon+ redshift distribution
@jit
def model_SNIa(u):
    """
    Evaluates the theoretical model for SNIa observations given a set of cosmological parameters.
    
    Parameters:
    - u: The cosmological parameters as a list [h, Omega_c, Omega_k, w0].
    
    Returns:
    - Theoretical distance moduli for SNIa.
    """
    # Define the cosmological model
    cosmo = jc.Planck15(h=u[0], Omega_c = u[1], Omega_b = 0, Omega_k = u[2], w0 = u[3]) # Barion mass doesn't matter for the expansion of the universe
    
    alist = 1/(1+z_SNIa)
    Dcom = jc.background.transverse_comoving_distance(cosmo, alist)/cosmo.h # o jax cosmo retorn o valor em Mpc/h por isso eu multiplico por h
    dlum = (1+z_SNIa)*Dcom 

    mu_theory = 5*jnp.log10(dlum) + 25
    return mu_theory

@jit
def logposterior_owCDM(u, mu_SNIa=mu_SNIa):
    """
    Calculates the log-posterior probability of the owCDM cosmological model given the data on supernovae type Ia.

    This function computes the log-posterior probability by comparing theoretical distance moduli, 
    derived from a given set of cosmological parameters, with observed distance moduli. The comparison is made under 
    the assumption of a normal distribution characterized by the Pantheon+ covariance matrix. 

    Parameters:
    - u (array_like): An array of cosmological parameters used in the model. The parameters are ordered as follows:
        - h (float): Hubble parameter (H0/100), dimensionless.
        - Omega_m (float): Matter density parameter, dimensionless.
        - Omega_k (float): Curvature density parameter, dimensionless.
        - w (float): Equation of state parameter for dark energy.

    - mu_SNIa (array_like, optional): Observed distance moduli of supernovae type Ia. Default is the global `mu_SNIa` array defined earlier in the script.

    Returns:
    - logP (float): The log-posterior probability of the owCDM model given the observed SNIa data.
    """

    theta = jnp.array([u[0], u[1], u[2], u[3]])
    model = model_SNIa(theta)
    logP = jnp.nan_to_num(log_normal_cholesky(model, mu_SNIa), nan=-1e32)
    return logP

@jit
def logposterior_oLCDM(u, mu_SNIa=mu_SNIa):
    """
    Calculates the log-posterior probability of the oLCDM cosmological model given the data on supernovae type Ia.

    This function evaluates the log-posterior probability by comparing the theoretical distance moduli, derived from a 
    set of cosmological parameters, with observed distance moduli. The comparison assumes a normal distribution with 
    the Pantheon+ covariance matrix. The oLCDM model assumes a cosmological constant (Lambda) dark energy component with w = -1.

    Parameters:
    - u (array_like): Cosmological parameters used in the model, specifically:
        - h (float): Hubble parameter (H0/100), dimensionless.
        - Omega_m (float): Matter density parameter, dimensionless.
        - Omega_k (float): Curvature density parameter, dimensionless.
    Note: The dark energy equation of state parameter (w) is fixed at -1.

    - mu_SNIa (array_like, optional): Observed distance moduli of supernovae type Ia. Defaults to the global `mu_SNIa` array.

    Returns:
    - logP (float): Log-posterior probability of the oLCDM model given the observed SNIa data.
    """
    h, Om, Ok = u
    w = -1
    theta = jnp.array([h, Om, Ok, w])
    model = model_SNIa(theta)
    logP = jnp.nan_to_num(log_normal_cholesky(model, mu_SNIa), nan=-1e32)
    return logP

@jit
def logposterior_wCDM(u, mu_SNIa=mu_SNIa):
    """
    Calculates the log-posterior probability of the wCDM cosmological model given the data on supernovae type Ia.

    This function computes the log-posterior probability by comparing the theoretical distance moduli from a set of 
    cosmological parameters with observed distance moduli. It assumes a normal distribution characterized by the Pantheon+ 
    covariance matrix. The wCDM model allows for a varying dark energy equation of state (w) but assumes a flat universe (Omega_k = 0).

    Parameters:
    - u (array_like): Cosmological parameters for the model, comprising:
        - h (float): Hubble parameter (H0/100), dimensionless.
        - Omega_m (float): Matter density parameter, dimensionless.
        - w (float): Equation of state parameter for dark energy.

    - mu_SNIa (array_like, optional): Observed distance moduli of supernovae type Ia. Defaults to the global `mu_SNIa` array.

    Returns:
    - logP (float): Log-posterior probability of the wCDM model given the observed SNIa data.
    """
    h, Om, w = u
    Ok = 0.0
    theta = jnp.array([h, Om, Ok, w])
    model = model_SNIa(theta)
    logP = jnp.nan_to_num(log_normal_cholesky(model, mu_SNIa), nan=-1e32)
    return logP

@jit
def logposterior_LCDM(u, mu_SNIa=mu_SNIa):
    """
    Calculates the log-posterior probability of the ΛCDM (Lambda Cold Dark Matter) cosmological model given the data on supernovae type Ia.

    This function evaluates the log-posterior by comparing theoretical distance moduli from the ΛCDM model, which assumes a flat universe (Ω_k = 0) and a constant equation of state for dark energy (w = -1), with observed distance moduli. The theoretical and observed values are compared under the assumption of a normal distribution characterized by the Pantheon+ covariance matrix.

    Parameters:
    - u (array_like): An array of cosmological parameters used in the model, specifically:
        - h (float): Hubble parameter (H0/100), dimensionless.
        - Om (float): Matter density parameter, dimensionless.

    - mu_SNIa (array_like, optional): Observed distance moduli of supernovae type Ia. The default is the global `mu_SNIa` array defined earlier in the script.

    Returns:
    - logP (float): The log-posterior probability of the ΛCDM model given the observed SNIa data.
    """
    h, Om = u
    Ok = 0.0
    w  = -1.
    theta = jnp.array([h, Om, Ok, w])
    model = model_SNIa(theta)
    logP = jnp.nan_to_num(log_normal_cholesky(model, mu_SNIa), nan=-1e32)
    return logP

def create_mock(fiducial_params, noise = True):
    """
    Generates a mock set of supernovae type Ia distance moduli based on a given set of fiducial cosmological parameters.

    This function creates a mock dataset by first computing theoretical distance moduli for supernovae using the specified fiducial parameters. It then adds simulated observational noise to these theoretical values. The noise is drawn from a multivariate normal distribution characterized by the Pantheon+ covariance matrix, ensuring that the mock data reflects realistic observational uncertainties.

    Parameters:
    - fiducial_params (array_like): The fiducial cosmological parameters used to generate the theoretical model of SNIa observations. These parameters should include:
        - h (float): Hubble parameter (H0/100), dimensionless.
        - Omega_m (float): Matter density parameter, dimensionless.
        - Omega_k (float): Curvature density parameter, dimensionless (for models that consider curvature).
        - w (float): Equation of state parameter for dark energy (for models that allow variation in w).

    Returns:
    - mock_vector (array_like): A mock dataset of distance moduli for supernovae type Ia, incorporating both the theoretical model predictions and simulated observational noise.
    """
    model_vector = model_SNIa(fiducial_params)
    zero_vector  = model_vector*0
    if noise:
        noise_vector = jnp.array(np.random.multivariate_normal(mean=zero_vector, cov=covariance_pantheon))
        mock_vector  = model_vector+noise_vector
    else:
        mock_vector  = model_vector
    return mock_vector

######################################## posterior predictive distribution ########################################

def generate_samples(theory_vec, L, nz, sample_size):
    """
    Generate Gaussian samples for a given theory vector.
    """
    z = np.random.normal(size=(sample_size, nz))
    gaussian_samples = theory_vec + np.dot(z, L.T)
    return gaussian_samples

def create_ppd_chain(th1_samples, sample_size=10, n_jobs=4, cov_matrix = covariance_pantheon):
    """
    Create the Posterior Predictive Distribution (PPD) chain.
    
    Parameters:
    - th1_samples: The samples from the parameter space of the posterior distribution. Given as input to create_mock function. Should be (h, Om, Ok, w)
    - sample_size: The number of samples to be taken from the Gaussian distribution for each theory vector.
    - n_jobs: The number of parallel jobs to run.
    - cov_matrix: Covariance matrix of predicted distribution. Standard matrix is Pantheon covariance.
    
    Returns:
    - The PPD chain as a numpy array.
    """
    print("Evaluating theory from sample distribution p1")
    D1_th1_samples = np.zeros((th1_samples.shape[0], z_SNIa.size))
    for i, th in enumerate(tqdm(th1_samples, desc="Generating theory vectors")):
        D1_th = create_mock(th, noise=False)
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