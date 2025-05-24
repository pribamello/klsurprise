from operator import neg
import numpy as np
from scipy.stats import norm
from tqdm.auto import tqdm
from joblib import Parallel, delayed

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.linalg import cholesky
import jax_cosmo as jc
from jax import jit
import jax
from jax import vmap
from jax.scipy.special import logsumexp

'''
Implementation for BAO and BAO+BBN likelihood considering the new DESI data release. 
See "DESI 2024 VI: Cosmological Constraints from the Measurements of Baryon Acoustic Oscillations" (https://arxiv.org/abs/2404.03002) for reference.

Note to self:
Because the covariance matrix is fixed, the function log_multivariate_normal can be further optmized by performing Cholesky decomposition outside 
the function and passing the arguments. But as the covariance matrix is small and almost diagonal, performance gains should be marginal and I opt not to do it. 
'''

# filedir = '../Functions/'
domain_full = np.array([
       [0.3, 2.],
       [0.05, 0.5],
       [-0.4, 0.4],
       [-3.0, -0.5],
       [0.010, 0.040]])
domain = np.array([
       [0.3, 1.5],
       [0.01, 0.99],
       [-0.8, 0.8],
       [-3.0, -0.5]])

c    = 2.99792458e5 # speed of light in km/s
TCMB = 2.7255
wg   = 2.47436e-5
Neff = 3.046
wnu  = 0.2271*wg*Neff

# wb gaussian prior
mu_wb = 0.02218
sigma_wb = 0.00055

# data and covariance
z_eff_array = jnp.array([0.3, 0.51, 0.51, 0.71, 0.71, 0.93, 0.93, 1.32, 1.32, 1.49, 2.33, 2.33])
data_array  = jnp.array([7.93, 13.62, 20.98, 16.85, 20.08, 21.71, 17.88, 27.79, 13.82, 26.07, 39.71, 8.52 ])
covariance_matrix = jnp.array([[ 0.0225     ,  0.       ,  0.       ,  0.       ,  0.       ,
         0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
         0.       ,  0.       ],
       [ 0.       ,  0.0625     , -0.0678625,  0.       ,  0.       ,
         0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
         0.       ,  0.       ],
       [ 0.       , -0.0678625,  0.3721     ,  0.       ,  0.       ,
         0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
         0.       ,  0.       ],
       [ 0.       ,  0.       ,  0.       ,  0.1024     , -0.08064  ,
         0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
         0.       ,  0.       ],
       [ 0.       ,  0.       ,  0.       , -0.08064  ,  0.36      ,
         0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
         0.       ,  0.       ],
       [ 0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
         0.0784     , -0.038122 ,  0.       ,  0.       ,  0.       ,
         0.       ,  0.       ],
       [ 0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
        -0.038122 ,  0.1225     ,  0.       ,  0.       ,  0.       ,
         0.       ,  0.       ],
       [ 0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
         0.       ,  0.       ,  0.4761     , -0.128671 ,  0.       ,
         0.       ,  0.       ],
       [ 0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
         0.       ,  0.       , -0.128671 ,  0.1764     ,  0.       ,
         0.       ,  0.       ],
       [ 0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
         0.       ,  0.       ,  0.       ,  0.       ,  0.4489     ,
         0.       ,  0.       ],
       [ 0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
         0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
         0.8836     , -0.0762246],
       [ 0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
         0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
        -0.0762246,  0.0289     ]])


##### Other functions and usefull definitions ######
# https://jax-cosmo.readthedocs.io/en/latest/_modules/jax_cosmo/background.html
@jit
def H(z, h, Om, Ok, w):
    Ez = Om * (1 + z) ** 3 + Ok * (1 + z) ** 2 + (1 - Ok - Om) * (1 + z) ** (3 * (1 + w))
    # Using jnp.where to handle the conditional which is JAX compatible
    res = jnp.where(Ez > 0, 100 * h * jnp.sqrt(Ez), jnp.nan)
    return res

@jit
def w_b(Ob, h):
    return Ob*h**2

@jit
def rs(h, Om, wb): # isso aqui na verdade é rd = rs(zd)
    # Veja https://arxiv.org/pdf/1411.1074.pdf
    # Equation 3.2 from arXiv:1906.11628v2
    # https://arxiv.org/abs/1906.11628
    return 55.154*jnp.exp(-72.3*(wnu + 0.0006)**2)/((Om*h*h)**0.25351*wb**0.12807)

@jit
def Dvrs(z, params):
    # Luca Amendola,eq. 5.53
    h, Om, Ok, w, wb = params
    alist = 1/(1+z)
    cosmo = cosmo = jc.Planck15(h=h, Omega_c=Om-wb/h**2, Omega_k=Ok, w0=w, Omega_b = wb/h**2)
    Dcom = jc.background.transverse_comoving_distance(cosmo, alist)/cosmo.h # o jax cosmo retorn o valor em Mpc/h por isso eu multiplico por h
    Hz = H(z, h, Om, Ok, w)
    
    res = (Dcom**2*c*z/Hz)**(1/3)/rs(h, Om, wb)
    return res

@jit
def Dars(z, params):
    h, Om, Ok, w, wb = params
    alist = 1/(1+z)
    cosmo = cosmo = jc.Planck15(h=h, Omega_c=Om-wb/h**2, Omega_k=Ok, w0=w, Omega_b = wb/h**2)
    Da = jc.background.angular_diameter_distance(cosmo, alist)/cosmo.h # o jax cosmo retorn o valor em Mpc/h por isso eu multiplico por h

    res = Da/rs(h, Om, wb)
    return res

@jit
def Hrs(z, params):
    h, Om, Ok, w, wb = params
    res = c/(H(z, h, Om, Ok, w)*rs(h, Om, wb))
    return jnp.array(res)



#### prior and gauss function #### 
# def log_prior_wb(wb):
    # return -0.5*(wb-0.02166)**2/0.000186**2

@jit
def logGauss1D(x, mu, sigma):
    # log of 1D normal distribution
    return -0.5*(x-mu)**2/sigma**2

@jit
def log_prior_wb(wb):
    return logGauss1D(wb, mu_wb, sigma_wb)


def prior_transform(utheta, flat_domain = None):
    """
    Transforms samples `u` drawn from the unit cube to samples from the domain.
    The first n-1 dimensions use flat priors, and the last dimension uses a Gaussian prior.

    Parameters:
        utheta (array): Array of parameters in the unit cube.
        domain (np.array): The domain of the parameters (h, Om, Ok, w) as a 2D numpy array. If None, standard domain will be used.
    """
    
    domain = flat_domain
    if domain is None:
        domain = np.array([
                      [0.3, 0.9], # h
                      [0.05, 0.7], # Omega_m
                      [-0.8, 0.8], # Omega_k
                      [-3., -0.5]]) # w

    # Parameters of the Gaussian prior for the fifth dimension wb
    mu = mu_wb # mean of the Gaussian
    sigma = sigma_wb # standard deviation of the Gaussian
    ndim = domain.shape[0]+1
    
    # Transform for the flat prior dimensions
    theta = domain[:, 0] + utheta[:ndim-1]*(domain[:, 1] - domain[:, 0])

    # Transform for the last dimension (Gaussian prior)
    theta = np.append(theta, norm.ppf(utheta[-1], loc=mu, scale=sigma))

    return theta


##### Likelihood function ######
def log_prior_wb(wb):
    return -0.5*(wb-mu_wb)**2/sigma_wb**2 - jnp.log(jnp.sqrt(2*jnp.pi)*sigma_wb)

def logGauss1D(x, mu, sigma):
    # log of 1D normal distribution
    return -0.5*(x-mu)**2/sigma**2 - np.log(np.sqrt(2*np.pi)*sigma)

    
def log_multivariate_normal(x, mean, cov):
    """
    Compute the log-likelihood of x for a multivariate normal distribution.
    Args:
    - x: The observed data.
    - mean: The mean of the distribution.
    - cov: The covariance matrix.

    Returns:
    - The log-likelihood of x.
    """
    n = x.shape[0]
    diff = x - mean
    chol_cov = cholesky(cov, lower=True)
    solve = solve_triangular(chol_cov, diff, lower=True)
    log_det_cov = 2 * jnp.sum(jnp.log(jnp.diagonal(chol_cov)))
    log_likelihood = -0.5 * (n * jnp.log(2 * jnp.pi) + log_det_cov + jnp.dot(solve, solve))

    return log_likelihood


@jit
def loglikelihood(params, datavec=data_array, covariance=covariance_matrix):
    """
    Calculate the log-likelihood of Baryon Acoustic Oscillations + Big Bang Nucleosynthesis (BAO+BBN) data based on input cosmological parameters and observed data vector.

    This function computes the theoretical predictions for BAO measurements given a set of cosmological parameters, compares these predictions to observed data, and calculates the log-likelihood of the observed data given the model predictions and their associated uncertainties.

    Parameters:
    - params (jax.numpy.ndarray): An array of cosmological parameters [h, Om, Ok, w, wb], where:
        - h is the Hubble parameter,
        - Om is the matter density parameter,
        - Ok is the curvature density parameter,
        - w is the equation of state parameter,
        - wb is the baryon density parameter.
    - datavec (jax.numpy.ndarray): An array of observed BAO data to compare against the model.

    Returns:
    - float: The log-likelihood of the observed data given the input model parameters.

    The function utilizes JAX for its computations to leverage automatic differentiation and potential acceleration on GPU/TPU hardware. 

    References:
    - The likelihood implementation is based on the methodology described in:
      "DESI 2024 VI: Cosmological Constraints from the Measurements of Baryon Acoustic Oscillations"
    """
    h, Om, Ok, w, wb = params

    # gal bao 
    # rsfid = 147.78 # from https://arxiv.org/pdf/1607.03155.pdf
    c = 299792.458  # Speed of light in km/s
    
    # Cosmological calculations
    z = z_eff_array
    alist = 1/(1+z)
    cosmo = jc.Planck15(h=h, Omega_c=Om-wb/h**2, Omega_k=Ok, w0=w, Omega_b = wb/h**2)
    Dcom = jc.background.transverse_comoving_distance(cosmo, alist)/cosmo.h # o jax cosmo retorn o valor em Mpc/h por isso eu multiplico por h
    Dang = jc.background.angular_diameter_distance(cosmo, alist)/cosmo.h
    Hz = H(z, h, Om, Ok, w)  # Assuming H is defined elsewhere

    # Background quantities
    rsvl = rs(h, Om, wb)  # Assuming rs is defined elsewhere
    Dvrs = (Dcom**2*c*z/Hz)**(1/3) / rsvl
    Dars = Dang / rsvl
    Dcomrs = Dcom/rsvl
    Hrs = c/(Hz*rsvl)
    
    DMDHDV_model_vec = jnp.array([Dvrs[0], Dcomrs[1], Hrs[2], Dcomrs[3], Hrs[4], 
                                Dcomrs[5], Hrs[6], Dcomrs[7], Hrs[8], Dvrs[9], Dcomrs[10], Hrs[11]])
    # observable vec
    observec = DMDHDV_model_vec
    
    # loglikelihood of function
    loglike = log_multivariate_normal(observec, datavec, covariance)
    
    logL_return = jnp.nan_to_num(loglike, nan = -1e+32, neginf = -1e+32)
    return logL_return

@jit
def logP(params, datavec):
    h, Om, Ok, w, wb = params
    try:
        lgl = jnp.nan_to_num(loglikelihood(params, datavec) + log_prior_wb(wb), nan = -1e+32)
    except:
        lgl = -1e+32
    return lgl


######### Marginalized posterior

@jax.jit
def logP_BAO(x):
    return loglikelihood(x, data_array) + log_prior_wb(x[-1])

@jax.jit
def logP_BAO_marginalized(th, domain_wb=domain_full[-1], Nwb=31, logP_BAO=logP_BAO):
    # Ensure the logP_BAO function is provided
    if logP_BAO is None:
        raise ValueError("logP_BAO function must be provided")
    
    # Parameters setup
    h, Om, Ok, w = th
    wb_samples = jnp.linspace(domain_wb[0], domain_wb[1], Nwb)
    params = jnp.stack([jnp.full_like(wb_samples, h),
                        jnp.full_like(wb_samples, Om),
                        jnp.full_like(wb_samples, Ok),
                        jnp.full_like(wb_samples, w),
                        wb_samples], axis=-1)

    # Vectorized log likelihood computation
    log_probabilities = vmap(logP_BAO)(params)
    
    # These lines generate the coefficients used in Simpson's rule. 
    # In Simpson's rule, every second element (starting from the second) is multiplied by 4, and the others (except the first and last) by 2.
    # These are then scaled by 1/3, according to the rule.
    coeffs = jnp.ones(Nwb)
    coeffs = coeffs.at[1:-1:2].set(4)
    coeffs = coeffs.at[2:-1:2].set(2)
    coeffs = coeffs * (1/3)
    
    # Apply Simpson's rule with logsumexp for numerical stability
    log_integrated_prob = logsumexp(log_probabilities + jnp.log(coeffs)) - jnp.log(jnp.sum(coeffs))
    
    return log_integrated_prob

######################################################## Modelos cosmológicos diferentes #######################################################
#################################### Posterior density function after numerically marginalizign wb parameter ################################### 
@jax.jit
def logP_marg_owCDM(th):
    return logP_BAO_marginalized(th)

@jax.jit
def logP_marg_oLCDM(th):
    h, Om, Ok = th
    w = -1
    params = jnp.array([h, Om, Ok, w])
    return logP_BAO_marginalized(params)

@jax.jit
def logP_marg_wCDM(th):
    h, Om, w = th
    Ok = 0
    params = jnp.array([h, Om, Ok, w])
    return logP_BAO_marginalized(params)

@jax.jit
def logP_marg_FLCDM(th):
    h, Om = th
    w = -1
    Ok = 0
    params = jnp.array([h, Om, Ok, w])
    return logP_BAO_marginalized(params)



####################################################################################################
################### For creating the Posterior Predictive Distribution #############################
####################################################################################################

def create_mock(params, noise=False):
    h, Om, Ok, w, wb = params

    c = 299792.458  # Speed of light in km/s
    
    # Cosmological calculations
    z = z_eff_array
    alist = 1/(1+z)
    cosmo = jc.Planck15(h=h, Omega_c=Om-wb/h**2, Omega_k=Ok, w0=w, Omega_b = wb/h**2)
    Dcom = jc.background.transverse_comoving_distance(cosmo, alist)/cosmo.h # o jax cosmo retorn o valor em Mpc/h por isso eu multiplico por h
    Hz = H(z, h, Om, Ok, w)  # Assuming H is defined elsewhere

    # Background quantities
    rsvl = rs(h, Om, wb)  # Assuming rs is defined elsewhere
    Dvrs = (Dcom**2*c*z/Hz)**(1/3) / rsvl
    Dcomrs = Dcom/rsvl
    Hrs = c/(Hz*rsvl)
    
    datavec = jnp.array([Dvrs[0], Dcomrs[1], Hrs[2], Dcomrs[3], Hrs[4], 
                                Dcomrs[5], Hrs[6], Dcomrs[7], Hrs[8], Dvrs[9], Dcomrs[10], Hrs[11]])
    
    if not noise:
        return datavec
    else:
        print("Not implemented...")

def generate_samples(theory_vec, L, nz, sample_size):
    """
    Generate Gaussian samples for a given theory vector.
    """
    z = np.random.normal(size=(sample_size, nz))
    gaussian_samples = theory_vec + np.dot(z, L.T)
    return gaussian_samples

def create_ppd_chain(th1_samples, sample_size=10, n_jobs=4, cov_matrix = covariance_matrix):
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
    D1_th1_samples = np.zeros((th1_samples.shape[0], covariance_matrix.shape[0]))
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