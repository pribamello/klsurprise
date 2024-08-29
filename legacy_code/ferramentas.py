import numpy as np
import warnings
from tqdm import tqdm 
import optax

"""
This code holds functions to evaluate the Surprise analytically and some other chain plot related helper functions
"""

def FGauss(u, BF, F):
    """
    Compute the logarithm of a multivariate Gaussian probability density function at a point u.

    Parameters:
    - u (ndarray): The point at which to evaluate the log PDF.
    - BF (ndarray): The mean vector of the Gaussian distribution.
    - F (ndarray): The precision matrix (inverse of the covariance matrix) of the Gaussian distribution.

    Returns:
    - logP (float): The log probability density of the multivariate Gaussian at point u.
    """
    du = u - BF
    Fsum = np.einsum('ij, i, j ->', F, du, du)
    logP = -0.5*Fsum
    return logP

def find_pval(Sdist, S):
    """
    Calculate the p-value from a distribution of surprise values given an observed surprise.

    Parameters:
    - Sdist (ndarray): An array of surprise values from simulations or a distribution.
    - S (float): The observed surprise value for which the p-value is to be calculated.

    Returns:
    - pval (float): The calculated p-value indicating the probability of observing a surprise at least as extreme as S.
    """
    pval = Sdist[Sdist > S].size/Sdist.size
    print("p-value = {:.1f} %".format(100*pval))
    return pval

def surprise_stats(dkbl, dkbl_chain):
    """
    Calculate the surprise statistic, its p-value, and related statistics for a given divergence and its distribution.

    Parameters:
    - dkbl (float): The observed Kullback-Leibler divergence.
    - dkbl_chain (ndarray): An array of Kullback-Leibler divergences representing the distribution or chain of divergences.

    Returns:
    - S (float): The calculated surprise statistic.
    - pval (float): The p-value associated with the surprise statistic.
    - expD (float): The expected value of the Kullback-Leibler divergence from the chain.
    - Sdist (ndarray): The distribution of surprise statistics from the chain.
    """
    expD = dkbl_chain.mean()
    S = dkbl - expD
    Sdist = dkbl_chain - expD
    print("Dkbl = {:.2f}".format(dkbl))
    print("<D>  = {:.2f}".format(expD))
    print("S    = {:.2f}".format(S))
    pval = find_pval(Sdist, S)
    return S, pval, expD, Sdist

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


def find_domain(chain, nvar = 0, delta = None):
    if delta is not None:
        domain = np.array([chain.min(axis = 0) - nvar*chain.var(axis = 0) - np.array(delta),
                           chain.max(axis = 0) + nvar*chain.var(axis = 0) + np.array(delta)]).T
    else:
        domain = np.array([chain.min(axis = 0) - nvar*chain.var(axis = 0),
                           chain.max(axis = 0) + nvar*chain.var(axis = 0)]).T
    return domain

def infer_stats(chain):
    """Infer mean and covariance from a chain."""
    mu = np.mean(chain, axis=0)
    cov = np.cov(chain, rowvar=False)
    return mu, cov

def cube_prior(u, bound):
    '''
    Takes a vector of random numbers between 0 and 1 and scales it to fit in a box defined by prior edges
    To be used as starting points p0 for the emcee run  
    '''
    ul = np.zeros(u.shape)
    i = 0
    for b in bound:
        ul[:,i] = (b[1] - b[0])*u[:,i] + b[0]
        i += 1
    return ul

def bounds(u, bound):
    '''
    Creates a function that returns -infinity in case paramter vector u is outside bounds else, returns zero
    '''
    condition = True
    i = 0
    for el in bound:
        condition = condition and (el[0] < u[i] < el[1])
        if condition == False:
            return -np.inf
        i+=1  
    return 0
#############################################################################################3
######### used in BAO PPD
#############################################################################################

def insert_parameters(samples_theta, fix_values, chain_params=['Om', 'Ok', 'w'], param_names = {'h': 0, 'Om': 1, 'Ok': 2, 'w': 3, 'wb': 4}):
    """
    Insert fixed parameter values into samples_theta at the appropriate positions.
    You can use this function to take a chain of points ['Om', 'Ok', 'w'] and transform it
    into a chain of points ['h = 0.7','Om', 'Ok', 'w', 'wb=0.022'] for instance.

    Parameters:
    - samples_theta (ndarray): Array of sampled parameters.
    - fix_values (dict): Dictionary of fixed parameter values with parameter names as keys.
                         ex: {'h':0.7, 'wb':0.02166}
    - chain_params (list): List of parameter names in the order they appear in samples_theta.
    - param_names (dict): Dictionary containing all parameters in the final chain with their 
                          names as keys pointing to their respective positional index in chain.
                          ex: {'h': 0, 'Om': 1, 'Ok': 2, 'w': 3, 'wb': 4}
    Returns:
    - theta_pass (ndarray): New array with both sampled and fixed parameters.
    """
    samples_theta = np.atleast_2d(samples_theta)
    theta_pass = np.zeros((samples_theta.shape[0], 5))

    # Insert sampled parameters
    for i, param in enumerate(chain_params):
        idx = param_names[param]
        theta_pass[:, idx] = samples_theta[:, i]

    # Insert fixed parameters
    for param, value in fix_values.items():
        idx = param_names[param]
        theta_pass[:, idx] = value

    return theta_pass


def process_samples_with_mock(samples_theta_i, create_mock_func, fix_values, chain_params, param_names='BAO'):
    """
    Process samples and create mock data, handling NaN values and inserting fixed parameters.
    This function is an intermediate step into evalluating the Posterior Predictive Distribution (PPD),
    it's used to create D(theta) data and save it once, so it can be used to call the PPD multiple 
    times without running into a big computational expense.

    Parameters:
    - samples_theta_i: theta samples of posterior distribution p(theta|D1).
    - create_mock_func: Function to create mock data from parameters. (BAO.create_mock for instance)
    - fix_values (dict): Dictionary of fixed parameter values.
    - chain_params (list): List of parameter names in the order they appear in the samples.
    - param_names (dict): Dictionary containing all parameters in the final chain with their 
                          names as keys pointing to their respective positional index in chain.
                          ex: {'h': 0, 'Om': 1, 'Ok': 2, 'w': 3, 'wb': 4}
    Returns:
    - D_theta_i (list): List of mock data, excluding NaN results.
    """
    D_theta_i = []

    if param_names=='BAO':
        param_names = {'h': 0, 'Om': 1, 'Ok': 2, 'w': 3, 'wb': 4}
    
    for th in tqdm(samples_theta_i):
        th_with_fixed = insert_parameters(np.array([th]), fix_values, chain_params, param_names = param_names)[0]
        D_th = create_mock_func(th_with_fixed)

        if np.any(np.isnan(D_th)):
            warnings.warn("NaN encountered in result")
        else:
            D_theta_i.append(D_th)

    return np.array(D_theta_i)


###################################################################################################################
def find_mle(results):
    """
    Find the Maximum Likelihood Estimator (MLE) from a dynesty results object.

    Parameters:
        results (object): The results object from a dynesty nested sampling run.

    Returns:
        mle (array): The parameters of the sample with the highest log-likelihood.
    """
    # Extract log-likelihoods
    loglikelihoods = results['logl']

    # Find the index of the maximum log-likelihood
    max_logl_index = np.argmax(loglikelihoods)

    # Extract the corresponding sample
    mle = results['samples'][max_logl_index]

    return mle


############################################ Scipy walker based optimization function ############################################
from scipy import optimize as opt

def optimize_parameters_with_scipy(logP, initial_guess, guess_scale=0.3, num_initial_guesses=8, method='Nelder-Mead', verbose=0
                                  ):
    ndim = initial_guess.size
    
    # Define the objective function to minimize
    def min_fun(theta):
        return -logP(theta)
    
    best_result = None
    
    # Iterate over multiple initial guesses
    for _ in tqdm(range(num_initial_guesses)):
        # Generate a new initial guess by adding Gaussian noise
        current_guess = initial_guess + np.random.normal(size=ndim, scale=guess_scale)
        
        # Use scipy's optimize.minimize with the Nelder-Mead method
        result = opt.minimize(min_fun, current_guess, method=method)
        
        # Update the best result if it's either the first run or an improvement
        if best_result is None or result.fun < best_result.fun:
            best_result = result
            if verbose>1:
                print("Best theta:", result.x)
                print("logP(theta):", logP(result.x))
    
    # Extract the optimized parameters from the best result
    u_opt = best_result.x

    if verbose>0:
        # Print the optimization results
        print("MLE:", u_opt)
        print("logP(MLE):", logP(u_opt))

    # Return the best optimization result for further use if needed
    return best_result

###### for use with scikit-learn gmm 
def minimize_gmm(gmm, Nsamples_init=50000, num_initial_guesses=32, guess_scale=0.1, verbose=1):
    samples  = gmm.sample(Nsamples_init)[0]
    logpdf_arr = gmm.score_samples(samples)
    def logpdf_fun(th):
        th_reshape = np.reshape(th, [1,-1])
        return gmm.score_samples(th_reshape)[0]
    initial_guess = samples[logpdf_arr.max()==logpdf_arr][0]
    best_fit = fe.optimize_parameters_with_scipy(logpdf_fun, initial_guess=initial_guess, 
                                                 num_initial_guesses=num_initial_guesses, 
                                                 guess_scale=guess_scale, verbose=verbose)
    return best_fit

############################################ Gradient based optmization function ############################################

import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm.auto import tqdm

def optimize_parameters(logP, initial_guess, guess_scale=0.3, num_initial_guesses=8, initial_learning_rate=1e-2, final_learning_rate=1e-6, n_steps=2000, verbose=1000):
    
    ndim = initial_guess.size
    @jax.jit
    def min_fun(theta):
        return -logP(theta)
    
    # Initialize parameters for multiple initial guesses
    u_init = jnp.array([initial_guess + np.random.normal(size=ndim, scale=guess_scale) for _ in range(num_initial_guesses)])

    # Define the learning rate schedule
    lr_schedule = optax.linear_schedule(init_value=initial_learning_rate,
                                        end_value=final_learning_rate,
                                        transition_steps=n_steps)

    # Define the optimizer with the learning rate schedule
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(u_init)

    # Vectorize the optimization step function to work on batches
    @jax.jit
    def step(params, opt_state):
        grads = jax.vmap(jax.grad(min_fun))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # Initialize a variable to track the optimal parameters for each initial guess
    u_opt = u_init

    # Optimization loop, now handling batches of parameters
    for iteration in tqdm(range(n_steps)):
        u_init, opt_state = step(u_init, opt_state)
        min_fun_values = jax.vmap(min_fun)(u_init)
        for i in range(num_initial_guesses):
            if min_fun_values[i] < min_fun(u_opt[i]):
                u_opt = u_init.at[i].set(u_init[i])

        # Optionally, print details periodically to monitor convergence
        if verbose is not None:
            if iteration % verbose == 0:
                print_progress(u_opt, logP)

    # Print final parameters and their logP values
    if verbose is not None:
        print("Optimization ended:")
        print_progress(u_opt, logP)
    
    return u_opt

def print_progress(u_opt, logP):
    min_fun_array = -jax.vmap(logP)(u_opt)
    MLE = u_opt[min_fun_array.argmin()]
    print("logP = ", logP(MLE))
    print("theta = ", MLE)
    print("-----------------------------------------------")





############################## minimize gmm ##############################
def logpdf_gmm(th, gmm):
    th_reshape = np.reshape(th, [1,-1])
    return gmm.score_samples(th_reshape)[0]
    
def minimize_gmm(gmm, Nsamples_init=50000, num_initial_guesses=32, guess_scale=0.1, verbose=1):
    samples  = gmm.sample(Nsamples_init)[0]
    logpdf_arr = gmm.score_samples(samples)
    def logpdf_fun(th):
        th_reshape = np.reshape(th, [1,-1])
        return gmm.score_samples(th_reshape)[0]
    initial_guess = samples[logpdf_arr.max()==logpdf_arr][0]
    best_fit = fe.optimize_parameters_with_scipy(logpdf_fun, initial_guess=initial_guess, 
                                                 num_initial_guesses=num_initial_guesses, 
                                                 guess_scale=guess_scale, verbose=verbose)
    

def marginalize_MCMC_chain(chain, marginalize_params, param_names):
    dict_samples = {}

    # create a dictionary from chain with param_names as keys
    for i, pname in enumerate(param_names):
        dict_samples[pname] = chain.T[i] 

    # marginalize over parameters in marginalize_params
    for marg_name in marginalize_params: 
        dict_samples.pop(marg_name)

    marginalized_chain = np.array(list(dict_samples.values())).T
    
    return marginalized_chain


