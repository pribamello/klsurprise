import numpy as np
from scipy import special


def sampler(chain, nsample):
    """
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
    """
    index = np.arange(0, len(chain))
    rnd_el = np.random.choice(index, nsample)
    sampled = chain[rnd_el]
    return sampled


def calculate_flat_prior_volume(domain):
    """
    Calculate the volume of a flat (uniform) prior distribution over a specified multidimensional domain.

    Parameters:
    domain (numpy array): A 2D array where each row corresponds to a different dimension of the parameter space.
                        The first column contains the lower bounds, and the second column contains the upper bounds for each dimension.

    Returns:
    float: The calculated volume of the domain, which is the product of the lengths of the intervals for each dimension.
    """
    lengths = np.diff(domain, axis=1).T
    volume = np.prod(lengths)
    return volume


def find_pval(Sdist, S, verbose=0):
    """
    Calculate the p-value from a distribution of surprise values given an observed surprise.

    Parameters:
    - Sdist (ndarray): An array of surprise values from simulations or a distribution.
    - S (float): The observed surprise value for which the p-value is to be calculated.

    Returns:
    - pval (float): The calculated p-value indicating the probability of observing a surprise at least as extreme as S.
    """
    pval = Sdist[Sdist > S].size / Sdist.size
    if verbose > 0:
        print("p-value = {:.1f} %".format(100 * pval))
    return pval


def sigma_discordance(p_value):
    return np.sqrt(2) * special.erfinv(1 - p_value)
