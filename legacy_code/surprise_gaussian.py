import numpy as np

def kld_1d(chain_2, chain_1):
    mean, cov = chain_2.mean(), chain_2.std()

class SurpriseGauss:
    def __init__(self, chain_2, chain_1):
        self.chain_1 = chain_1
        self.mu_1, self.cov_1 = self._infer_stats(chain_1)
        self.chain_2 = chain_2
        self.mu_2, self.cov_2 = self._infer_stats(chain_2)
    
    @staticmethod
    def _infer_stats(chain):
        """Infer mean and covariance from a chain."""
        if isinstance(chain, tuple) and len(chain) == 2: # check if it's (mean, cov)
            mu = chain[0] 
            cov = chain[1] 
            return mu, cov 
        else: # it's a chain, so infeer (mean, cov)
            mu = np.mean(chain, axis=0)
            cov = np.cov(chain, rowvar=False)
        return mu, cov
    
    def calculate_surprise(self, Nsamples = 10000):
        """
        Calculate the surprise statistics between chain_2 and chain_1 - S(p2|p1).
        Nsamples (int) : number of samples to draw from the Surprise distribution. Standard is 10^4
        returns:
            Sval (float) : value of the Surprise statistics
            Sdist (ndarray) : Surprise statistics distribution
        """ 
        Sval = self.surprise((self.mu_2, self.cov_2), (self.mu_1, self.cov_1))
        Sdist = self.S_dist(self.cov_2, self.cov_1, Nsamples)
        return Sval, Sdist
    
    def calculate_kld(self, Nsamples = 10000):
        """
        Calculate the kld between chain_2 and chain_1 - kld(p2|p1).
        Nsamples (int) : number of samples to draw from the Surprise distribution. Standard is 10^4
        returns:
            kld_val (float) : value of the Kullback-Leibler divergence.
            kld_exp (float) : expected value of the Kullback-Leibler divergence.
            kld_dist (ndarray) : Kullback-Leibler divergence distribution.
        """ 
        kld_val = self.kld((self.mu_2, self.cov_2), (self.mu_1, self.cov_1))
        kld_exp = self.expected_kld(self.cov_2, self.cov_1)
        Sdist = self.S_dist(self.cov_2, self.cov_1, Nsamples)
        kld_dist = Sdist+kld_exp
        return kld_val, kld_exp, kld_dist
    
    @staticmethod
    def kld(chain_2, chain_1):
        """
        Compute the Kullback-Leibler Divergence between two multivariate Gaussian distributions.

        Parameters:
            gauss_2 (tuple) : (mean_2, cov_2) where mean_2, cov_2 are numpy arrays
            gauss_1 (tuple) : (mean_1, cov_1) where mean_1, cov_1 are numpy arrays
            
        Returns:
            kld (float): The analytical KLD from the first Gaussian to the second Gaussian.
        """
        mu2, cov2 = SurpriseGauss._infer_stats(chain_2)
        mu1, cov1 = SurpriseGauss._infer_stats(chain_1)
        cov1_inv = np.linalg.inv(cov1)
        k = len(mu2)
        term1 = np.trace(cov1_inv.dot(cov2))
        # term1 = np.trace(np.matmul(cov1_inv, cov2))
        term2 = (mu2 - mu1).T.dot(cov1_inv).dot(mu2 - mu1)
        term3 = -k
        term4 = np.linalg.slogdet(cov1)[1] - np.linalg.slogdet(cov2)[1]
        kld = 0.5 * (term1 + term2 + term3 + term4)
        return kld
    
    @staticmethod    
    def expected_kld(cov2, cov1):
        """
        Calculate a statistical measure based on Equation A38 from Seehars et. al. 2014, 
        "Information Gains from Cosmic Microwave Background Experiments".

        Parameters (optional when used as an instance method):
        - cov2 (numpy.ndarray): A covariance matrix for the second dataset or experiment. 
                                It should be a square matrix.
        - cov1 (numpy.ndarray): A covariance matrix for the first dataset or experiment. 
                                It should be a square matrix and of the same dimension as cov2.

        Returns:
        - <kld> (float): The calculated value of the statistical measure.
        """

        invC1 = np.linalg.inv(cov1)
        term1 = -0.5*(np.linalg.slogdet(cov2)[1] - np.linalg.slogdet(cov1)[1])
        term2 = np.diag(np.matmul(cov2, invC1)).sum()
        return term1 + term2
    
    @staticmethod
    def surprise(chain_2, chain_1):
        """
        Calculate the surprise statistics between two multivariate normal distributions. S(p(th|D2), p(th|D1))

        Parameters:
        - mu2 (ndarray): Mean vector of the first distribution.
        - cov2 (ndarray): Covariance matrix of the first distribution.
        - mu1 (ndarray): Mean vector of the second distribution.
        - cov1 (ndarray): Covariance matrix of the second distribution.

        Returns:
        - S (float): The calculated surprise.
        """
        mu2, cov2 = SurpriseGauss._infer_stats(chain_2)
        mu1, cov1 = SurpriseGauss._infer_stats(chain_1)
        
        dim = mu2.shape[0]
        cov_inv_1 = np.linalg.inv(cov1)
        var = np.identity(dim) + np.matmul(cov2, cov_inv_1)
        tr_var = np.diag(var).sum()
        dMu = mu2-mu1
        S = 0.5*(np.matmul(dMu, np.matmul(cov_inv_1, dMu)) - tr_var)
        return S
    
    @staticmethod
    def S_dist(cov2, cov1, Nsamples = 10000):
        """
        Generate a distribution of the surprise metric by sampling from a chi-square distribution.

        Parameters:
        - cov2 (ndarray): Covariance matrix of the first distribution.
        - cov1 (ndarray): Covariance matrix of the second distribution.
        - Nsamples (int): The number of samples to generate. Default is 10,000.

        Returns:
        - S distribution (ndarray): An array of sampled surprise metrics.
        """
            
        cov_inv_1 = np.linalg.inv(cov1)
        var = np.identity(cov1.shape[0]) + np.matmul(cov2, cov_inv_1)

        eigenvals = np.linalg.eig(var)[0]
        dist = np.zeros(Nsamples)
        for eig in eigenvals: 
            dist += 0.5*eig*(np.random.chisquare(1, Nsamples)-1) # é + ou append?
        return dist 
